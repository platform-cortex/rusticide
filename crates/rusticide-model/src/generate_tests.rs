// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};
use rusticide_sampling::{Sampler, SamplingParams};
use rusticide_tokenizer::{Tokenizer, TokenizerModel};

use super::{GenerateOptions, generate};
use crate::error::{ModelError, Result};
use crate::model::Model;

const BOS: u32 = 0;
const EOS: u32 = 1;
const PROMPT: u32 = 2;
const B: u32 = 3;
const C: u32 = 4;
const REPLACEMENT: u32 = 5;
const ENDOFTEXT: u32 = 6;

struct ScriptedModel {
    peaks: Vec<u32>,
    context: usize,
    forwards: Vec<(u32, usize)>,
    resets: usize,
}

impl ScriptedModel {
    fn new(peaks: Vec<u32>, context: usize) -> Self {
        Self {
            peaks,
            context,
            forwards: Vec::new(),
            resets: 0,
        }
    }
}

impl Model for ScriptedModel {
    fn vocab_size(&self) -> usize {
        7
    }

    fn context_length(&self) -> usize {
        self.context
    }

    fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        self.forwards.push((token, position));
        let mut logits = vec![0.0; self.vocab_size()];
        let peak = self.peaks[position.min(self.peaks.len() - 1)];
        logits[usize::try_from(peak).unwrap()] = 1.0;
        Ok(logits)
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

fn tokenizer(add_bos: bool) -> Tokenizer {
    let vocab = vec![
        "<s>",
        "</s>",
        "\u{2581}a",
        "b",
        "c",
        "\u{fffd}",
        "<|endoftext|>",
    ];
    Tokenizer::new(TokenizerModel::LLaMA, vocab, None, BOS, EOS, add_bos)
}

fn tokenizer_with_end_texts() -> Tokenizer {
    let tokens = [
        "<s>",
        "</s>",
        "\u{2581}a",
        "b",
        "c",
        "\u{fffd}",
        "<|endoftext|>",
    ]
    .iter()
    .map(|t| GGUFValue::String((*t).to_owned()))
    .collect();
    let metadata: Metadata = [
        (
            "tokenizer.ggml.model".to_owned(),
            GGUFValue::String("llama".to_owned()),
        ),
        ("tokenizer.ggml.tokens".to_owned(), GGUFValue::Array(tokens)),
    ]
    .into_iter()
    .collect();
    Tokenizer::from_gguf(&metadata).unwrap()
}

fn options(max_tokens: usize, temperature: f32) -> GenerateOptions {
    let params = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };
    GenerateOptions { max_tokens, params }
}

fn run(
    model: &mut ScriptedModel,
    prompt: &str,
    options: &GenerateOptions,
) -> (Result<super::Generation>, Vec<String>) {
    let mut pieces = Vec::new();
    let mut emit = |text: &str| {
        pieces.push(text.to_owned());
        Ok(())
    };
    let result = generate(
        model,
        &tokenizer(true),
        prompt,
        options,
        &mut Sampler::new(7),
        &mut emit,
    );
    (result, pieces)
}

#[test]
fn greedy_generation_follows_the_argmax() {
    let mut model = ScriptedModel::new(vec![B, B, C, EOS], 16);
    let (result, pieces) = run(&mut model, "a", &options(8, 0.0));
    let generation = result.unwrap();
    assert_eq!(generation.prompt_tokens, [BOS, PROMPT]);
    assert_eq!(generation.generated_tokens, [B, C]);
    assert_eq!(pieces, ["b", "c"]);
    assert_eq!(model.resets, 1);
    assert_eq!(model.forwards, [(BOS, 0), (PROMPT, 1), (B, 2), (C, 3)]);
}

#[test]
fn sampling_temperature_still_follows_a_dominant_logit() {
    let mut model = ScriptedModel::new(vec![B], 16);
    let (result, pieces) = run(&mut model, "a", &options(1, 0.8));
    assert_eq!(result.unwrap().generated_tokens.len(), 1);
    assert_eq!(pieces.len(), 1);
}

#[test]
fn generation_stops_at_eos_without_emitting_it() {
    let mut model = ScriptedModel::new(vec![B, B, EOS], 16);
    let (result, pieces) = run(&mut model, "a", &options(8, 0.0));
    assert_eq!(result.unwrap().generated_tokens, [B]);
    assert_eq!(pieces, ["b"]);
}

#[test]
fn generation_stops_at_any_end_of_generation_token() {
    let mut model = ScriptedModel::new(vec![B, B, ENDOFTEXT], 16);
    let mut pieces = Vec::new();
    let mut emit = |text: &str| {
        pieces.push(text.to_owned());
        Ok(())
    };
    let result = generate(
        &mut model,
        &tokenizer_with_end_texts(),
        "a",
        &options(8, 0.0),
        &mut Sampler::new(7),
        &mut emit,
    );
    assert_eq!(result.unwrap().generated_tokens, [B]);
    assert_eq!(pieces, ["b"]);
}

#[test]
fn generation_stops_after_max_tokens() {
    let mut model = ScriptedModel::new(vec![B], 16);
    let (result, pieces) = run(&mut model, "a", &options(3, 0.0));
    assert_eq!(result.unwrap().generated_tokens, [B, B, B]);
    assert_eq!(pieces, ["b", "b", "b"]);
}

#[test]
fn prompt_filling_the_context_is_context_full() {
    let mut model = ScriptedModel::new(vec![B], 2);
    let (result, pieces) = run(&mut model, "a", &options(3, 0.0));
    assert!(matches!(result, Err(ModelError::ContextFull { limit: 2 })));
    assert_eq!(model.forwards, [] as [(u32, usize); 0]);
    assert_eq!(pieces, [] as [String; 0]);
}

#[test]
fn generation_stops_at_the_context_length() {
    let mut model = ScriptedModel::new(vec![B], 3);
    let (result, _) = run(&mut model, "a", &options(10, 0.0));
    assert_eq!(result.unwrap().generated_tokens, [B, B]);
    assert_eq!(model.forwards, [(BOS, 0), (PROMPT, 1), (B, 2)]);
}

#[test]
fn empty_prompt_generates_from_bos() {
    let mut model = ScriptedModel::new(vec![B], 16);
    let (result, _) = run(&mut model, "", &options(1, 0.0));
    let generation = result.unwrap();
    assert_eq!(generation.prompt_tokens, [BOS]);
    assert_eq!(generation.generated_tokens, [B]);
}

#[test]
fn a_prompt_encoding_to_no_tokens_is_invalid() {
    let mut model = ScriptedModel::new(vec![B], 16);
    let mut emit = |_: &str| Ok(());
    let result = generate(
        &mut model,
        &tokenizer(false),
        "",
        &options(1, 0.0),
        &mut Sampler::new(7),
        &mut emit,
    );
    assert!(matches!(result, Err(ModelError::InvalidConfig { .. })));
    assert_eq!(model.forwards, [] as [(u32, usize); 0]);
}

#[test]
fn emit_errors_propagate() {
    let mut model = ScriptedModel::new(vec![B], 16);
    let mut emit = |_: &str| Err(ModelError::ContextFull { limit: 99 });
    let result = generate(
        &mut model,
        &tokenizer(true),
        "a",
        &options(3, 0.0),
        &mut Sampler::new(7),
        &mut emit,
    );
    assert!(matches!(result, Err(ModelError::ContextFull { limit: 99 })));
}

#[test]
fn an_unfinished_character_is_emitted_when_generation_ends() {
    let mut model = ScriptedModel::new(vec![B, B, REPLACEMENT], 16);
    let (result, pieces) = run(&mut model, "a", &options(2, 0.0));
    assert_eq!(result.unwrap().generated_tokens, [B, REPLACEMENT]);
    assert_eq!(pieces, ["b", "\u{fffd}"]);
}
