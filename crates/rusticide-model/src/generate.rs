// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use rusticide_core::Tensor;
use rusticide_sampling::{Sampler, SamplingParams};
use rusticide_tokenizer::Tokenizer;

use crate::error::{ModelError, Result};
use crate::model::Model;
use crate::stream::TextStream;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub params: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub elapsed: Duration,
}

/// Encodes `prompt`, fills the model's context with it and streams the
/// generated text through `emit` until EOS, `max_tokens` or the context end.
///
/// # Errors
///
/// `ContextFull` when the prompt alone fills the context, `InvalidConfig`
/// when it encodes to no tokens, and any tokenizer, sampling, model or
/// `emit` error.
pub fn generate(
    model: &mut dyn Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    sampler: &mut Sampler,
    emit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<Generation> {
    let started = Instant::now();
    let prompt_tokens = tokenizer.encode(prompt)?;
    let limit = model.context_length();
    if prompt_tokens.len() >= limit {
        return Err(ModelError::ContextFull { limit });
    }
    let logits = prefill(model, &prompt_tokens)?;
    let generated_tokens = decode_loop(
        model,
        tokenizer,
        options,
        sampler,
        emit,
        logits,
        prompt_tokens.len(),
    )?;
    Ok(Generation {
        prompt_tokens,
        generated_tokens,
        elapsed: started.elapsed(),
    })
}

fn prefill(model: &mut dyn Model, tokens: &[u32]) -> Result<Vec<f32>> {
    model.reset();
    let mut last = None;
    for (position, &token) in tokens.iter().enumerate() {
        last = Some(model.forward(token, position)?);
    }
    last.ok_or(ModelError::InvalidConfig {
        reason: "prompt encodes to no tokens",
    })
}

fn decode_loop(
    model: &mut dyn Model,
    tokenizer: &Tokenizer,
    options: &GenerateOptions,
    sampler: &mut Sampler,
    emit: &mut dyn FnMut(&str) -> Result<()>,
    mut logits: Vec<f32>,
    mut position: usize,
) -> Result<Vec<u32>> {
    let mut generated = Vec::new();
    let mut stream = TextStream::default();
    for _ in 0..options.max_tokens {
        let next = pick(sampler, &logits, &options.params)?;
        if next == tokenizer.eos_id() {
            break;
        }
        generated.push(next);
        let text = tokenizer.decode(&generated)?;
        emit_non_empty(emit, stream.advance(&text))?;
        if position == model.context_length() {
            break;
        }
        logits = model.forward(next, position)?;
        position += 1;
    }
    let text = tokenizer.decode(&generated)?;
    emit_non_empty(emit, stream.finish(&text))?;
    Ok(generated)
}

fn emit_non_empty(emit: &mut dyn FnMut(&str) -> Result<()>, text: &str) -> Result<()> {
    if text.is_empty() { Ok(()) } else { emit(text) }
}

fn pick(sampler: &mut Sampler, logits: &[f32], params: &SamplingParams) -> Result<u32> {
    let tensor = Tensor::from_slice(logits, &[logits.len()])?;
    let token = if params.temperature == 0.0 {
        sampler.sample_greedy(&tensor)?
    } else {
        sampler.sample(&tensor, params)?
    };
    Ok(token)
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
