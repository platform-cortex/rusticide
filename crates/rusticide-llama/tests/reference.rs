// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Greedy parity with llama.cpp on the canonical `TinyLlama` model. Skips
//! unless `RUSTICIDE_MODELS` names a directory holding the model file.

use std::error::Error;
use std::path::PathBuf;

use rusticide_io::GgufFile;
use rusticide_model::Model;
use rusticide_tokenizer::Tokenizer;
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.json");
const NEAR_TIE: f32 = 0.1;

type TestResult = Result<(), Box<dyn Error>>;

#[derive(Deserialize)]
struct Fixture {
    model: ModelInfo,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ModelInfo {
    file: String,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

fn top_two(logits: &[f32]) -> ((usize, f32), f32) {
    let mut best = (0, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > best.1 {
            second = best.1;
            best = (index, logit);
        } else if logit > second {
            second = logit;
        }
    }
    (best, second)
}

fn check_case(model: &mut dyn Model, tokenizer: &Tokenizer, case: &Case) -> TestResult {
    assert_eq!(
        tokenizer.encode(&case.prompt)?,
        case.prompt_tokens,
        "prompt tokens for {:?}",
        case.prompt
    );
    model.reset();
    let mut logits = Vec::new();
    for (position, &token) in case.prompt_tokens.iter().enumerate() {
        logits = model.forward(token, position)?;
    }
    let positions = case.prompt_tokens.len()..;
    for (position, (step, &expected)) in positions.zip(case.generated_tokens.iter().enumerate()) {
        let ((top1, top1_logit), top2_logit) = top_two(&logits);
        if top1 != usize::try_from(expected)? {
            let margin = top1_logit - top2_logit;
            assert!(
                margin < NEAR_TIE,
                "{:?}: step {step} got {top1}, llama.cpp {expected}, margin {margin}",
                case.prompt
            );
            println!(
                "{:?}: near tie at step {step} (margin {margin}), stopping",
                case.prompt
            );
            return Ok(());
        }
        logits = model.forward(expected, position)?;
    }
    println!(
        "{:?}: all {} tokens match",
        case.prompt,
        case.generated_tokens.len()
    );
    Ok(())
}

#[test]
fn matches_llama_cpp_greedy() {
    let Some(models) = std::env::var_os("RUSTICIDE_MODELS") else {
        println!("skipping: RUSTICIDE_MODELS is not set");
        return;
    };
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    let path = PathBuf::from(models).join(&fixture.model.file);
    let source = GgufFile::load(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(source.metadata()).unwrap();
    let mut model = rusticide_llama::load(&source).unwrap();
    for case in &fixture.cases {
        check_case(model.as_mut(), &tokenizer, case).unwrap();
    }
}
