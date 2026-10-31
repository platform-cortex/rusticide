// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Greedy parity with llama.cpp on the canonical `TinyLlama` and Qwen2.5
//! models. Skips unless `RUSTICIDE_MODELS` names a directory holding the
//! model files, and in unoptimized builds, where a real model is too slow.

use std::error::Error;
use std::path::PathBuf;

use rusticide_io::GgufFile;
use rusticide_model::{Model, Registry};
use rusticide_tokenizer::Tokenizer;
use serde::Deserialize;

const TINYLLAMA: &str = include_str!("fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.json");
const QWEN2: &str = include_str!("fixtures/qwen2.5-0.5b-instruct-q4_k_m.json");
const NEAR_TIE: f32 = 0.1;
const N_PREDICT: usize = 32;

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
    if case.generated_tokens.len() < N_PREDICT {
        let ((top1, _), _) = top_two(&logits);
        assert!(
            tokenizer.is_end_of_generation(u32::try_from(top1)?),
            "{:?}: llama.cpp stopped, rusticide continues with {top1}",
            case.prompt
        );
    }
    println!(
        "{:?}: all {} tokens match",
        case.prompt,
        case.generated_tokens.len()
    );
    Ok(())
}

fn models_dir() -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        println!("skipping: reference tests need --release");
        return None;
    }
    let models = std::env::var_os("RUSTICIDE_MODELS");
    if models.is_none() {
        println!("skipping: RUSTICIDE_MODELS is not set");
    }
    models.map(PathBuf::from)
}

fn check_fixture(fixture: &str) -> TestResult {
    let Some(models) = models_dir() else {
        return Ok(());
    };
    let fixture: Fixture = serde_json::from_str(fixture)?;
    let source = GgufFile::load(models.join(&fixture.model.file))?;
    let tokenizer = Tokenizer::from_gguf(source.metadata())?;
    let registry = Registry::new()
        .with("llama", rusticide_llama::load)
        .with("qwen2", rusticide_llama::load_qwen2);
    let mut model = registry.load(&source)?;
    for case in &fixture.cases {
        check_case(model.as_mut(), &tokenizer, case)?;
    }
    Ok(())
}

#[test]
fn tinyllama_matches_llama_cpp_greedy() -> TestResult {
    check_fixture(TINYLLAMA)
}

#[test]
fn qwen2_matches_llama_cpp_greedy() -> TestResult {
    check_fixture(QWEN2)
}
