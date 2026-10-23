// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{self, Write};
use std::time::Duration;

use rusticide_io::GGUFValue;
use rusticide_llama::testing::{TinySource, tiny_source};

use super::{RunArgs, run, run_source, tokens_per_second};

fn args(seed: Option<u64>) -> RunArgs {
    RunArgs {
        model: "unused.gguf".into(),
        prompt: "x".to_owned(),
        max_tokens: 4,
        temperature: 0.0,
        top_k: 40,
        top_p: 0.95,
        seed,
    }
}

fn generate_with(args: &RunArgs) -> (String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    run_source(&tiny_source(), args, &mut out, &mut err).unwrap();
    (
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

struct Broken;

impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("closed"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn the_broken_stream_flushes_cleanly() {
    assert!(Broken.flush().is_ok());
}

#[test]
fn streams_text_from_a_tiny_model() {
    let (out, err) = generate_with(&args(Some(1)));
    assert_ne!(out, "");
    assert!(err.starts_with("prompt: "), "{err}");
    assert!(err.contains(" tokens, generated: 4 tokens, "), "{err}");
    assert!(err.ends_with(" tokens/s\n"), "{err}");
}

#[test]
fn is_deterministic_with_a_seed() {
    let sampled = RunArgs {
        temperature: 0.8,
        ..args(Some(7))
    };
    assert_eq!(generate_with(&sampled).0, generate_with(&sampled).0);
}

#[test]
fn derives_a_seed_from_the_clock_when_absent() {
    assert_ne!(generate_with(&args(None)).0, "");
}

#[test]
fn reports_an_unknown_architecture() {
    let source = TinySource::new()
        .with_metadata("general.architecture", GGUFValue::String("gptj".to_owned()))
        .build();
    let error = run_source(&source, &args(Some(1)), &mut Vec::new(), &mut Vec::new()).unwrap_err();
    assert_eq!(
        format!("{error:#}"),
        "unknown architecture gptj; supported: llama"
    );
}

#[test]
fn reports_a_tokenizer_failure() {
    let source = TinySource::new()
        .without_metadata("tokenizer.ggml.tokens")
        .build();
    assert!(run_source(&source, &args(Some(1)), &mut Vec::new(), &mut Vec::new()).is_err());
}

#[test]
fn reports_a_failing_output_stream() {
    let error =
        run_source(&tiny_source(), &args(Some(1)), &mut Broken, &mut Vec::new()).unwrap_err();
    assert_eq!(format!("{error:#}"), "closed");
}

#[test]
fn reports_a_failing_error_stream() {
    let error =
        run_source(&tiny_source(), &args(Some(1)), &mut Vec::new(), &mut Broken).unwrap_err();
    assert_eq!(format!("{error:#}"), "closed");
}

#[test]
fn reports_a_missing_file_with_its_path() {
    let missing = RunArgs {
        model: "/nonexistent.gguf".into(),
        ..args(Some(1))
    };
    let error = run(&missing, &mut Vec::new(), &mut Vec::new()).unwrap_err();
    assert!(
        format!("{error:#}").starts_with("cannot load /nonexistent.gguf: "),
        "{error:#}"
    );
}

#[test]
fn rate_is_tokens_over_seconds() {
    assert!((tokens_per_second(10, Duration::from_secs(4)) - 2.5).abs() < f64::EPSILON);
}

#[test]
fn rate_is_zero_when_no_time_elapsed() {
    assert!(tokens_per_second(10, Duration::ZERO).abs() < f64::EPSILON);
}
