// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;
use std::time::Duration;

use rusticide_io::GGUFValue;
use rusticide_llama::testing::{TinySource, tiny_qwen2_source, tiny_source};

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

#[test]
fn streams_text_from_a_tiny_model() {
    let (out, err) = generate_with(&args(Some(1)));
    assert_ne!(out, "");
    assert!(err.starts_with("\nprompt: "), "{err}");
    assert!(err.contains(" tokens, generated: 4 tokens, "), "{err}");
    assert!(err.ends_with(" tokens/s\n"), "{err}");
}

#[test]
fn run_streams_text_from_a_tiny_qwen2_model() {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    run_source(&tiny_qwen2_source(), &args(Some(1)), &mut out, &mut err).unwrap();
    assert_ne!(String::from_utf8(out).unwrap(), "");
    assert!(
        String::from_utf8(err)
            .unwrap()
            .contains("generated: 4 tokens")
    );
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
        "unknown architecture gptj; supported: llama, qwen2"
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
    let error = run_source(
        &tiny_source(),
        &args(Some(1)),
        &mut Recorder::broken(&Log::default()),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert_eq!(format!("{error:#}"), "closed");
}

#[test]
fn reports_a_failing_error_stream() {
    let error = run_source(
        &tiny_source(),
        &args(Some(1)),
        &mut Vec::new(),
        &mut Recorder::broken(&Log::default()),
    )
    .unwrap_err();
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

type Log = Rc<RefCell<Vec<String>>>;

struct Recorder {
    name: &'static str,
    log: Log,
    fail_write: bool,
    fail_flush_at: Option<usize>,
    flushes: usize,
}

impl Recorder {
    fn new(name: &'static str, log: &Log) -> Self {
        Self {
            name,
            log: Rc::clone(log),
            fail_write: false,
            fail_flush_at: None,
            flushes: 0,
        }
    }

    fn broken(log: &Log) -> Self {
        Self {
            fail_write: true,
            ..Self::new("broken", log)
        }
    }
}

impl Write for Recorder {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.fail_write {
            return Err(io::Error::other("closed"));
        }
        let text = String::from_utf8_lossy(bytes);
        self.log
            .borrow_mut()
            .push(format!("{} write {text}", self.name));
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flushes += 1;
        self.log.borrow_mut().push(format!("{} flush", self.name));
        if self.fail_flush_at == Some(self.flushes) {
            return Err(io::Error::other("flush closed"));
        }
        Ok(())
    }
}

fn recorded_run(out: &mut Recorder, err: &mut Recorder) -> anyhow::Result<()> {
    run_source(&tiny_source(), &args(Some(1)), out, err)
}

#[test]
fn flushes_every_chunk_and_before_the_stats_line() {
    let log = Log::default();
    let (mut out, mut err) = (Recorder::new("out", &log), Recorder::new("err", &log));
    recorded_run(&mut out, &mut err).unwrap();
    let events = log.borrow();
    let stats = events
        .iter()
        .position(|event| event.starts_with("err write"))
        .unwrap();
    assert!(
        events[stats].starts_with("err write \nprompt: "),
        "{events:?}"
    );
    assert_eq!(events[stats - 1], "out flush");
    for (index, event) in events[..stats].iter().enumerate() {
        if event.starts_with("out write") {
            assert_eq!(events[index + 1], "out flush", "{events:?}");
        }
    }
}

#[test]
fn reports_a_failing_chunk_flush() {
    let log = Log::default();
    let (mut out, mut err) = (Recorder::new("out", &log), Recorder::new("err", &log));
    out.fail_flush_at = Some(1);
    let error = recorded_run(&mut out, &mut err).unwrap_err();
    assert_eq!(format!("{error:#}"), "flush closed");
}

#[test]
fn reports_a_failing_final_flush() {
    let log = Log::default();
    let (mut out, mut err) = (Recorder::new("out", &log), Recorder::new("err", &log));
    recorded_run(&mut out, &mut err).unwrap();
    let total = out.flushes;
    let (mut out, mut err) = (Recorder::new("out", &log), Recorder::new("err", &log));
    out.fail_flush_at = Some(total);
    let error = recorded_run(&mut out, &mut err).unwrap_err();
    assert_eq!(format!("{error:#}"), "flush closed");
}
