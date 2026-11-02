// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::BenchError;

#[test]
fn messages_name_the_failure() {
    assert_eq!(BenchError::NoRuns.to_string(), "no runs to summarize");
    assert_eq!(BenchError::ZeroElapsed.to_string(), "elapsed time is zero");
    assert_eq!(BenchError::ZeroWork.to_string(), "work done is zero");
    let wrong = BenchError::WrongMachine {
        expected: "a".into(),
        got: "b".into(),
    };
    assert_eq!(
        wrong.to_string(),
        "baseline was recorded on a, results come from b"
    );
}

#[test]
fn other_adapter_message_names_both() {
    let error = BenchError::OtherAdapter {
        expected: String::new(),
        got: "neon".into(),
    };
    assert_eq!(
        error.to_string(),
        r#"baseline adapter "" differs from the results' "neon""#
    );
}

#[test]
fn missing_metric_message_names_it() {
    let error = BenchError::MissingInSession {
        metric: "e2e/x".into(),
    };
    assert_eq!(error.to_string(), "metric e2e/x is missing from a session");
}

#[test]
fn not_finite_message() {
    assert_eq!(
        BenchError::NotFinite.to_string(),
        "throughput is not a finite positive number"
    );
}
