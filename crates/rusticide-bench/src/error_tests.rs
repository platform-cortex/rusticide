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
fn json_errors_convert() {
    let parse = serde_json::from_str::<u8>("x").unwrap_err();
    let error = BenchError::from(parse);
    assert!(matches!(error, BenchError::Json(_)));
    assert!(error.to_string().starts_with("invalid json: "));
}

#[test]
fn not_finite_message() {
    assert_eq!(
        BenchError::NotFinite.to_string(),
        "throughput is not a finite positive number"
    );
}
