// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use super::Plan;
use crate::BenchError;

const PLAN: Plan = Plan {
    runs: 3,
    budget: Duration::from_millis(1),
};

#[test]
fn sample_discards_the_warm_up_run() {
    let mut calls = 0.0;
    let sample = PLAN
        .sample(|| {
            calls += 1.0;
            Ok(calls)
        })
        .unwrap();
    assert_eq!(calls, 4.0);
    assert_eq!(sample.median, 3.0);
}

#[test]
fn sample_propagates_a_failing_run() {
    let result = PLAN.sample(|| Err(BenchError::ZeroWork));
    assert!(matches!(result, Err(BenchError::ZeroWork)));
}

#[test]
fn sample_of_zero_runs_is_an_error() {
    let plan = Plan { runs: 0, ..PLAN };
    assert!(matches!(plan.sample(|| Ok(1.0)), Err(BenchError::NoRuns)));
}
