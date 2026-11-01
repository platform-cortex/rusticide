// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{Baseline, E2E_TOLERANCE, Expected, KERNEL_TOLERANCE};
use crate::{BenchError, Results, Sample};

fn sample(best: f64) -> Sample {
    Sample {
        best,
        median: best * 0.5,
        spread: 0.9,
    }
}

fn session(machine: &str, entries: &[(&str, f64)]) -> Results {
    Results {
        machine: machine.to_string(),
        metrics: entries
            .iter()
            .map(|(name, best)| ((*name).to_string(), sample(*best)))
            .collect(),
    }
}

#[test]
fn baseline_is_the_median_of_the_session_bests_with_a_tolerance_per_kind() {
    let sessions = [
        session("box", &[("kernel/Q4_0/row", 10.0), ("e2e/q/decode", 20.0)]),
        session("box", &[("kernel/Q4_0/row", 30.0), ("e2e/q/decode", 22.0)]),
        session("box", &[("kernel/Q4_0/row", 11.0), ("e2e/q/decode", 90.0)]),
    ];
    let baseline = Baseline::from_sessions(&sessions).unwrap();
    assert_eq!(baseline.machine, "box");
    assert_eq!(
        baseline.metrics["kernel/Q4_0/row"],
        Expected {
            best: 11.0,
            tolerance: KERNEL_TOLERANCE
        }
    );
    assert_eq!(
        baseline.metrics["e2e/q/decode"],
        Expected {
            best: 22.0,
            tolerance: E2E_TOLERANCE
        }
    );
}

#[test]
fn a_single_session_is_its_own_baseline() {
    let baseline = Baseline::from_sessions(&[session("box", &[("e2e/q/prefill", 5.0)])]).unwrap();
    assert_eq!(baseline.metrics["e2e/q/prefill"].best, 5.0);
}

#[test]
fn sessions_from_two_machines_are_rejected() {
    let sessions = [session("a", &[("m", 1.0)]), session("b", &[("m", 1.0)])];
    assert!(matches!(
        Baseline::from_sessions(&sessions),
        Err(BenchError::WrongMachine { expected, got }) if expected == "a" && got == "b"
    ));
}

#[test]
fn a_metric_missing_from_a_session_is_rejected() {
    let sessions = [
        session("box", &[("m", 1.0), ("n", 2.0)]),
        session("box", &[("m", 1.0)]),
    ];
    assert!(matches!(
        Baseline::from_sessions(&sessions),
        Err(BenchError::MissingInSession { metric }) if metric == "n"
    ));
}

#[test]
fn no_sessions_is_an_error() {
    assert!(matches!(
        Baseline::from_sessions(&[]),
        Err(BenchError::NoRuns)
    ));
}

#[test]
fn baseline_round_trip_through_json() {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "a".to_string(),
        Expected {
            best: 1.5,
            tolerance: 0.05,
        },
    );
    let original = Baseline {
        machine: "box".to_string(),
        metrics,
    };
    let text = serde_json::to_string(&original).unwrap();
    assert_eq!(serde_json::from_str::<Baseline>(&text).unwrap(), original);
}
