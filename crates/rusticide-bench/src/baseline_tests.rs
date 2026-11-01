// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{Baseline, E2E_TOLERANCE, Expected, KERNEL_TOLERANCE};
use crate::{Results, Sample};

fn sample(best: f64) -> Sample {
    Sample {
        best,
        median: best * 0.5,
        spread: 0.9,
    }
}

#[test]
fn baseline_records_the_best_with_a_tolerance_per_metric_kind() {
    let mut metrics = BTreeMap::new();
    metrics.insert("kernel/Q4_0/row".to_string(), sample(10.0));
    metrics.insert("e2e/qwen2.5/decode".to_string(), sample(20.0));
    metrics.insert("e2e/qwen2.5/prefill".to_string(), sample(30.0));
    let results = Results {
        machine: "box".to_string(),
        metrics,
    };
    let baseline = Baseline::from_results(&results);
    assert_eq!(baseline.machine, "box");
    assert_eq!(
        baseline.metrics["kernel/Q4_0/row"],
        Expected {
            best: 10.0,
            tolerance: KERNEL_TOLERANCE
        }
    );
    assert_eq!(
        baseline.metrics["e2e/qwen2.5/decode"],
        Expected {
            best: 20.0,
            tolerance: E2E_TOLERANCE
        }
    );
    assert_eq!(
        baseline.metrics["e2e/qwen2.5/prefill"].tolerance,
        E2E_TOLERANCE
    );
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
