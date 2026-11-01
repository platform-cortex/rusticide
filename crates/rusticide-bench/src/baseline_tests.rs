// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{Baseline, Expected, TOLERANCE_FLOOR};
use crate::{Results, Sample};

#[test]
fn baseline_tolerance_is_the_spread_with_a_floor() {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "quiet".to_string(),
        Sample {
            median: 10.0,
            spread: 0.02,
        },
    );
    metrics.insert(
        "noisy".to_string(),
        Sample {
            median: 20.0,
            spread: 0.12,
        },
    );
    let results = Results {
        machine: "box".to_string(),
        metrics,
    };
    let baseline = Baseline::from_results(&results);
    assert_eq!(baseline.machine, "box");
    assert_eq!(
        baseline.metrics["quiet"],
        Expected {
            median: 10.0,
            tolerance: TOLERANCE_FLOOR
        }
    );
    assert_eq!(
        baseline.metrics["noisy"],
        Expected {
            median: 20.0,
            tolerance: 0.12
        }
    );
}

#[test]
fn baseline_round_trip_through_json() {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "a".to_string(),
        Expected {
            median: 1.5,
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
