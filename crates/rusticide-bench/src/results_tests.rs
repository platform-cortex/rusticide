// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::Results;
use crate::Sample;

fn results() -> Results {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "zeta".to_string(),
        Sample {
            best: 3.0,
            median: 2.0,
            spread: 0.25,
        },
    );
    metrics.insert(
        "alpha".to_string(),
        Sample {
            best: 105.0,
            median: 100.5,
            spread: 0.05,
        },
    );
    Results {
        machine: "box".to_string(),
        metrics,
    }
}

#[test]
fn results_round_trip_through_json() {
    let original = results();
    let text = serde_json::to_string(&original).unwrap();
    assert_eq!(serde_json::from_str::<Results>(&text).unwrap(), original);
}

#[test]
fn to_markdown_lists_one_row_per_metric_in_name_order() {
    let expected = "| metric | best | median | spread |\n| --- | --- | --- | --- |\n| alpha | 105.00 | 100.50 | 5.0% |\n| zeta | 3.00 | 2.00 | 25.0% |\n";
    assert_eq!(results().to_markdown(), expected);
}
