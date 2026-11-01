// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{Row, Status, compare, passed, rows_to_markdown};
use crate::{Baseline, BenchError, Expected, Results, Sample};

fn baseline(entries: &[(&str, f64)]) -> Baseline {
    let metrics = entries
        .iter()
        .map(|(name, best)| {
            (
                (*name).to_string(),
                Expected {
                    best: *best,
                    tolerance: 0.05,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    Baseline {
        machine: "box".to_string(),
        metrics,
    }
}

fn results(entries: &[(&str, f64)]) -> Results {
    let metrics = entries
        .iter()
        .map(|(name, best)| {
            (
                (*name).to_string(),
                Sample {
                    best: *best,
                    median: *best * 0.5,
                    spread: 0.0,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    Results {
        machine: "box".to_string(),
        metrics,
    }
}

#[test]
fn compare_flags_a_drop_beyond_tolerance() {
    let base = baseline(&[("a", 100.0)]);
    let within = compare(&base, &results(&[("a", 96.0)])).unwrap();
    assert_eq!(within[0].status, Status::Within);
    let regressed = compare(&base, &results(&[("a", 94.0)])).unwrap();
    assert_eq!(regressed[0].status, Status::Regressed);
    assert_eq!(regressed[0].baseline, Some(100.0));
    assert_eq!(regressed[0].measured, Some(94.0));
    assert!(passed(&within));
    assert!(!passed(&regressed));
}

#[test]
fn compare_reports_missing_and_new_metrics() {
    let rows = compare(
        &baseline(&[("a", 1.0), ("b", 2.0)]),
        &results(&[("a", 1.0), ("c", 3.0)]),
    )
    .unwrap();
    let statuses: Vec<(&str, Status)> = rows
        .iter()
        .map(|row| (row.metric.as_str(), row.status))
        .collect();
    assert_eq!(
        statuses,
        [
            ("a", Status::Within),
            ("b", Status::Missing),
            ("c", Status::New)
        ]
    );
    assert_eq!(rows[1].measured, None);
    assert_eq!(rows[2].baseline, None);
    assert!(!passed(&rows));
}

#[test]
fn new_metrics_alone_pass_the_gate() {
    let rows = compare(&baseline(&[]), &results(&[("c", 3.0)])).unwrap();
    assert!(passed(&rows));
}

#[test]
fn compare_refuses_another_machine() {
    let mut other = results(&[]);
    other.machine = "elsewhere".to_string();
    let error = compare(&baseline(&[]), &other).unwrap_err();
    assert!(
        matches!(error, BenchError::WrongMachine { ref expected, ref got }
        if expected == "box" && got == "elsewhere")
    );
}

#[test]
fn rows_to_markdown_renders_absent_values_as_dashes() {
    let rows = [
        Row {
            metric: "a".into(),
            status: Status::Within,
            baseline: Some(100.0),
            measured: Some(98.5),
        },
        Row {
            metric: "b".into(),
            status: Status::Missing,
            baseline: Some(2.0),
            measured: None,
        },
        Row {
            metric: "c".into(),
            status: Status::New,
            baseline: None,
            measured: Some(3.0),
        },
        Row {
            metric: "d".into(),
            status: Status::Regressed,
            baseline: Some(1.0),
            measured: Some(0.5),
        },
    ];
    let expected = "| metric | status | baseline | measured |\n| --- | --- | --- | --- |\n| a | within | 100.00 | 98.50 |\n| b | missing | 2.00 | - |\n| c | new | - | 3.00 |\n| d | regressed | 1.00 | 0.50 |\n";
    assert_eq!(rows_to_markdown(&rows), expected);
}
