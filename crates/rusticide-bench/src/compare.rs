// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::{Baseline, BenchError, Expected, Results};

/// Outcome of comparing one metric with its baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Within,
    Regressed,
    Missing,
    New,
}

impl Status {
    const fn label(self) -> &'static str {
        match self {
            Self::Within => "within",
            Self::Regressed => "regressed",
            Self::Missing => "missing",
            Self::New => "new",
        }
    }
}

/// One metric of a comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub metric: String,
    pub status: Status,
    pub baseline: Option<f64>,
    pub measured: Option<f64>,
}

/// Compares `results` with `baseline`, one row per metric in name order.
///
/// # Errors
///
/// Returns [`BenchError::WrongMachine`] when the machines differ.
pub fn compare(baseline: &Baseline, results: &Results) -> Result<Vec<Row>, BenchError> {
    if baseline.machine != results.machine {
        return Err(BenchError::WrongMachine {
            expected: baseline.machine.clone(),
            got: results.machine.clone(),
        });
    }
    let names: BTreeSet<&String> = baseline
        .metrics
        .keys()
        .chain(results.metrics.keys())
        .collect();
    let rows = names
        .into_iter()
        .map(|name| {
            let expected = baseline.metrics.get(name);
            let measured = results.metrics.get(name).map(|sample| sample.median);
            Row {
                metric: name.clone(),
                status: status(expected, measured),
                baseline: expected.map(|e| e.median),
                measured,
            }
        })
        .collect();
    Ok(rows)
}

fn status(expected: Option<&Expected>, measured: Option<f64>) -> Status {
    match (expected, measured) {
        (Some(expected), Some(measured))
            if measured < expected.median * (1.0 - expected.tolerance) =>
        {
            Status::Regressed
        }
        (Some(_), Some(_)) => Status::Within,
        (Some(_), None) => Status::Missing,
        (None, _) => Status::New,
    }
}

/// True when no metric regressed or went missing.
#[must_use]
pub fn passed(rows: &[Row]) -> bool {
    rows.iter()
        .all(|row| !matches!(row.status, Status::Regressed | Status::Missing))
}

/// Renders the comparison as a markdown table.
#[must_use]
pub fn rows_to_markdown(rows: &[Row]) -> String {
    let mut table =
        String::from("| metric | status | baseline | measured |\n| --- | --- | --- | --- |\n");
    for row in rows {
        let _ = writeln!(
            table,
            "| {} | {} | {} | {} |",
            row.metric,
            row.status.label(),
            cell(row.baseline),
            cell(row.measured)
        );
    }
    table
}

fn cell(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |number| format!("{number:.2}"))
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
