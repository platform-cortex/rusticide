// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stats::median_of_sorted;
use crate::{BenchError, Results};

/// Relative drop of a `kernel/*` metric below its baseline that still passes.
pub const KERNEL_TOLERANCE: f64 = 0.05;
/// Relative drop of an `e2e/*` metric (decode and prefill) that still passes.
pub const E2E_TOLERANCE: f64 = 0.05;

/// Recorded best and the relative drop allowed below it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Expected {
    pub best: f64,
    pub tolerance: f64,
}

/// Expected metrics recorded on one machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub machine: String,
    pub metrics: BTreeMap<String, Expected>,
}

impl Baseline {
    /// Records the median across `sessions` of each metric's best, with the
    /// tolerance of its kind.
    ///
    /// # Errors
    ///
    /// [`BenchError::NoRuns`] without sessions, [`BenchError::WrongMachine`]
    /// when they come from different machines and
    /// [`BenchError::MissingInSession`] when one lacks a metric another has.
    pub fn from_sessions(sessions: &[Results]) -> Result<Self, BenchError> {
        let (first, rest) = sessions.split_first().ok_or(BenchError::NoRuns)?;
        if let Some(other) = rest.iter().find(|session| session.machine != first.machine) {
            return Err(BenchError::WrongMachine {
                expected: first.machine.clone(),
                got: other.machine.clone(),
            });
        }
        let metrics = first
            .metrics
            .keys()
            .chain(rest.iter().flat_map(|session| session.metrics.keys()))
            .map(|name| Ok((name.clone(), expected(name, sessions)?)))
            .collect::<Result<_, BenchError>>()?;
        Ok(Self {
            machine: first.machine.clone(),
            metrics,
        })
    }
}

fn expected(name: &str, sessions: &[Results]) -> Result<Expected, BenchError> {
    let mut bests = sessions
        .iter()
        .map(|session| {
            session
                .metrics
                .get(name)
                .map(|sample| sample.best)
                .ok_or_else(|| BenchError::MissingInSession {
                    metric: name.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    bests.sort_by(f64::total_cmp);
    let tolerance = if name.starts_with("e2e/") {
        E2E_TOLERANCE
    } else {
        KERNEL_TOLERANCE
    };
    Ok(Expected {
        best: median_of_sorted(&bests),
        tolerance,
    })
}

#[cfg(test)]
#[path = "baseline_tests.rs"]
mod tests;
