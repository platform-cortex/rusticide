// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Results;

/// Relative drop of a `kernel/*` metric below its baseline that still passes.
pub const KERNEL_TOLERANCE: f64 = 0.10;
/// Relative drop of an `e2e/*` metric (decode and prefill) that still passes.
pub const E2E_TOLERANCE: f64 = 0.15;

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
    /// Records the best of `results` with the tolerance of each metric kind.
    #[must_use]
    pub fn from_results(results: &Results) -> Self {
        let metrics = results
            .metrics
            .iter()
            .map(|(name, sample)| {
                let tolerance = if name.starts_with("e2e/") {
                    E2E_TOLERANCE
                } else {
                    KERNEL_TOLERANCE
                };
                let expected = Expected {
                    best: sample.best,
                    tolerance,
                };
                (name.clone(), expected)
            })
            .collect();
        Self {
            machine: results.machine.clone(),
            metrics,
        }
    }
}

#[cfg(test)]
#[path = "baseline_tests.rs"]
mod tests;
