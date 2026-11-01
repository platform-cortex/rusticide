// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Results;

/// Smallest relative drop a baseline tolerates.
pub const TOLERANCE_FLOOR: f64 = 0.05;

/// Recorded median and the relative drop allowed below it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Expected {
    pub median: f64,
    pub tolerance: f64,
}

/// Expected metrics recorded on one machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub machine: String,
    pub metrics: BTreeMap<String, Expected>,
}

impl Baseline {
    /// Records `results`, tolerating at least [`TOLERANCE_FLOOR`].
    #[must_use]
    pub fn from_results(results: &Results) -> Self {
        let metrics = results
            .metrics
            .iter()
            .map(|(name, sample)| {
                let expected = Expected {
                    median: sample.median,
                    tolerance: sample.spread.max(TOLERANCE_FLOOR),
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
