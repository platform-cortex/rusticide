// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::BenchError;

/// Median throughput of several runs and its relative spread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub median: f64,
    /// `(max - min) / median`.
    pub spread: f64,
}

/// Summarizes throughputs into their median and relative spread.
///
/// # Errors
///
/// Returns [`BenchError::NoRuns`] when `throughputs` is empty.
pub fn summarize(throughputs: &[f64]) -> Result<Sample, BenchError> {
    let mut sorted = throughputs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (Some(&min), Some(&max)) = (sorted.first(), sorted.last()) else {
        return Err(BenchError::NoRuns);
    };
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[middle - 1], sorted[middle])
    } else {
        sorted[middle]
    };
    Ok(Sample {
        median,
        spread: (max - min) / median,
    })
}

/// Work units per second.
///
/// # Errors
///
/// Returns [`BenchError::ZeroElapsed`] or [`BenchError::ZeroWork`] when the
/// ratio would be meaningless.
pub fn throughput(work: f64, elapsed: Duration) -> Result<f64, BenchError> {
    if elapsed.is_zero() {
        return Err(BenchError::ZeroElapsed);
    }
    if work <= 0.0 {
        return Err(BenchError::ZeroWork);
    }
    Ok(work / elapsed.as_secs_f64())
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
