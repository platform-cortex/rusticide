// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::BenchError;

/// Fastest and median throughput of several runs and the relative spread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Interference only slows a run down, so the fastest run is the stable
    /// estimator the gate compares.
    pub best: f64,
    pub median: f64,
    /// `(max - min) / median`.
    pub spread: f64,
}

/// Summarizes throughputs into their best, median and relative spread.
///
/// # Errors
///
/// Returns [`BenchError::NoRuns`] when `throughputs` is empty and
/// [`BenchError::NotFinite`] when a run is not a finite positive number.
pub fn summarize(throughputs: &[f64]) -> Result<Sample, BenchError> {
    if throughputs
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(BenchError::NotFinite);
    }
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
        best: max,
        median,
        spread: (max - min) / median,
    })
}

/// Work units per second.
///
/// # Errors
///
/// Returns [`BenchError::ZeroElapsed`] or [`BenchError::ZeroWork`] for zero
/// inputs and [`BenchError::NotFinite`] for a non-finite work or result.
pub fn throughput(work: f64, elapsed: Duration) -> Result<f64, BenchError> {
    if elapsed.is_zero() {
        return Err(BenchError::ZeroElapsed);
    }
    if !work.is_finite() {
        return Err(BenchError::NotFinite);
    }
    if work <= 0.0 {
        return Err(BenchError::ZeroWork);
    }
    let rate = work / elapsed.as_secs_f64();
    if rate.is_finite() {
        Ok(rate)
    } else {
        Err(BenchError::NotFinite)
    }
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
