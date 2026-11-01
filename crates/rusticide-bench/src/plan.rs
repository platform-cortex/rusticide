// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use crate::{BenchError, Sample, summarize};

/// How a metric is sampled: `runs` measured runs after one discarded warm-up.
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    pub runs: usize,
    /// How long a kernel run repeats its work.
    pub budget: Duration,
}

impl Plan {
    /// Runs `run` once as a discarded warm-up, then `self.runs` times, and
    /// summarizes the throughputs it returns.
    pub(crate) fn sample(
        &self,
        mut run: impl FnMut() -> Result<f64, BenchError>,
    ) -> Result<Sample, BenchError> {
        run()?;
        let throughputs = (0..self.runs)
            .map(|_| run())
            .collect::<Result<Vec<_>, _>>()?;
        summarize(&throughputs)
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
