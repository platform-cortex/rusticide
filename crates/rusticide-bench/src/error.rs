// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failures raised while summarizing, serializing or comparing benchmarks.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("no runs to summarize")]
    NoRuns,
    #[error("elapsed time is zero")]
    ZeroElapsed,
    #[error("work done is zero")]
    ZeroWork,
    #[error("throughput is not a finite positive number")]
    NotFinite,
    #[error("baseline was recorded on {expected}, results come from {got}")]
    WrongMachine { expected: String, got: String },
    #[error("model error: {0}")]
    Model(#[from] rusticide_model::ModelError),
    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
