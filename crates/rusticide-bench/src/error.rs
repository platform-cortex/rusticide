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
    #[error("baseline adapter {expected:?} differs from the results' {got:?}")]
    OtherAdapter { expected: String, got: String },
    #[error("metric {metric} is missing from a session")]
    MissingInSession { metric: String },
    #[error("no kernel for format {format}")]
    MissingKernel { format: &'static str },
    #[error("format {format} cannot hold rows of {columns} columns")]
    UnalignedRow {
        format: &'static str,
        columns: usize,
    },
    #[error("synthetic {format} blocks do not fit in memory")]
    TooLarge { format: &'static str },
    #[error("model error: {0}")]
    Model(#[from] rusticide_model::ModelError),
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
