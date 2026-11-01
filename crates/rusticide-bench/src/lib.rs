// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark results, baselines and comparison for the local performance gate.

mod baseline;
mod compare;
mod error;
mod results;
mod stats;

pub use baseline::{Baseline, Expected, TOLERANCE_FLOOR};
pub use compare::{Row, Status, compare, passed, rows_to_markdown};
pub use error::BenchError;
pub use results::Results;
pub use stats::{Sample, summarize, throughput};
