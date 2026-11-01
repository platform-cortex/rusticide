// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark results, baselines and comparison for the local performance gate.

mod baseline;
mod compare;
mod e2e;
mod error;
mod kernels;
mod plan;
mod results;
mod stats;

pub use baseline::{Baseline, Expected, TOLERANCE_FLOOR};
pub use compare::{Row, Status, compare, passed, rows_to_markdown};
pub use e2e::{DECODE_PROMPT, DECODE_TOKENS, PREFILL_PROMPT, e2e_metrics};
pub use error::BenchError;
pub use kernels::kernel_metrics;
pub use plan::Plan;
pub use results::Results;
pub use stats::{Sample, summarize, throughput};
