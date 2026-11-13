// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rusticide_kernels::{Q8Input, Q8Kind, QuantKernel, active};
use rusticide_quantize::{BlockFormat, formats};

use crate::{BenchError, Plan, Sample, throughput};

const ROWS: u32 = 256;
const COLUMNS: u32 = 4096;
const BATCH: u32 = 8;

const fn count(value: u32) -> usize {
    value as usize
}

fn row_work() -> f64 {
    f64::from(ROWS) * f64::from(COLUMNS)
}

/// Deterministic values in `[-1, 1)`.
fn activations(seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..COLUMNS)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).unwrap_or(0)) / 32768.0 - 1.0
        })
        .collect()
}

fn quantized(kind: Q8Kind, seed: u32) -> Q8Input {
    let mut input = Q8Input::default();
    input.quantize(&activations(seed), kind);
    input
}

/// Repeats `pass` until `budget` elapses and returns work units per second.
fn sustained(
    budget: Duration,
    work_per_pass: f64,
    mut pass: impl FnMut(),
) -> Result<f64, BenchError> {
    let start = Instant::now();
    let mut passes = 0.0;
    while start.elapsed() < budget {
        pass();
        passes += 1.0;
    }
    throughput(work_per_pass * passes, start.elapsed())
}

fn require_kernel(
    format: &BlockFormat,
    kernel: Option<&'static QuantKernel>,
) -> Result<&'static QuantKernel, BenchError> {
    kernel.ok_or_else(|| BenchError::MissingKernel {
        format: format.name(),
    })
}

fn synthetic(format: &BlockFormat, blocks: usize) -> Result<Vec<u8>, BenchError> {
    format
        .synthetic_blocks(blocks, 1)
        .ok_or_else(|| BenchError::TooLarge {
            format: format.name(),
        })
}

/// A synthetic `ROWS` x `columns` matrix of `format`, prepared for the
/// active adapter, its row length in bytes, and the kernels that read it.
fn layout(
    format: &BlockFormat,
    columns: u32,
) -> Result<(Vec<u8>, usize, &'static QuantKernel), BenchError> {
    let columns = count(columns);
    let row_bytes = format
        .row_bytes(columns)
        .ok_or_else(|| BenchError::UnalignedRow {
            format: format.name(),
            columns,
        })?;
    let mut matrix = synthetic(format, count(ROWS) * columns / format.block_size())?;
    let kernel = active().prepare_matrix(format.type_id(), &mut matrix, row_bytes);
    Ok((matrix, row_bytes, require_kernel(format, kernel)?))
}

fn row_metric(format: &BlockFormat, plan: &Plan) -> Result<Sample, BenchError> {
    let (matrix, row_bytes, kernel) = layout(format, COLUMNS)?;
    let input = [quantized(kernel.activation(), 2)];
    let mut out = [0.0_f32; count(ROWS)];
    plan.sample(|| {
        sustained(plan.budget, row_work(), || {
            kernel.dot_matrix(black_box(&matrix), row_bytes, black_box(&input), &mut out);
            black_box(&out);
        })
    })
}

fn rows8_metric(format: &BlockFormat, plan: &Plan) -> Result<Sample, BenchError> {
    let (matrix, row_bytes, kernel) = layout(format, COLUMNS)?;
    let inputs: Vec<Q8Input> = (3..3 + BATCH)
        .map(|seed| quantized(kernel.activation(), seed))
        .collect();
    let mut out = [0.0_f32; count(ROWS * BATCH)];
    plan.sample(|| {
        sustained(plan.budget, row_work() * f64::from(BATCH), || {
            kernel.dot_matrix(black_box(&matrix), row_bytes, black_box(&inputs), &mut out);
            black_box(&out);
        })
    })
}

fn quantize_metric(kind: Q8Kind, plan: &Plan) -> Result<Sample, BenchError> {
    let values = activations(4);
    let mut input = Q8Input::default();
    plan.sample(|| {
        sustained(plan.budget, f64::from(COLUMNS), || {
            input.quantize(black_box(&values), kind);
            black_box(&input);
        })
    })
}

/// Single-thread kernel throughputs for every block format with a kernel
/// and both activation quantizers.
///
/// # Errors
///
/// A failure to summarize the runs.
pub fn kernel_metrics(plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError> {
    let mut metrics = BTreeMap::new();
    for format in formats() {
        let name = format.name();
        let row = match row_metric(format, plan) {
            Err(BenchError::MissingKernel { .. }) => continue,
            row => row?,
        };
        metrics.insert(format!("kernel/{name}/row"), row);
        metrics.insert(format!("kernel/{name}/rows8"), rows8_metric(format, plan)?);
    }
    metrics.insert(
        "kernel/q8/block".into(),
        quantize_metric(Q8Kind::Block, plan)?,
    );
    metrics.insert(
        "kernel/q8/superblock".into(),
        quantize_metric(Q8Kind::SuperBlock, plan)?,
    );
    Ok(metrics)
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
