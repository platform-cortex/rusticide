// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matrix products against one batch of inputs, run as a single parallel
//! region: every region costs a wake-up and a join of the whole thread
//! pool, so projections that share an input share the region too.

use rayon::prelude::*;

use crate::facade::dot;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::QuantKernel;

/// Rayon tasks per worker thread: enough to keep slower cores from holding
/// up the join, few enough that scheduling stays small next to a task.
const TASKS_PER_THREAD: usize = 16;

/// A borrowed row-major weight matrix, as the scheduler reads it.
#[derive(Debug, Clone, Copy)]
pub enum Weights<'a> {
    /// `f32` rows of `cols` values each.
    Dense { data: &'a [f32], cols: usize },
    /// Block-quantized rows of `row_bytes` bytes each, read by `kernel`.
    Blocks {
        kernel: &'static QuantKernel,
        data: &'a [u8],
        row_bytes: usize,
    },
}

impl Weights<'_> {
    /// The number of whole rows; zero when a row has no width.
    #[must_use]
    pub const fn rows(&self) -> usize {
        let (len, width) = match self {
            Self::Dense { data, cols } => (data.len(), *cols),
            Self::Blocks {
                data, row_bytes, ..
            } => (data.len(), *row_bytes),
        };
        match len.checked_div(width) {
            Some(rows) => rows,
            None => 0,
        }
    }

    const fn activation(&self) -> Option<Q8Kind> {
        match self {
            Self::Dense { .. } => None,
            Self::Blocks { kernel, .. } => Some(kernel.activation()),
        }
    }

    /// Writes the products of rows `first_row..` with every token of the
    /// input into `out`, the tokens of one row side by side.
    fn compute_rows(&self, first_row: usize, out: &mut [f32], input: &Activations<'_>) {
        let tokens = input.tokens().max(1);
        match self {
            Self::Dense { data, cols } => {
                let rows = data.chunks_exact(*cols).skip(first_row);
                for (slots, row) in out.chunks_mut(tokens).zip(rows) {
                    for (slot, token) in slots.iter_mut().zip(&input.dense) {
                        *slot = dot(row, token);
                    }
                }
            }
            Self::Blocks {
                kernel,
                data,
                row_bytes,
            } => {
                let quantized = input.quantized(kernel.activation());
                let rows = data.chunks_exact(*row_bytes).skip(first_row);
                for (slots, row) in out.chunks_mut(tokens).zip(rows) {
                    kernel.dot_rows(row, quantized, slots);
                }
            }
        }
    }
}

/// The token rows of a batch in every form its matrices consume, each
/// computed once per token.
#[derive(Debug, Default)]
struct Activations<'a> {
    dense: Vec<&'a [f32]>,
    block: Vec<Q8Input>,
    super_block: Vec<Q8Input>,
}

impl<'a> Activations<'a> {
    fn new<'w: 'm, 'm>(
        inputs: &'a [f32],
        tokens: usize,
        weights: impl Iterator<Item = &'m Weights<'w>>,
    ) -> Self {
        let width = inputs.len().checked_div(tokens).unwrap_or_default();
        let dense: Vec<&[f32]> = (0..tokens)
            .map(|token| {
                inputs
                    .get(token * width..(token + 1) * width)
                    .unwrap_or_default()
            })
            .collect();
        let mut activations = Self {
            dense,
            ..Self::default()
        };
        for kind in weights.filter_map(Weights::activation) {
            if activations.quantized(kind).is_empty() {
                let quantized = activations.dense.iter().map(|row| {
                    let mut input = Q8Input::default();
                    input.quantize(row, kind);
                    input
                });
                *activations.slot(kind) = quantized.collect();
            }
        }
        activations
    }

    const fn slot(&mut self, kind: Q8Kind) -> &mut Vec<Q8Input> {
        match kind {
            Q8Kind::Block => &mut self.block,
            Q8Kind::SuperBlock => &mut self.super_block,
        }
    }

    /// One quantized input per token.
    fn quantized(&self, kind: Q8Kind) -> &[Q8Input] {
        match kind {
            Q8Kind::Block => &self.block,
            Q8Kind::SuperBlock => &self.super_block,
        }
    }

    const fn tokens(&self) -> usize {
        self.dense.len()
    }
}

/// Writes `weights x input` into each job's output, all jobs in one parallel
/// region. Each output holds one value per row of its weights.
pub fn matvec_batch(input: &[f32], jobs: &mut [(Weights<'_>, &mut [f32])]) {
    matmul_batch(input, 1, jobs);
}

/// Multiplies every weight matrix with each of `tokens` input rows, all jobs
/// in one parallel region.
///
/// `inputs` holds the rows token after token; each output
/// holds `out[row * tokens + token]`, so a weight row is read once for the
/// whole batch. Every value is bit-identical to a [`matvec_batch`] of that
/// token alone.
pub fn matmul_batch(inputs: &[f32], tokens: usize, jobs: &mut [(Weights<'_>, &mut [f32])]) {
    if tokens == 0 {
        return;
    }
    let activations = Activations::new(inputs, tokens, jobs.iter().map(|(weights, _)| weights));
    let total: usize = jobs.iter().map(|(weights, _)| weights.rows()).sum();
    let tasks = rayon::current_num_threads() * TASKS_PER_THREAD;
    let rows_per_task = total.div_ceil(tasks).max(1);
    let mut runs = Vec::with_capacity(tasks + jobs.len());
    for (weights, output) in jobs.iter_mut().filter(|(weights, _)| weights.rows() > 0) {
        for (index, run) in output.chunks_mut(rows_per_task * tokens).enumerate() {
            runs.push((*weights, index * rows_per_task, run));
        }
    }
    runs.into_par_iter()
        .for_each(|(weights, first_row, run)| weights.compute_rows(first_row, run, &activations));
}

#[cfg(test)]
#[path = "matmul_tests.rs"]
mod tests;
