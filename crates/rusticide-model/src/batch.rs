// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matrix products against one batch of inputs, run as a single parallel
//! region: every region costs a wake-up and a join of the whole thread
//! pool, so projections that share an input share the region too.

use rayon::prelude::*;
use rusticide_quantize::{Q8Input, Q8Kind};

use crate::weights::Matrix;

/// Rayon tasks per worker thread: enough to keep slower cores from holding
/// up the join, few enough that scheduling stays small next to a task.
const TASKS_PER_THREAD: usize = 16;

/// The token rows of a batch in every form its matrices consume, each
/// computed once per token.
#[derive(Debug, Default)]
pub(crate) struct Activations<'a> {
    pub(crate) dense: Vec<&'a [f32]>,
    block: Vec<Q8Input>,
    super_block: Vec<Q8Input>,
}

impl<'a> Activations<'a> {
    fn new<'m>(
        inputs: &'a [f32],
        tokens: usize,
        matrices: impl Iterator<Item = &'m Matrix>,
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
        for kind in matrices.filter_map(Matrix::activation) {
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
    pub(crate) fn quantized(&self, kind: Q8Kind) -> &[Q8Input] {
        match kind {
            Q8Kind::Block => &self.block,
            Q8Kind::SuperBlock => &self.super_block,
        }
    }

    pub(crate) const fn tokens(&self) -> usize {
        self.dense.len()
    }
}

/// Writes `matrix x input` into each job's output, all jobs in one parallel
/// region. Each output holds one value per row of its matrix.
pub fn matvec_batch(input: &[f32], jobs: &mut [(&Matrix, &mut [f32])]) {
    matmul_batch(input, 1, jobs);
}

/// Multiplies every matrix with each of `tokens` input rows, all jobs in one
/// parallel region.
///
/// `inputs` holds the rows token after token; each output
/// holds `out[row * tokens + token]`, so a weight row is read once for the
/// whole batch. Every value is bit-identical to a [`matvec_batch`] of that
/// token alone.
pub fn matmul_batch(inputs: &[f32], tokens: usize, jobs: &mut [(&Matrix, &mut [f32])]) {
    if tokens == 0 {
        return;
    }
    let activations = Activations::new(inputs, tokens, jobs.iter().map(|(matrix, _)| *matrix));
    let total: usize = jobs.iter().map(|(matrix, _)| matrix.rows()).sum();
    let tasks = rayon::current_num_threads() * TASKS_PER_THREAD;
    let rows_per_task = total.div_ceil(tasks).max(1);
    let mut runs = Vec::with_capacity(tasks + jobs.len());
    for (matrix, output) in jobs.iter_mut() {
        for (index, run) in output.chunks_mut(rows_per_task * tokens).enumerate() {
            runs.push((*matrix, index * rows_per_task, run));
        }
    }
    runs.into_par_iter()
        .for_each(|(matrix, first_row, run)| matrix.compute_rows(first_row, run, &activations));
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
