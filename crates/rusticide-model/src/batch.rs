// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matrix-vector products against one input, run as a single parallel
//! region: every region costs a wake-up and a join of the whole thread
//! pool, so projections that share an input share the region too.

use rayon::prelude::*;
use rusticide_quantize::{Q8Input, Q8Kind};

use crate::weights::Matrix;

/// Rayon tasks per worker thread: enough to keep slower cores from holding
/// up the join, few enough that scheduling stays small next to a task.
const TASKS_PER_THREAD: usize = 16;

/// One input in every form the matrices of a batch consume, each computed
/// once.
#[derive(Debug, Default)]
pub(crate) struct Activations<'a> {
    pub(crate) dense: &'a [f32],
    block: Q8Input,
    super_block: Q8Input,
}

impl<'a> Activations<'a> {
    fn new<'m>(dense: &'a [f32], matrices: impl Iterator<Item = &'m Matrix>) -> Self {
        let mut activations = Self {
            dense,
            ..Self::default()
        };
        for kind in matrices.filter_map(Matrix::activation) {
            let slot = activations.slot(kind);
            if slot.is_empty() {
                slot.quantize(dense, kind);
            }
        }
        activations
    }

    const fn slot(&mut self, kind: Q8Kind) -> &mut Q8Input {
        match kind {
            Q8Kind::Block => &mut self.block,
            Q8Kind::SuperBlock => &mut self.super_block,
        }
    }

    pub(crate) const fn quantized(&self, kind: Q8Kind) -> &Q8Input {
        match kind {
            Q8Kind::Block => &self.block,
            Q8Kind::SuperBlock => &self.super_block,
        }
    }
}

/// Writes `matrix x input` into each job's output, all jobs in one parallel
/// region. Each output holds one value per row of its matrix.
pub fn matvec_batch(input: &[f32], jobs: &mut [(&Matrix, &mut [f32])]) {
    let activations = Activations::new(input, jobs.iter().map(|(matrix, _)| *matrix));
    let total: usize = jobs.iter().map(|(_, output)| output.len()).sum();
    let tasks = rayon::current_num_threads() * TASKS_PER_THREAD;
    let rows_per_task = total.div_ceil(tasks).max(1);
    let mut runs = Vec::with_capacity(tasks + jobs.len());
    for (matrix, output) in jobs.iter_mut() {
        for (index, run) in output.chunks_mut(rows_per_task).enumerate() {
            runs.push((*matrix, index * rows_per_task, run));
        }
    }
    runs.into_par_iter()
        .for_each(|(matrix, first_row, run)| matrix.compute_rows(first_row, run, &activations));
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
