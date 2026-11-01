// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rayon::prelude::*;

const LANES: usize = 8;

/// Dot product. The shorter slice bounds the work.
#[must_use]
pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    let (lhs_blocks, lhs_tail) = lhs.as_chunks::<LANES>();
    let (rhs_blocks, rhs_tail) = rhs.as_chunks::<LANES>();
    let mut lanes = [0.0_f32; LANES];
    for (left, right) in lhs_blocks.iter().zip(rhs_blocks) {
        for ((lane, &a), &b) in lanes.iter_mut().zip(left).zip(right) {
            *lane = a.mul_add(b, *lane);
        }
    }
    let tail: f32 = lhs_tail.iter().zip(rhs_tail).map(|(&a, &b)| a * b).sum();
    lanes.iter().sum::<f32>() + tail
}

/// Rayon tasks per worker thread: enough to balance uneven cores, few enough
/// that scheduling stays negligible next to a task's rows.
const TASKS_PER_THREAD: usize = 4;

/// Row-major matrix-vector product: `output[r] = dot(weight row r, input)`
/// for a `[output.len(), cols]` weight, rows in parallel.
pub fn matvec(weight: &[f32], cols: usize, input: &[f32], output: &mut [f32]) {
    map_rows(weight, cols, output, |row| dot(row, input));
}

/// `output[r] = row_value(row r of weight)` for rows of `row_len` items,
/// computed in parallel over contiguous runs of rows. A zero `row_len`
/// writes zeros.
pub fn map_rows<T: Sync>(
    weight: &[T],
    row_len: usize,
    output: &mut [f32],
    row_value: impl Fn(&[T]) -> f32 + Sync,
) {
    if row_len == 0 {
        output.fill(0.0);
        return;
    }
    let tasks = rayon::current_num_threads() * TASKS_PER_THREAD;
    let rows_per_task = output.len().div_ceil(tasks).max(1);
    output
        .par_chunks_mut(rows_per_task)
        .zip(weight.par_chunks(rows_per_task * row_len))
        .for_each(|(slots, rows)| {
            for (slot, row) in slots.iter_mut().zip(rows.chunks_exact(row_len)) {
                *slot = row_value(row);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_of_known_vectors() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn dot_spans_more_than_one_lane_block() {
        let lhs: Vec<f32> = (1..=19_u8).map(f32::from).collect();
        assert_eq!(dot(&lhs, &[1.0; 19]), 190.0);
    }

    #[test]
    fn dot_is_bounded_by_the_shorter_slice() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[1.0, 1.0]), 3.0);
    }

    #[test]
    fn matvec_multiplies_each_row() {
        let mut out = [0.0; 2];
        matvec(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            &[1.0, 0.0, -1.0],
            &mut out,
        );
        assert_eq!(out, [-2.0, -2.0]);
    }

    #[test]
    fn map_rows_covers_every_row_across_tasks() {
        let rows = 1000;
        let weight: Vec<u16> = (0..rows * 3).collect();
        let mut out = vec![0.0; usize::from(rows)];
        map_rows(&weight, 3, &mut out, |row| f32::from(row[0]));
        let expected: Vec<f32> = (0..rows).map(|r| f32::from(3 * r)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn matvec_with_zero_columns_writes_zeros() {
        let mut out = [9.0; 2];
        matvec(&[], 0, &[], &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }
}
