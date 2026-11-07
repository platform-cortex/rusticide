// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `f32` dot kernels of the portable adapter.
//!
//! Always inlined, so that `crate::avx2::float` compiles the same code
//! with `fma` enabled.
#![expect(
    clippy::inline_always,
    reason = "a caller with `fma` enabled must compile these bodies itself"
)]

const LANES: usize = 8;

/// Dot product. The shorter slice bounds the work.
#[must_use]
#[inline(always)]
pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    let bound = lhs.len().min(rhs.len());
    let (lhs, rhs) = (lhs.split_at(bound).0, rhs.split_at(bound).0);
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

/// Scores every `row_stride`-wide row of `rows`: `out[i]` is
/// `dot(query, rows_i[offset..]) / scale`. The shortest of `out`, the rows
/// and `query` bounds the work; a zero stride is treated as one.
#[inline(always)]
pub fn dot_rows(
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    let end = offset.checked_add(query.len());
    for (slot, row) in out.iter_mut().zip(rows.chunks_exact(row_stride.max(1))) {
        let window = end.and_then(|end| row.get(offset..end)).unwrap_or_default();
        *slot = dot(query, window) / scale;
    }
}

/// Adds `weights[i] x rows_i[offset..offset + out.len()]` into `out`,
/// each product fused into its slot.
#[inline(always)]
pub fn mix_rows(weights: &[f32], rows: &[f32], row_stride: usize, offset: usize, out: &mut [f32]) {
    let end = offset.checked_add(out.len());
    for (&weight, row) in weights.iter().zip(rows.chunks_exact(row_stride.max(1))) {
        let window = end.and_then(|end| row.get(offset..end)).unwrap_or_default();
        for (slot, &entry) in out.iter_mut().zip(window) {
            *slot = entry.mul_add(weight, *slot);
        }
    }
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
    fn dot_rows_scores_each_stride_window() {
        let rows = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0];
        let mut out = [0.0; 2];
        dot_rows(&[1.0, 1.0], &rows, 3, 0, 2.0, &mut out);
        assert_eq!(out, [1.5, 3.5]);
        dot_rows(&[1.0], &rows, 3, 2, 1.0, &mut out);
        assert_eq!(out, [9.0, 9.0]);
    }

    #[test]
    fn dot_rows_with_an_offset_past_the_row_scores_zero() {
        let mut out = [1.0];
        dot_rows(&[1.0], &[1.0, 2.0], 2, 5, 1.0, &mut out);
        assert_eq!(out, [0.0]);
    }

    #[test]
    fn dot_rows_with_an_offset_that_overflows_scores_zero() {
        let mut out = [1.0];
        dot_rows(&[1.0], &[1.0, 2.0], 2, usize::MAX, 1.0, &mut out);
        assert_eq!(out, [0.0]);
    }

    #[test]
    fn mix_rows_accumulates_each_weighted_stride_window() {
        let rows = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0];
        let mut out = [10.0, 20.0];
        mix_rows(&[2.0, 0.5], &rows, 3, 0, &mut out);
        assert_eq!(out, [13.5, 26.0]);
        let mut tail = [0.0];
        mix_rows(&[1.0, 1.0], &rows, 3, 2, &mut tail);
        assert_eq!(tail, [18.0]);
    }

    #[test]
    fn mix_rows_matches_a_fused_reference_bit_for_bit() {
        let rows: Vec<f32> = (1..=48_u8).map(|v| f32::from(v) / 7.0).collect();
        let weights = [0.3, 0.7];
        for offset in [0, 12] {
            let mut out = [0.0; 12];
            mix_rows(&weights, &rows, 24, offset, &mut out);
            let mut expected = [0.0_f32; 12];
            for (index, weight) in weights.iter().enumerate() {
                let start = index * 24 + offset;
                for (slot, entry) in expected.iter_mut().zip(&rows[start..start + 12]) {
                    *slot = entry.mul_add(*weight, *slot);
                }
            }
            assert_eq!(out.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }

    #[test]
    fn mix_rows_skips_windows_past_the_row() {
        let mut out = [5.0];
        mix_rows(&[1.0], &[1.0, 2.0], 2, 5, &mut out);
        assert_eq!(out, [5.0]);
        mix_rows(&[1.0], &[1.0, 2.0], 2, usize::MAX, &mut out);
        assert_eq!(out, [5.0]);
        mix_rows(&[1.0], &[1.0, 2.0], 0, 0, &mut out);
        assert_eq!(out, [6.0]);
    }

    #[test]
    fn dot_with_a_longer_right_side_is_bounded_by_the_left() {
        assert_eq!(dot(&[1.0; 12], &[1.0; 24]), 12.0);
        assert_eq!(dot(&[1.0; 24], &[1.0; 12]), 12.0);
    }

    #[test]
    fn dot_rows_scores_each_head_window_of_a_wide_row() {
        let rows: Vec<f32> = (1..=48_u8).map(f32::from).collect();
        let query: Vec<f32> = (1..=12_u8).map(f32::from).collect();
        for offset in [0, 12] {
            let mut out = [0.0; 2];
            dot_rows(&query, &rows, 24, offset, 1.0, &mut out);
            for (index, score) in out.iter().enumerate() {
                let start = index * 24 + offset;
                let expected: f32 = query
                    .iter()
                    .zip(&rows[start..start + 12])
                    .map(|(a, b)| a * b)
                    .sum();
                assert_eq!(*score, expected);
            }
        }
    }
}
