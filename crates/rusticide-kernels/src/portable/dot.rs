// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

const LANES: usize = 8;

/// Dot product. The shorter slice bounds the work.
#[must_use]
pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
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

pub(super) fn dot_rows(
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    for (slot, row) in out.iter_mut().zip(rows.chunks_exact(row_stride.max(1))) {
        *slot = dot(
            query,
            row.get(offset..offset + query.len()).unwrap_or_default(),
        ) / scale;
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
