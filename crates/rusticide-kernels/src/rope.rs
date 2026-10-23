// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::lossy;

/// Rotates adjacent pairs `(2p, 2p+1)` for `p < min(dim_pairs, row.len() / 2)`
/// by angle `position * theta^(-2p / row.len())`.
pub fn rope_adjacent(row: &mut [f32], position: usize, theta: f32, dim_pairs: usize) {
    let columns = lossy(row.len());
    let position = lossy(position);
    let (pairs, _) = row.as_chunks_mut::<2>();
    for (pair, [low, high]) in pairs.iter_mut().take(dim_pairs).enumerate() {
        let frequency = theta.powf(-lossy(2 * pair) / columns);
        let (sin, cos) = (position * frequency).sin_cos();
        let (old_low, old_high) = (*low, *high);
        *low = old_low.mul_add(cos, -old_high * sin);
        *high = old_high.mul_add(cos, old_low * sin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_close;

    #[test]
    fn rope_adjacent_rotates_by_position() {
        let mut row = [1.0, 0.0, 1.0, 0.0];
        rope_adjacent(&mut row, 1, 10000.0, 2);
        let slow = 10000.0_f32.powf(-2.0 / 4.0);
        assert_close(
            &row,
            &[1.0_f32.cos(), 1.0_f32.sin(), slow.cos(), slow.sin()],
        );
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut row = [1.0, 2.0, 3.0, 4.0];
        rope_adjacent(&mut row, 0, 10000.0, 2);
        assert_close(&row, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_only_dim_pairs_pairs() {
        let mut row = [1.0, 0.0, 1.0, 0.0];
        rope_adjacent(&mut row, 1, 10000.0, 1);
        assert_close(&row, &[1.0_f32.cos(), 1.0_f32.sin(), 1.0, 0.0]);
    }

    #[test]
    fn rope_ignores_a_trailing_unpaired_column() {
        let mut row = [1.0, 0.0, 7.0];
        rope_adjacent(&mut row, 1, 10000.0, 4);
        assert_eq!(row[2], 7.0);
    }
}
