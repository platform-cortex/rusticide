// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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
}
