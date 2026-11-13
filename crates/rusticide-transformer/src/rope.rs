// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{rope_adjacent, rope_half_split};

/// How `RoPE` pairs the columns of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RopeStyle {
    /// Pairs `(2p, 2p + 1)`.
    Adjacent,
    /// Pairs `(p, p + len / 2)`.
    HalfSplit,
}

impl RopeStyle {
    /// Rotates `row` in place with the kernel matching this style.
    pub fn apply(self, row: &mut [f32], position: usize, theta: f32, dim_pairs: usize) {
        match self {
            Self::Adjacent => rope_adjacent(row, position, theta, dim_pairs),
            Self::HalfSplit => rope_half_split(row, position, theta, dim_pairs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_styles_pair_different_columns() {
        let (mut adjacent, mut half) = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        RopeStyle::Adjacent.apply(&mut adjacent, 1, 10000.0, 2);
        RopeStyle::HalfSplit.apply(&mut half, 1, 10000.0, 2);
        assert!(adjacent[1] > 0.8 && adjacent[2] == 0.0);
        assert!(half[2] > 0.8 && half[1] == 0.0);
    }
}
