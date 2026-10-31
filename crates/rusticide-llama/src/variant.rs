// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{rope_adjacent, rope_half_split};

/// A model family sharing the llama forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Variant {
    Llama,
    Qwen2,
}

/// How `RoPE` pairs the columns of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RopeStyle {
    /// Pairs `(2p, 2p + 1)`.
    Adjacent,
    /// Pairs `(p, p + len / 2)`.
    HalfSplit,
}

impl Variant {
    /// The `general.architecture` value of this family.
    #[must_use]
    pub const fn architecture(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }

    #[must_use]
    pub const fn rope(self) -> RopeStyle {
        match self {
            Self::Llama => RopeStyle::Adjacent,
            Self::Qwen2 => RopeStyle::HalfSplit,
        }
    }

    #[must_use]
    pub const fn requires_attention_bias(self) -> bool {
        matches!(self, Self::Qwen2)
    }
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
    fn variants_name_their_architecture_and_rope() {
        assert_eq!(Variant::Llama.architecture(), "llama");
        assert_eq!(Variant::Qwen2.architecture(), "qwen2");
        assert_eq!(Variant::Llama.rope(), RopeStyle::Adjacent);
        assert_eq!(Variant::Qwen2.rope(), RopeStyle::HalfSplit);
        assert!(!Variant::Llama.requires_attention_bias());
        assert!(Variant::Qwen2.requires_attention_bias());
    }

    #[test]
    fn rope_styles_pair_different_columns() {
        let (mut adjacent, mut half) = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        RopeStyle::Adjacent.apply(&mut adjacent, 1, 10000.0, 2);
        RopeStyle::HalfSplit.apply(&mut half, 1, 10000.0, 2);
        assert!(adjacent[1] > 0.8 && adjacent[2] == 0.0);
        assert!(half[2] > 0.8 && half[1] == 0.0);
    }
}
