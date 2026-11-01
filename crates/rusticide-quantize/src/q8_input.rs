// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activations quantized to signed bytes, the right-hand side of the integer
//! dot kernels.

use half::f16;

/// Elements per activation block.
pub const Q8_BLOCK: usize = 32;

/// An `f32` vector quantized in blocks of [`Q8_BLOCK`]: each block keeps one
/// scale, its values rounded to `i8`, and the sum of those `i8` values.
#[derive(Debug, Default, Clone)]
pub struct Q8Input {
    scales: Vec<f32>,
    quants: Vec<i8>,
    sums: Vec<i32>,
}

/// The [`Q8Input`] blocks covering one weight block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Q8Blocks<'a> {
    pub(crate) scales: &'a [f32],
    pub(crate) quants: &'a [i8],
    pub(crate) sums: &'a [i32],
}

/// How [`Q8Input::quantize`] scales the activation, after ggml: matching the
/// scheme llama.cpp's CPU backend pairs with each weight format keeps greedy
/// decoding on the same path as llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8Kind {
    /// ggml `Q8_0`: one f16 scale `max|x| / 127` per 32 elements.
    Block,
    /// ggml `Q8_K`: one f32 scale per 256 elements, signed so that the
    /// largest-magnitude element maps to -127.
    SuperBlock,
}

const SUPER_BLOCK: usize = 256;

impl Q8Input {
    /// Quantizes `input` as `kind`, reusing this value's buffers. A trailing
    /// partial block is dropped; callers pass whole blocks.
    pub fn quantize(&mut self, input: &[f32], kind: Q8Kind) {
        self.scales.clear();
        self.quants.clear();
        self.sums.clear();
        match kind {
            Q8Kind::Block => {
                for block in input.as_chunks::<Q8_BLOCK>().0 {
                    let amax = block.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
                    let scale = amax / 127.0;
                    let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                    self.push_block(block, inverse, f16::from_f32(scale).to_f32());
                }
            }
            Q8Kind::SuperBlock => {
                for super_block in input.as_chunks::<SUPER_BLOCK>().0 {
                    let max = super_block
                        .iter()
                        .fold(0.0_f32, |max, &v| if v.abs() > max.abs() { v } else { max });
                    let inverse = if max == 0.0 { 0.0 } else { -127.0 / max };
                    let scale = if max == 0.0 { 0.0 } else { 1.0 / inverse };
                    for block in super_block.as_chunks::<Q8_BLOCK>().0 {
                        self.push_block(block, inverse, scale);
                    }
                }
            }
        }
    }

    fn push_block(&mut self, block: &[f32; Q8_BLOCK], inverse: f32, scale: f32) {
        let start = self.quants.len();
        self.quants
            .extend(block.iter().map(|value| round_to_i8(value * inverse)));
        self.sums
            .push(self.quants[start..].iter().map(|&q| i32::from(q)).sum());
        self.scales.push(scale);
    }

    /// The number of quantized elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.quants.len()
    }

    /// Whether no element is quantized.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.quants.is_empty()
    }

    /// The decoded values, `scale * quant`, for tests and reference checks.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        self.quants
            .as_chunks::<Q8_BLOCK>()
            .0
            .iter()
            .zip(&self.scales)
            .flat_map(|(quants, &scale)| quants.iter().map(move |&q| scale * f32::from(q)))
            .collect()
    }

    /// Runs of `block_size` elements, one per weight block.
    pub(crate) fn chunks(&self, block_size: usize) -> impl Iterator<Item = Q8Blocks<'_>> {
        let per_block = block_size / Q8_BLOCK;
        self.scales
            .chunks_exact(per_block)
            .zip(self.quants.chunks_exact(block_size))
            .zip(self.sums.chunks_exact(per_block))
            .map(|((scales, quants), sums)| Q8Blocks {
                scales,
                quants,
                sums,
            })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is rounded and clamped to the i8 range first; ties go to \
              even, as NEON's vcvtnq and ggml's nearest_int round"
)]
const fn round_to_i8(value: f32) -> i8 {
    value.round_ties_even().clamp(-127.0, 127.0) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_scales_each_block_by_its_largest_magnitude() {
        let mut input = [0.0_f32; 64];
        input[0] = -254.0;
        input[1] = 128.0;
        input[32] = 0.5;
        let mut q8 = Q8Input::default();
        q8.quantize(&input, Q8Kind::Block);
        assert_eq!(q8.len(), 64);
        assert_eq!(q8.scales, [2.0, f16::from_f32(0.5 / 127.0).to_f32()]);
        assert_eq!(q8.quants[..2], [-127, 64]);
        assert_eq!(q8.quants[32], 127);
        assert_eq!(q8.sums, [-63, 127]);
    }

    #[test]
    fn an_all_zero_block_has_zero_scale_and_quants() {
        let mut q8 = Q8Input::default();
        q8.quantize(&[0.0; 32], Q8Kind::Block);
        assert_eq!(q8.scales, [0.0]);
        assert!(q8.quants.iter().all(|&q| q == 0));
        assert_eq!(q8.dequantize(), [0.0; 32]);
    }

    #[test]
    fn quantize_drops_a_partial_block_and_reuses_buffers() {
        let mut q8 = Q8Input::default();
        q8.quantize(&[1.0; 40], Q8Kind::Block);
        assert_eq!(q8.len(), 32);
        q8.quantize(&[], Q8Kind::SuperBlock);
        assert!(q8.is_empty());
    }

    #[test]
    fn dequantize_reproduces_representable_values() {
        let mut input: Vec<f32> = (-16..16)
            .map(|v| f32::from(i8::try_from(v).unwrap()))
            .collect();
        input[31] = 127.0;
        let mut q8 = Q8Input::default();
        q8.quantize(&input, Q8Kind::Block);
        let decoded = q8.dequantize();
        assert!(
            decoded
                .iter()
                .zip(&input)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }

    #[test]
    fn a_super_block_shares_one_signed_scale() {
        let mut input = [0.0_f32; 256];
        input[0] = 1.0;
        input[40] = -2.0;
        input[255] = 0.5;
        let mut q8 = Q8Input::default();
        q8.quantize(&input, Q8Kind::SuperBlock);
        let scale = 1.0 / (-127.0 / -2.0);
        assert_eq!(q8.scales, [scale; 8]);
        assert_eq!(
            (q8.quants[0], q8.quants[40], q8.quants[255]),
            (64, -127, 32)
        );
        assert_eq!(q8.sums[0], 64);
        assert_eq!(q8.sums[1], -127);
    }

    #[test]
    fn an_all_zero_super_block_has_zero_scale() {
        let mut q8 = Q8Input::default();
        q8.quantize(&[0.0; 256], Q8Kind::SuperBlock);
        assert_eq!(q8.scales, [0.0; 8]);
    }

    #[test]
    fn ties_round_to_even() {
        assert_eq!((round_to_i8(2.5), round_to_i8(-3.5)), (2, -4));
    }
}
