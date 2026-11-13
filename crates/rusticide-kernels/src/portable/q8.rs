// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;

use crate::q8_input::{Q8_BLOCK, Q8Input, Q8Kind};

/// Activation blocks per `Q8_K` super block of 256 elements.
const PER_SUPER_BLOCK: usize = 8;

fn abs_max(values: &[f32]) -> f32 {
    values.iter().fold(0.0_f32, |max, v| max.max(v.abs()))
}

/// The element of largest magnitude, the first one on a tie, as ggml's
/// `Q8_K` picks it; zero for an all-zero input.
#[expect(
    clippy::float_cmp,
    reason = "amax is exactly the magnitude of one of the elements, so exact equality finds it"
)]
fn signed_max(blocks: &[[f32; Q8_BLOCK]]) -> f32 {
    let amax = blocks
        .iter()
        .map(|block| abs_max(block))
        .fold(0.0_f32, f32::max);
    blocks
        .as_flattened()
        .iter()
        .copied()
        .find(|v| v.abs() == amax)
        .unwrap_or(0.0)
}

/// bitnet.cpp's legacy `127 / max|x|`. The running max is updated as C's
/// `MAX(a, b)` macro, `a > b ? a : b`, does: a NaN replaces it, and the
/// next element replaces the NaN.
fn row_divisor(values: &[f32]) -> f32 {
    let max = values.iter().fold(1e-5_f64, |max, value| {
        let magnitude = f64::from(value.abs());
        if max > magnitude { max } else { magnitude }
    });
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reference divides in f64 and rounds the divisor to f32"
    )]
    let divisor = (127.0 / max) as f32;
    divisor
}

/// Rounds `block * inverse` into `quants` with `round`, returning their sum.
fn quantize_block(
    block: &[f32; Q8_BLOCK],
    inverse: f32,
    quants: &mut [i8; Q8_BLOCK],
    round: fn(f32) -> i8,
) -> i32 {
    for (quant, &value) in quants.iter_mut().zip(block) {
        *quant = round(value * inverse);
    }
    quants.iter().map(|&q| i32::from(q)).sum()
}

pub(crate) fn quantize(target: &mut Q8Input, values: &[f32], kind: Q8Kind) {
    let (blocks, _) = values.as_chunks::<Q8_BLOCK>();
    let blocks = match kind {
        Q8Kind::Block | Q8Kind::Row => blocks,
        Q8Kind::SuperBlock => &blocks[..blocks.len() / PER_SUPER_BLOCK * PER_SUPER_BLOCK],
    };
    target.quants.resize(blocks.len() * Q8_BLOCK, 0);
    target.scales.resize(blocks.len(), 0.0);
    target.sums.resize(blocks.len(), 0);
    let (quants, _) = target.quants.as_chunks_mut::<Q8_BLOCK>();
    let outputs = quants
        .iter_mut()
        .zip(target.scales.iter_mut().zip(&mut target.sums));
    match kind {
        Q8Kind::Block => {
            for (block, (quants, (scale, sum))) in blocks.iter().zip(outputs) {
                let step = abs_max(block) / 127.0;
                let inverse = if step == 0.0 { 0.0 } else { 1.0 / step };
                *sum = quantize_block(block, inverse, quants, round_to_i8);
                *scale = f16::from_f32(step).to_f32();
            }
        }
        Q8Kind::SuperBlock => {
            let (supers, _) = blocks.as_chunks::<PER_SUPER_BLOCK>();
            let mut outputs = outputs;
            for super_block in supers {
                let max = signed_max(super_block);
                let inverse = if max == 0.0 { 0.0 } else { -127.0 / max };
                let step = if max == 0.0 { 0.0 } else { 1.0 / inverse };
                for (block, (quants, (scale, sum))) in super_block.iter().zip(outputs.by_ref()) {
                    *sum = quantize_block(block, inverse, quants, round_to_i8);
                    *scale = step;
                }
            }
        }
        Q8Kind::Row => {
            let divisor = row_divisor(blocks.as_flattened());
            for (block, (quants, (scale, sum))) in blocks.iter().zip(outputs) {
                *sum = quantize_block(block, divisor, quants, round_row_to_i8);
                *scale = divisor;
            }
        }
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

/// Rounds half to even and saturates to `[-128, 127]`. This equals the
/// reference's `nearest_int` and clamp for `|value| <= 4194303`, where its
/// `+ 12582912.f` trick is exact, a bound a row's scaled values (near
/// `[-127, 127]`) never reach. A NaN gives 0, as `nearest_int` gives for
/// the default quiet NaN.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the saturating cast is the reference's clamp to [-128, 127], and \
              takes NaN to 0 as ggml's nearest_int does"
)]
const fn round_row_to_i8(value: f32) -> i8 {
    value.round_ties_even() as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_round_to_even() {
        assert_eq!((round_to_i8(2.5), round_to_i8(-3.5)), (2, -4));
    }
}
