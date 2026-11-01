// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer dot kernels: one weight block against the [`Q8Input`] blocks it
//! spans. Weights are unpacked to `i8` on the stack and multiplied in `i32`,
//! which the compiler vectorizes; floats appear only once per 16 or 32
//! elements, for the scales.
//!
//! [`Q8Input`]: crate::Q8Input

use crate::half_bits::read_f16_le;
use crate::q8_input::{Q8_BLOCK, Q8Blocks};

const HALF: usize = Q8_BLOCK / 2;

fn dot_i8<const N: usize>(weights: &[i8; N], quants: &[i8]) -> i32 {
    weights
        .iter()
        .zip(quants)
        .map(|(&w, &q)| i32::from(w) * i32::from(q))
        .sum()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "an i32 block sum is far below 2^24 in magnitude for i8 x i8 over 32 lanes"
)]
const fn lossy(value: i32) -> f32 {
    value as f32
}

/// Reinterprets an unsigned value below 128 as `i8`.
const fn signed(value: u8) -> i8 {
    value.cast_signed()
}

pub(crate) fn q8_0(block: &[u8], input: Q8Blocks<'_>) -> f32 {
    let mut weights = [0_i8; Q8_BLOCK];
    for (weight, &byte) in weights.iter_mut().zip(&block[2..]) {
        *weight = byte.cast_signed();
    }
    read_f16_le(block, 0) * input.scales[0] * lossy(dot_i8(&weights, input.quants))
}

pub(crate) fn q4_0(block: &[u8], input: Q8Blocks<'_>) -> f32 {
    let mut weights = [0_i8; Q8_BLOCK];
    for (index, &byte) in block[2..2 + HALF].iter().enumerate() {
        weights[index] = signed(byte & 0x0F) - 8;
        weights[index + HALF] = signed(byte >> 4) - 8;
    }
    read_f16_le(block, 0) * input.scales[0] * lossy(dot_i8(&weights, input.quants))
}

pub(crate) fn q5_0(block: &[u8], input: Q8Blocks<'_>) -> f32 {
    let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let fifth = |bit: usize| u8::from((high_bits >> bit) & 1 != 0) << 4;
    let mut weights = [0_i8; Q8_BLOCK];
    for (index, &byte) in block[6..6 + HALF].iter().enumerate() {
        weights[index] = signed((byte & 0x0F) | fifth(index)) - 16;
        weights[index + HALF] = signed((byte >> 4) | fifth(index + HALF)) - 16;
    }
    read_f16_le(block, 0) * input.scales[0] * lossy(dot_i8(&weights, input.quants))
}

/// ggml `get_scale_min_k4`.
const fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// `Q4_K`: eight 32-element sub-blocks, each `d * scale * q - dmin * min`.
/// The `min` term needs only the input block's sum.
pub(crate) fn q4_k(block: &[u8], input: Q8Blocks<'_>) -> f32 {
    let scales = &block[4..16];
    let mut by_scale = 0.0_f32;
    let mut by_min = 0.0_f32;
    let (pairs, _) = block[16..144].as_chunks::<Q8_BLOCK>();
    let (activations, _) = input.quants.as_chunks::<Q8_BLOCK>();
    for (pair, packed) in pairs.iter().enumerate() {
        let mut low = [0_i8; Q8_BLOCK];
        let mut high = [0_i8; Q8_BLOCK];
        for ((l, h), &byte) in low.iter_mut().zip(high.iter_mut()).zip(packed) {
            *l = signed(byte & 0x0F);
            *h = signed(byte >> 4);
        }
        for (sub, weights) in [(2 * pair, &low), (2 * pair + 1, &high)] {
            let (scale, min) = scale_min_k4(sub, scales);
            let x_scale = input.scales[sub];
            let product = lossy(dot_i8(weights, &activations[sub]));
            by_scale = (x_scale * f32::from(scale)).mul_add(product, by_scale);
            by_min = (x_scale * f32::from(min)).mul_add(lossy(input.sums[sub]), by_min);
        }
    }
    read_f16_le(block, 2).mul_add(-by_min, read_f16_le(block, 0) * by_scale)
}

/// One `Q6_K` value: four low bits, two high bits, offset by 32.
const fn six_bit(low: u8, high: u8, shift: u32) -> i8 {
    signed(low | (((high >> shift) & 3) << 4)) - 32
}

/// The four 32-element runs of one `Q6_K` half block.
fn q6_k_runs(low: &[u8], high: &[u8]) -> [[i8; Q8_BLOCK]; 4] {
    let mut runs = [[0_i8; Q8_BLOCK]; 4];
    for (l, (&h, (&first, &second))) in high.iter().zip(low.iter().zip(&low[32..])).enumerate() {
        runs[0][l] = six_bit(first & 0x0F, h, 0);
        runs[1][l] = six_bit(second & 0x0F, h, 2);
        runs[2][l] = six_bit(first >> 4, h, 4);
        runs[3][l] = six_bit(second >> 4, h, 6);
    }
    runs
}

/// `Q6_K`: two halves of 128, each four 32-element runs whose two 16-element
/// halves carry their own signed scale.
pub(crate) fn q6_k(block: &[u8], input: Q8Blocks<'_>) -> f32 {
    let (activations, _) = input.quants.as_chunks::<HALF>();
    let mut total = 0.0_f32;
    for half in 0..2 {
        let runs = q6_k_runs(
            &block[64 * half..64 * half + 64],
            &block[128 + 32 * half..160 + 32 * half],
        );
        let scales = &block[192 + 8 * half..200 + 8 * half];
        for (run, weights) in runs.iter().enumerate() {
            let index = 4 * half + run;
            let (groups, _) = weights.as_chunks::<HALF>();
            let sum: i32 = groups
                .iter()
                .zip(&activations[2 * index..2 * index + 2])
                .zip(&scales[2 * run..2 * run + 2])
                .map(|((w, q), &scale)| i32::from(scale.cast_signed()) * dot_i8(w, q))
                .sum();
            total = input.scales[index].mul_add(lossy(sum), total);
        }
    }
    read_f16_le(block, 208) * total
}
