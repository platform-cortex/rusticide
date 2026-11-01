// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batched integer dot kernels: one quantized row against several
//! [`Q8Input`]s. Unpacking a weight block costs about as much as its dot
//! product, so each block is unpacked once, on the stack, for every input.
//!
//! Each output adds its blocks' terms in row order, from the `-0.0` that
//! `f32`'s `Sum` starts at, and each term is computed as the matching
//! single-input kernel computes it: its integer sums are exact, so only the
//! float operations need the same order. The two therefore agree to the bit.

use rusticide_quantize::read_f16_le;

use super::{FIFTH_BITS, HALF, k4_scales_and_mins, lossy, signed, six_bit};
use crate::q8_input::{Q8_BLOCK, Q8Input, Span};

/// Blocks unpacked at a time: enough that finding an input's spans, once
/// per run, stays small next to the dots, few enough to stay on the stack
/// (1.1 KiB for `Q5_0`, 1.2 KiB for the K formats).
const Q5_0_RUN: usize = 32;
const K_RUN: usize = 4;

fn int_dot<const N: usize>(weights: &[i8; N], quants: &[i8; N]) -> i32 {
    weights
        .iter()
        .zip(quants)
        .map(|(&w, &q)| i32::from(w) * i32::from(q))
        .sum()
}

/// Two dot products in one loop, the shape the compiler turns into `sdot`
/// reliably once the loops around it are unrolled.
fn pair_dots<const N: usize>(
    low_w: &[i8; N],
    high_w: &[i8; N],
    low_q: &[i8; N],
    high_q: &[i8; N],
) -> (i32, i32) {
    let mut low = 0_i32;
    let mut high = 0_i32;
    for (((&lw, &hw), &lq), &hq) in low_w.iter().zip(high_w).zip(low_q).zip(high_q) {
        low += i32::from(lw) * i32::from(lq);
        high += i32::from(hw) * i32::from(hq);
    }
    (low, high)
}

/// Sums `kernel` over the `BYTES`-byte blocks of `row`, unpacked `RUN` at a
/// time, into one slot of `out` per input.
fn dot_rows<const BYTES: usize, const K: usize, const RUN: usize, W>(
    row: &[u8],
    inputs: &[Q8Input],
    out: &mut [f32],
    unpack: impl Fn(&[u8; BYTES]) -> W,
    kernel: impl Fn(&W, Span<'_, K>) -> f32,
) {
    out.fill(-0.0);
    let (blocks, _) = row.as_chunks::<BYTES>();
    for (run, run_blocks) in blocks.chunks(RUN).enumerate() {
        // A short last run is padded with all-zero blocks, which no span
        // reaches: a plain array keeps the loop below free of branches.
        let unpacked: [W; RUN] =
            std::array::from_fn(|i| unpack(run_blocks.get(i).unwrap_or(&[0; BYTES])));
        for (slot, input) in out.iter_mut().zip(inputs) {
            let spans = input.spans_from::<K>(run * RUN);
            for (weights, span) in unpacked[..run_blocks.len()].iter().zip(spans) {
                *slot += kernel(weights, span);
            }
        }
    }
}

/// The 32 weights of a `Q5_0` block, in activation order, and its scale.
fn q5_0_block(block: &[u8; 22]) -> ([i8; Q8_BLOCK], f32) {
    let mut fifth = [0_u8; Q8_BLOCK];
    for (spread, &bits) in fifth.as_chunks_mut::<8>().0.iter_mut().zip(&block[2..6]) {
        *spread = FIFTH_BITS[usize::from(bits)].to_le_bytes();
    }
    let mut weights = [0_i8; Q8_BLOCK];
    let (low, high) = weights.split_at_mut(HALF);
    for (i, &byte) in block[6..].iter().enumerate() {
        low[i] = signed((byte & 0x0F) | fifth[i]) - 16;
        high[i] = signed((byte >> 4) | fifth[i + HALF]) - 16;
    }
    (weights, read_f16_le(block, 0))
}

pub(crate) fn q5_0_rows(row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    dot_rows::<22, 1, Q5_0_RUN, _>(row, inputs, out, q5_0_block, |(weights, d), span| {
        d * span.scales[0] * lossy(int_dot(weights, &span.quants[0]))
    });
}

/// An unpacked `Q4_K` block: the raw nibbles of its eight sub-blocks.
struct Q4K {
    nibbles: [[i8; Q8_BLOCK]; 8],
    scales: [u8; 8],
    mins: [u8; 8],
    d: f32,
    dmin: f32,
}

fn q4_k_block(block: &[u8; 144]) -> Q4K {
    let (scales, mins) = k4_scales_and_mins(&block[4..16]);
    let mut nibbles = [[0_i8; Q8_BLOCK]; 8];
    let (pairs, _) = block[16..].as_chunks::<Q8_BLOCK>();
    for ([low, high], packed) in nibbles.as_chunks_mut::<2>().0.iter_mut().zip(pairs) {
        for ((low, high), &byte) in low.iter_mut().zip(high.iter_mut()).zip(packed) {
            (*low, *high) = (signed(byte & 0x0F), signed(byte >> 4));
        }
    }
    Q4K {
        nibbles,
        scales,
        mins,
        d: read_f16_le(block, 0),
        dmin: read_f16_le(block, 2),
    }
}

/// The single-input kernel offsets the nibbles by -8 and adds `8 * scale`
/// times each block sum back; the raw nibbles give the same integer.
pub(crate) fn q4_k_rows(row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    dot_rows::<144, 8, K_RUN, _>(row, inputs, out, q4_k_block, |block, span| {
        let (nibble_pairs, _) = block.nibbles.as_chunks::<2>();
        let (quant_pairs, _) = span.quants.as_chunks::<2>();
        let (scale_pairs, _) = block.scales.as_chunks::<2>();
        let mut sum = 0_i32;
        for (([low_n, high_n], [low_q, high_q]), [low_scale, high_scale]) in
            nibble_pairs.iter().zip(quant_pairs).zip(scale_pairs)
        {
            let (low, high) = pair_dots(low_n, high_n, low_q, high_q);
            sum += i32::from(*low_scale) * low;
            sum += i32::from(*high_scale) * high;
        }
        let mut min_sum = 0_i32;
        for (&min, &block_sum) in block.mins.iter().zip(span.sums) {
            min_sum += i32::from(min) * block_sum;
        }
        let (d, dmin) = (block.d, block.dmin);
        span.scales[0] * d.mul_add(lossy(sum), -(dmin * lossy(min_sum)))
    });
}

/// An unpacked `Q6_K` block: sixteen runs of 16 values, each with its scale.
struct Q6K {
    values: [[i8; HALF]; 16],
    scales: [i8; 16],
    d: f32,
}

fn q6_k_block(block: &[u8; 210]) -> Q6K {
    let mut values = [[0_i8; HALF]; 16];
    let flat = values.as_flattened_mut();
    for half in 0..2 {
        let low = &block[64 * half..64 * half + 64];
        let high = &block[128 + 32 * half..160 + 32 * half];
        let out = &mut flat[128 * half..128 * half + 128];
        for l in 0..Q8_BLOCK {
            let (first, second, bits) = (low[l], low[l + 32], high[l]);
            out[l] = six_bit(first & 0x0F, bits, 0);
            out[l + 32] = six_bit(second & 0x0F, bits, 2);
            out[l + 64] = six_bit(first >> 4, bits, 4);
            out[l + 96] = six_bit(second >> 4, bits, 6);
        }
    }
    let mut scales = [0_i8; 16];
    for (scale, &byte) in scales.iter_mut().zip(&block[192..208]) {
        *scale = byte.cast_signed();
    }
    Q6K {
        values,
        scales,
        d: read_f16_le(block, 208),
    }
}

pub(crate) fn q6_k_rows(row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    dot_rows::<210, 8, K_RUN, _>(row, inputs, out, q6_k_block, |block, span| {
        let (quants, _) = span.quants.as_flattened().as_chunks::<HALF>();
        let (quant_pairs, _) = quants.as_chunks::<2>();
        let (value_pairs, _) = block.values.as_chunks::<2>();
        let (scale_pairs, _) = block.scales.as_chunks::<2>();
        let mut sum = 0_i32;
        for (([low_v, high_v], [low_q, high_q]), [low_scale, high_scale]) in
            value_pairs.iter().zip(quant_pairs).zip(scale_pairs)
        {
            let (low, high) = pair_dots(low_v, high_v, low_q, high_q);
            sum += i32::from(*low_scale) * low;
            sum += i32::from(*high_scale) * high;
        }
        block.d * span.scales[0] * lossy(sum)
    });
}
