// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer dot kernels: a quantized row against a [`Q8Input`]. Each format's
//! row function is monomorphized over fixed-size blocks, so every inner loop
//! has a compile-time length; weights are unpacked to `i8` on the stack and
//! multiplied in `i32`, which the compiler vectorizes (to `sdot` where the
//! target has it). Floats appear only once per 16 or 32 elements.

use crate::half_bits::read_f16_le;
use crate::q8_input::{Q8_BLOCK, Q8Input, Span};

const HALF: usize = Q8_BLOCK / 2;

#[expect(
    clippy::cast_precision_loss,
    reason = "a block sum past 2^24 loses low bits, as ggml's int-to-float conversion does"
)]
const fn lossy(value: i32) -> f32 {
    value as f32
}

/// Reinterprets an unsigned value below 128 as `i8`.
const fn signed(value: u8) -> i8 {
    value.cast_signed()
}

/// Sums `kernel` over the `BYTES`-byte blocks of `row` and the [`Span`]s of
/// `K` activation blocks each covers.
fn dot_row<const BYTES: usize, const K: usize>(
    row: &[u8],
    input: &Q8Input,
    kernel: impl Fn(&[u8; BYTES], Span<'_, K>) -> f32,
) -> f32 {
    let (blocks, _) = row.as_chunks::<BYTES>();
    blocks
        .iter()
        .zip(input.spans::<K>())
        .map(|(block, span)| kernel(block, span))
        .sum()
}

pub(crate) fn q8_0(row: &[u8], input: &Q8Input) -> f32 {
    dot_row::<34, 1>(row, input, |block, span| {
        let product: i32 = block[2..]
            .iter()
            .zip(&span.quants[0])
            .map(|(&w, &q)| i32::from(w.cast_signed()) * i32::from(q))
            .sum();
        read_f16_le(block, 0) * span.scales[0] * lossy(product)
    })
}

pub(crate) fn q4_0(row: &[u8], input: &Q8Input) -> f32 {
    dot_row::<18, 1>(row, input, |block, span| {
        let (low_q, high_q) = span.quants[0].split_at(HALF);
        let mut sum = 0_i32;
        for ((&byte, &lq), &hq) in block[2..].iter().zip(low_q).zip(high_q) {
            sum += i32::from(signed(byte & 0x0F) - 8) * i32::from(lq);
            sum += i32::from(signed(byte >> 4) - 8) * i32::from(hq);
        }
        read_f16_le(block, 0) * span.scales[0] * lossy(sum)
    })
}

/// Byte `b` spread to eight bytes: byte `i` is `0x10` when bit `i` of `b` is
/// set, the fifth bit of eight `Q5_0` values at once (ggml `table_b2b_1`).
const FIFTH_BITS: [u64; 256] = {
    let mut table = [0_u64; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            if byte >> bit & 1 != 0 {
                table[byte] |= 0x10 << (8 * bit);
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

pub(crate) fn q5_0(row: &[u8], input: &Q8Input) -> f32 {
    dot_row::<22, 1>(row, input, |block, span| {
        let mut fifth = [0_u8; Q8_BLOCK];
        for (spread, &bits) in fifth.as_chunks_mut::<8>().0.iter_mut().zip(&block[2..6]) {
            *spread = FIFTH_BITS[usize::from(bits)].to_le_bytes();
        }
        let quants = &span.quants[0];
        let mut sum = 0_i32;
        for (i, &byte) in block[6..].iter().enumerate() {
            sum += i32::from(signed((byte & 0x0F) | fifth[i]) - 16) * i32::from(quants[i]);
            sum +=
                i32::from(signed((byte >> 4) | fifth[i + HALF]) - 16) * i32::from(quants[i + HALF]);
        }
        read_f16_le(block, 0) * span.scales[0] * lossy(sum)
    })
}

/// The dot products of 32 packed bytes' low and high nibbles, each offset by
/// -8, with two activation blocks.
#[inline]
fn pair_dots(
    packed: &[u8; Q8_BLOCK],
    low_q: &[i8; Q8_BLOCK],
    high_q: &[i8; Q8_BLOCK],
) -> (i32, i32) {
    let mut low = 0_i32;
    let mut high = 0_i32;
    for ((&byte, &lq), &hq) in packed.iter().zip(low_q).zip(high_q) {
        low += i32::from(signed(byte & 0x0F) - 8) * i32::from(lq);
        high += i32::from(signed(byte >> 4) - 8) * i32::from(hq);
    }
    (low, high)
}

/// The eight 6-bit scales and mins packed in a `Q4_K` block's 12 bytes,
/// unpacked four at a time in `u32` words as ggml does.
fn k4_scales_and_mins(packed: &[u8]) -> ([u8; 8], [u8; 8]) {
    const LOW_SIX: u32 = 0x3F3F_3F3F;
    const LOW_FOUR: u32 = 0x0F0F_0F0F;
    const LOW_TWO: u32 = 0x0303_0303;
    let word =
        |i: usize| u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]]);
    let (first, second, third) = (word(0), word(4), word(8));
    let high_scales = (third & LOW_FOUR) | (((first >> 6) & LOW_TWO) << 4);
    let high_mins = ((third >> 4) & LOW_FOUR) | (((second >> 6) & LOW_TWO) << 4);
    let join = |low: u32, high: u32| (u64::from(low) | (u64::from(high) << 32)).to_le_bytes();
    (
        join(first & LOW_SIX, high_scales),
        join(second & LOW_SIX, high_mins),
    )
}

/// `Q4_K`: eight 32-element sub-blocks, each `d * scale * q - dmin * min`.
/// Its activations are `Q8_K`, one scale per block, so the sub-blocks
/// accumulate in `i32` and convert to float once, as ggml does.
pub(crate) fn q4_k(row: &[u8], input: &Q8Input) -> f32 {
    dot_row::<144, 8>(row, input, |block, span| {
        let (scales, mins) = k4_scales_and_mins(&block[4..16]);
        let (pairs, _) = block[16..].as_chunks::<Q8_BLOCK>();
        let (quant_pairs, _) = span.quants.as_chunks::<2>();
        let (scale_pairs, _) = scales.as_chunks::<2>();
        let mut sum = 0_i32;
        for ((packed, [low_q, high_q]), [low_scale, high_scale]) in
            pairs.iter().zip(quant_pairs).zip(scale_pairs)
        {
            let (low, high) = pair_dots(packed, low_q, high_q);
            sum += i32::from(*low_scale) * low;
            sum += i32::from(*high_scale) * high;
        }
        // The nibbles were offset by -8 so both operands are signed, which is
        // what lets the compiler pick a signed dot instruction; the offset
        // comes back through the block sums, alongside the min term.
        let mut min_sum = 0_i32;
        for ((&scale, &min), &block_sum) in scales.iter().zip(&mins).zip(span.sums) {
            sum += 8 * i32::from(scale) * block_sum;
            min_sum += i32::from(min) * block_sum;
        }
        let d = read_f16_le(block, 0);
        let dmin = read_f16_le(block, 2);
        span.scales[0] * d.mul_add(lossy(sum), -(dmin * lossy(min_sum)))
    })
}

/// One `Q6_K` value: four low bits, two high bits, offset by 32.
const fn six_bit(low: u8, high: u8, shift: u32) -> i8 {
    signed(low | (((high >> shift) & 3) << 4)) - 32
}

/// `Q6_K`: two halves of 128, each four 32-element runs whose two 16-element
/// halves carry their own signed scale. Like `Q4_K`, it pairs with `Q8_K`
/// activations and accumulates in `i32`.
pub(crate) fn q6_k(row: &[u8], input: &Q8Input) -> f32 {
    dot_row::<210, 8>(row, input, |block, span| {
        let mut sum = 0_i32;
        let (halves, _) = span.quants.as_chunks::<4>();
        for (half, quants) in halves.iter().enumerate() {
            let low = &block[64 * half..64 * half + 64];
            let high = &block[128 + 32 * half..160 + 32 * half];
            let scales = &block[192 + 8 * half..200 + 8 * half];
            for group in 0..2 {
                let mut runs = [0_i32; 4];
                for l in HALF * group..HALF * (group + 1) {
                    let (first, second, bits) = (low[l], low[l + 32], high[l]);
                    runs[0] += i32::from(six_bit(first & 0x0F, bits, 0)) * i32::from(quants[0][l]);
                    runs[1] += i32::from(six_bit(second & 0x0F, bits, 2)) * i32::from(quants[1][l]);
                    runs[2] += i32::from(six_bit(first >> 4, bits, 4)) * i32::from(quants[2][l]);
                    runs[3] += i32::from(six_bit(second >> 4, bits, 6)) * i32::from(quants[3][l]);
                }
                for (run, partial) in runs.iter().enumerate() {
                    sum += i32::from(scales[2 * run + group].cast_signed()) * partial;
                }
            }
        }
        read_f16_le(block, 208) * span.scales[0] * lossy(sum)
    })
}
