// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on NEON. Per unit, each 32-element run's weights are
//! unpacked once into sixteen registers and multiplied with every input of
//! the chunk.
//!
//! A weight `q` (`0..64`) is unpacked as `4 * (q - 32)`: its high bits
//! land on bits 6-7 with a single shift, `vsri` inserts the nibble below
//! them, and flipping bit 7 subtracts the offset (done once per half, on
//! the high-bit bytes). The integer sums are then four times the true
//! ones, which an exact shift undoes before the float conversion.

use core::arch::aarch64::{
    float32x4_t, int8x8_t, int8x16_t, int32x4_t, uint8x16_t, vaddq_f32, vandq_u8, vcvtq_f32_s32,
    vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_low_s16, vmovl_high_s16,
    vmovl_s8, vmovl_s16, vmulq_f32, vmulq_n_f32, vreinterpretq_s8_u8, vshlq_n_u8, vshrq_n_s32,
    vsriq_n_u8,
};

use super::{D, QH, SCALES, UNIT};
use crate::group::{Q8Superblocks, ROWS, TOKENS};
use crate::neon::vector::{Rows, halves, run};

/// Sixteen 16-byte lanes: the same four bytes of 32 per row, for eight rows.
type Lanes = [[u8; 16]; 16];

/// A unit as lanes: the low bits of the four quarters of each half (the
/// first and second 32 bytes of each half's 64), the high bits of the two
/// halves, the scales of the eight rows (one 8-byte lane per scale), and
/// `d`.
struct Unit<'a> {
    low: [&'a Lanes; 4],
    high: [&'a Lanes; 2],
    scales: &'a [[u8; 8]; 16],
    d: [u8; 16],
}

// `split` hard-codes these offsets in its patterns.
const _: () = assert!(QH == 64 * 16 && SCALES == QH + 32 * 16 && D == SCALES + 8 * 16);
const _: () = assert!(UNIT == D + 16);

fn split(unit: &[u8; UNIT]) -> Unit<'_> {
    let lanes: &[[u8; 16]; 105] = bytemuck::cast_ref(unit);
    let [eights @ .., d] = lanes;
    let eights: &[[[u8; 16]; 8]; 13] = bytemuck::cast_ref(eights);
    let [bits @ .., scales] = eights;
    let [l0, l1, l2, l3, h0, h1]: &[Lanes; 6] = bytemuck::cast_ref(bits);
    Unit {
        low: [l0, l1, l2, l3],
        high: [h0, h1],
        scales: bytemuck::cast_ref(scales),
        d: *d,
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn rows_i32(scales: int8x8_t) -> Rows<int32x4_t> {
    let wide = vmovl_s8(scales);
    [vmovl_s16(vget_low_s16(wide)), vmovl_high_s16(wide)]
}

/// A half's high-bit bytes with bit 7 of each run's two bits flipped.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn flipped(high: &Lanes) -> [uint8x16_t; 16] {
    let mut out = [vdupq_n_u8(0); 16];
    for (slot, &lane) in out.iter_mut().zip(high) {
        *slot = veorq_u8(bytemuck::cast(lane), vdupq_n_u8(0xaa));
    }
    out
}

/// Run `run` (`0..4`) of a half: `low` holds its nibbles, `high` its two
/// high bits at `2 * run`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn unpack(low: &Lanes, high: &[uint8x16_t; 16], run: usize) -> [int8x16_t; 16] {
    let mut weights = [vdupq_n_s8(0); 16];
    for ((weight, &low), &high) in weights.iter_mut().zip(low).zip(high) {
        let low: uint8x16_t = bytemuck::cast(low);
        let nibble = if run < 2 {
            vshlq_n_u8::<4>(low)
        } else {
            vandq_u8(low, vdupq_n_u8(0xf0))
        };
        let top = match run {
            0 => vshlq_n_u8::<6>(high),
            1 => vshlq_n_u8::<4>(high),
            2 => vshlq_n_u8::<2>(high),
            _ => high,
        };
        *weight = vreinterpretq_s8_u8(vsriq_n_u8::<2>(top, nibble));
    }
    weights
}

/// `acc + d * d8 * sum` per row, as the portable kernel rounds it, from
/// four times the integer sum.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn combine(
    acc: Rows<float32x4_t>,
    sum: Rows<int32x4_t>,
    d: Rows<float32x4_t>,
    d8: f32,
) -> Rows<float32x4_t> {
    let mut out = acc;
    for ((slot, sum), d) in out.iter_mut().zip(sum).zip(d) {
        let term = vmulq_f32(vmulq_n_f32(d, d8), vcvtq_f32_s32(vshrq_n_s32::<2>(sum)));
        *slot = vaddq_f32(*slot, term);
    }
    out
}

/// One unit's integer sums for every input of the chunk.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn sums<const N: usize>(
    unit: &Unit<'_>,
    block: usize,
    inputs: &[Q8Superblocks<'_>],
) -> [Rows<int32x4_t>; N] {
    let mut sums = [[vdupq_n_s32(0); 2]; N];
    for (half, high) in unit.high.iter().enumerate() {
        let high = flipped(high);
        for quarter in 0..4 {
            let weights = unpack(unit.low[2 * half + quarter % 2], &high, quarter);
            let index = 4 * half + quarter;
            let pair = [
                rows_i32(bytemuck::cast(unit.scales[2 * index])),
                rows_i32(bytemuck::cast(unit.scales[2 * index + 1])),
            ];
            for (sum, input) in sums.iter_mut().zip(inputs) {
                if let Some(quants) = input.quants.get(block) {
                    let runs: &[[i8; 32]; 8] = bytemuck::cast_ref(quants);
                    *sum = run(*sum, &weights, runs[index], pair);
                }
            }
        }
    }
    sums
}

/// [`group`] for at most `N` inputs.
#[target_feature(enable = "neon,dotprod")]
fn chunk<const N: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Superblocks<'_>],
    out: &mut [[f32; ROWS]],
) {
    let mut acc = [[vdupq_n_f32(-0.0); 2]; N];
    for (block, unit) in units.iter().enumerate() {
        let unit = split(unit);
        let d = halves(unit.d);
        let sums = sums::<N>(&unit, block, inputs);
        for ((acc, sum), input) in acc.iter_mut().zip(sums).zip(inputs) {
            if let Some(scale) = input.scales.get(block) {
                *acc = combine(*acc, sum, d, scale[0]);
            }
        }
    }
    for (rows, acc) in out.iter_mut().zip(acc) {
        *rows = bytemuck::cast(acc);
    }
}

#[target_feature(enable = "neon,dotprod")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    // One input, as in decoding, gets a body whose sums all stay in registers.
    if inputs.len() == 1 {
        return chunk::<1>(units, inputs, out);
    }
    for (inputs, out) in inputs.chunks(TOKENS).zip(out.chunks_mut(TOKENS)) {
        chunk::<TOKENS>(units, inputs, out);
    }
}
