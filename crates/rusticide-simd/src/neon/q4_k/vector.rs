// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on NEON. Rows `0..4` and `4..8` of a unit each fill one
//! four-lane vector, so a row's sums never need a horizontal reduction.
//! Per unit, each 32-element run's 256 weight bytes are unpacked once
//! into sixteen registers and multiplied with every input of the chunk.

use core::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint8x8_t, uint8x16_t, uint16x8_t, uint32x4_t, vaddq_f32,
    vaddq_s32, vand_u8, vandq_u8, vandq_u32, vbslq_u32, vcgeq_u32, vcvtq_f32_s32, vdotq_laneq_s32,
    vdup_n_u8, vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vdupq_n_u32, vfmaq_f32,
    vget_low_u16, vmlaq_n_s32, vmlaq_s32, vmovl_high_u16, vmovl_u8, vmovl_u16, vmulq_f32,
    vmulq_n_f32, vnegq_f32, vorr_u8, vorrq_u32, vreinterpretq_f32_u32, vreinterpretq_s8_u8,
    vreinterpretq_s32_u32, vreinterpretq_u32_f32, vshl_n_u8, vshlq_n_u32, vshr_n_u8, vshrq_n_u8,
};

use super::{Q8Superblocks, ROWS, TOKENS, UNIT};

/// A value per row: rows `0..4`, then rows `4..8`.
type Rows<T> = [T; 2];

/// What every input shares for one unit: the rows' float scales and their
/// 6-bit run scales and mins, a vector pair per run.
struct Header {
    d: Rows<float32x4_t>,
    dmin: Rows<float32x4_t>,
    scales: [Rows<int32x4_t>; 8],
    mins: [Rows<int32x4_t>; 8],
}

/// A unit as 16-byte lanes: `d`, `dmin`, six of packed scales, then the
/// quants as four pairs of runs of sixteen lanes each.
fn lanes(unit: &[u8; UNIT]) -> &[[u8; 16]; 72] {
    bytemuck::cast_ref(unit)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn widen_f16(bits: uint32x4_t) -> float32x4_t {
    let magnitude = vandq_u32(bits, vdupq_n_u32(0x7fff));
    let shifted = vshlq_n_u32::<13>(magnitude);
    let rebiased = vmulq_f32(
        vreinterpretq_f32_u32(shifted),
        vdupq_n_f32(f32::from_bits(0x7780_0000)),
    );
    let special = vcgeq_u32(magnitude, vdupq_n_u32(0x7c00));
    let widened = vbslq_u32(
        special,
        vorrq_u32(shifted, vdupq_n_u32(0x7f80_0000)),
        vreinterpretq_u32_f32(rebiased),
    );
    let sign = vshlq_n_u32::<16>(vandq_u32(bits, vdupq_n_u32(0x8000)));
    vreinterpretq_f32_u32(vorrq_u32(widened, sign))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn halves(bytes: [u8; 16]) -> Rows<float32x4_t> {
    let bits: uint16x8_t = bytemuck::cast(bytes);
    [
        widen_f16(vmovl_u16(vget_low_u16(bits))),
        widen_f16(vmovl_high_u16(bits)),
    ]
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn rows_i32(bytes: uint8x8_t) -> Rows<int32x4_t> {
    let wide = vmovl_u8(bytes);
    [
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide))),
        vreinterpretq_s32_u32(vmovl_high_u16(wide)),
    ]
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn header(lanes: &[[u8; 16]; 72]) -> Header {
    let [d, dmin, s0, s1, s2, s3, s4, s5, ..] = *lanes;
    let packed: [uint8x8_t; 12] = bytemuck::cast([s0, s1, s2, s3, s4, s5]);
    let (six, four) = (vdup_n_u8(63), vdup_n_u8(0x0f));
    let mut scales = [[vdupq_n_s32(0); 2]; 8];
    let mut mins = [[vdupq_n_s32(0); 2]; 8];
    for low in 0..4 {
        let (scale, min, high) = (packed[low], packed[low + 4], packed[low + 8]);
        let (scale_top, min_top) = (vshr_n_u8::<6>(scale), vshr_n_u8::<6>(min));
        scales[low] = rows_i32(vand_u8(scale, six));
        mins[low] = rows_i32(vand_u8(min, six));
        scales[low + 4] = rows_i32(vorr_u8(vand_u8(high, four), vshl_n_u8::<4>(scale_top)));
        mins[low + 4] = rows_i32(vorr_u8(vshr_n_u8::<4>(high), vshl_n_u8::<4>(min_top)));
    }
    Header {
        d: halves(d),
        dmin: halves(dmin),
        scales,
        mins,
    }
}

/// Four `sdot`s into `sum`, weight vector `i` against lane `i` of `quants`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn four(sum: int32x4_t, weights: [int8x16_t; 4], quants: int8x16_t) -> int32x4_t {
    let sum = vdotq_laneq_s32::<0>(sum, weights[0], quants);
    let sum = vdotq_laneq_s32::<1>(sum, weights[1], quants);
    let sum = vdotq_laneq_s32::<2>(sum, weights[2], quants);
    vdotq_laneq_s32::<3>(sum, weights[3], quants)
}

/// Adds `scale` times one run's dot products to `sum`; weight vector
/// `2 * chunk + quad` holds elements `4 * chunk..` of rows `4 * quad..`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn run(
    sum: Rows<int32x4_t>,
    weights: &[int8x16_t; 16],
    quants: [i8; 32],
    scale: Rows<int32x4_t>,
) -> Rows<int32x4_t> {
    let [first, second]: [int8x16_t; 2] = bytemuck::cast(quants);
    let zero = vdupq_n_s32(0);
    let chunks = |quad: usize, from: usize| -> [int8x16_t; 4] {
        core::array::from_fn(|chunk| weights[2 * (from + chunk) + quad])
    };
    let mut out = sum;
    for (quad, slot) in out.iter_mut().enumerate() {
        let low = four(zero, chunks(quad, 0), first);
        let high = four(zero, chunks(quad, 4), second);
        *slot = vmlaq_s32(*slot, vaddq_s32(low, high), scale[quad]);
    }
    out
}

/// `acc + d8 * (d * sum - dmin * min_sum)` per row, as the portable kernel
/// rounds it.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn combine(
    acc: Rows<float32x4_t>,
    sum: Rows<int32x4_t>,
    header: &Header,
    block_sums: &[i32; 8],
    d8: f32,
) -> Rows<float32x4_t> {
    let mut out = acc;
    for (h, slot) in out.iter_mut().enumerate() {
        let mut min_sum = vdupq_n_s32(0);
        for (mins, &block_sum) in header.mins.iter().zip(block_sums) {
            min_sum = vmlaq_n_s32(min_sum, mins[h], block_sum);
        }
        let min_term = vmulq_f32(header.dmin[h], vcvtq_f32_s32(min_sum));
        let term = vfmaq_f32(vnegq_f32(min_term), header.d[h], vcvtq_f32_s32(sum[h]));
        *slot = vaddq_f32(*slot, vmulq_n_f32(term, d8));
    }
    out
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn unpack(raw: &[[u8; 16]; 16], high: bool) -> [int8x16_t; 16] {
    let mut weights = [vdupq_n_s8(0); 16];
    for (weight, &lane) in weights.iter_mut().zip(raw) {
        let lane: uint8x16_t = bytemuck::cast(lane);
        let nibbles = if high {
            vshrq_n_u8::<4>(lane)
        } else {
            vandq_u8(lane, vdupq_n_u8(0x0f))
        };
        *weight = vreinterpretq_s8_u8(nibbles);
    }
    weights
}

/// [`group`] for at most [`TOKENS`] inputs.
#[target_feature(enable = "neon,dotprod")]
fn chunk(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    let mut acc = [[vdupq_n_f32(-0.0); 2]; TOKENS];
    for (block, unit) in units.iter().enumerate() {
        let lanes = lanes(unit);
        let header = header(lanes);
        let [_, _, _, _, _, _, _, _, quants @ ..] = lanes;
        let pairs: &[[[u8; 16]; 16]; 4] = bytemuck::cast_ref(quants);
        let mut sums = [[vdupq_n_s32(0); 2]; TOKENS];
        for (pair, raw) in pairs.iter().enumerate() {
            for half in [false, true] {
                let weights = unpack(raw, half);
                let index = 2 * pair + usize::from(half);
                for (sum, input) in sums.iter_mut().zip(inputs) {
                    if let Some(quants) = input.quants.get(block) {
                        let runs: &[[i8; 32]; 8] = bytemuck::cast_ref(quants);
                        *sum = run(*sum, &weights, runs[index], header.scales[index]);
                    }
                }
            }
        }
        for ((acc, sum), input) in acc.iter_mut().zip(&sums).zip(inputs) {
            if let (Some(block_sums), Some(scale)) =
                (input.sums.get(block), input.scales.get(block))
            {
                *acc = combine(*acc, *sum, &header, block_sums, scale[0]);
            }
        }
    }
    for (rows, acc) in out.iter_mut().zip(acc) {
        *rows = bytemuck::cast(acc);
    }
}

#[target_feature(enable = "neon,dotprod")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    for (inputs, out) in inputs.chunks(TOKENS).zip(out.chunks_mut(TOKENS)) {
        chunk(units, inputs, out);
    }
}
