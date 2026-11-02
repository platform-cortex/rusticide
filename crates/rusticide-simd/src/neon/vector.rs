// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NEON helpers the group kernels share. Rows `0..4` and `4..8` of a unit
//! each fill one four-lane vector, so a row's sums never need a horizontal
//! reduction.

use core::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint16x8_t, uint32x4_t, vandq_u32, vbslq_u32, vcgeq_u32,
    vdotq_laneq_s32, vdupq_n_f32, vdupq_n_s32, vdupq_n_u32, vget_low_u16, vmlaq_s32,
    vmovl_high_u16, vmovl_u16, vmulq_f32, vorrq_u32, vreinterpretq_f32_u32, vreinterpretq_u32_f32,
    vshlq_n_u32,
};

/// A value per row: rows `0..4`, then rows `4..8`.
pub(crate) type Rows<T> = [T; 2];

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

/// Eight little-endian halves, one per row, widened exactly.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn halves(bytes: [u8; 16]) -> Rows<float32x4_t> {
    let bits: uint16x8_t = bytemuck::cast(bytes);
    [
        widen_f16(vmovl_u16(vget_low_u16(bits))),
        widen_f16(vmovl_high_u16(bits)),
    ]
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

/// Adds one 32-element run's dot products to `sum`, its first 16 elements
/// times `scales[0]` and its last 16 times `scales[1]`; weight vector
/// `2 * chunk + quad` holds elements `4 * chunk..` of rows `4 * quad..`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn run(
    sum: Rows<int32x4_t>,
    weights: &[int8x16_t; 16],
    quants: [i8; 32],
    scales: [Rows<int32x4_t>; 2],
) -> Rows<int32x4_t> {
    let [first, second]: [int8x16_t; 2] = bytemuck::cast(quants);
    let zero = vdupq_n_s32(0);
    let chunks = |quad: usize, from: usize| -> [int8x16_t; 4] {
        core::array::from_fn(|chunk| weights[2 * (from + chunk) + quad])
    };
    let mut out = sum;
    for (quad, slot) in out.iter_mut().enumerate() {
        let low = vmlaq_s32(*slot, four(zero, chunks(quad, 0), first), scales[0][quad]);
        *slot = vmlaq_s32(low, four(zero, chunks(quad, 4), second), scales[1][quad]);
    }
    out
}
