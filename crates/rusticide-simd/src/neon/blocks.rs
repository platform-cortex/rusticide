// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON body every 32-element block format shares: per unit, the
//! format unpacks the eight rows' weights once into sixteen registers, and
//! each input of the chunk adds `d * d8 * isum` per row.

use core::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vmulq_f32,
    vmulq_n_f32,
};

use super::group::{Q8Blocks, ROWS, TOKENS};
use super::vector::{Rows, dot, halves};

/// A unit's `d` bytes, one half per row, and its weights as `dot` reads
/// them.
pub(crate) type Unpacked = ([u8; 16], [int8x16_t; 16]);

/// `acc + d * d8 * sum` per row, as the portable kernels round it.
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
        *slot = vaddq_f32(*slot, vmulq_f32(vmulq_n_f32(d, d8), vcvtq_f32_s32(sum)));
    }
    out
}

/// [`group`] for at most `N` inputs.
#[target_feature(enable = "neon,dotprod")]
fn chunk<const UNIT: usize, const N: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
    unpack: impl Fn(&[u8; UNIT]) -> Unpacked,
) {
    let mut acc = [[vdupq_n_f32(-0.0); 2]; N];
    for (block, unit) in units.iter().enumerate() {
        let (d, weights) = unpack(unit);
        let d = halves(d);
        for (acc, input) in acc.iter_mut().zip(inputs) {
            if let (Some(&quants), Some(&d8)) = (input.quants.get(block), input.scales.get(block)) {
                *acc = combine(*acc, dot(&weights, quants), d, d8);
            }
        }
    }
    for (rows, acc) in out.iter_mut().zip(acc) {
        *rows = bytemuck::cast(acc);
    }
}

/// The products of a group's units with each input, every token computed
/// alone and in the same order.
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn group<const UNIT: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
    unpack: impl Fn(&[u8; UNIT]) -> Unpacked + Copy,
) {
    // One input, as in decoding, gets a body whose sums all stay in registers.
    if inputs.len() == 1 {
        return chunk::<UNIT, 1>(units, inputs, out, unpack);
    }
    for (inputs, out) in inputs.chunks(TOKENS).zip(out.chunks_mut(TOKENS)) {
        chunk::<UNIT, TOKENS>(units, inputs, out, unpack);
    }
}
