// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q5_0` weights against `Q8_0` activations, eight rows at a time.
//!
//! The kernel reads units: the `k`-th 22-byte block of eight rows, 176
//! bytes, interleaved so that one 16-byte load holds the same four nibble
//! bytes of four rows and one `vdotq_laneq_s32` accumulates a lane per row.
//! Row `r`'s block bytes sit at:
//!
//! - `d` (f16): `D + 2r`.
//! - its four fifth-bit bytes (bit `e` of the little-endian word is
//!   weight `e`'s): `QH + 4r`.
//! - nibble bytes `4c..4c + 4` (`c < 4`): `QS + 32c + 4r`; byte `i`
//!   holds weight `i` in its low nibble and weight `i + 16` in its high.

#[cfg(any(miri, test))]
mod scalar;
#[cfg(not(miri))]
mod vector;

use super::group::{Q8Blocks, ROWS};

#[cfg(miri)]
use scalar as body;
#[cfg(not(miri))]
use vector as body;

/// Bytes per unit: one 22-byte block of each of eight rows.
pub const UNIT: usize = 176;
/// Offset of the rows' `d` values in a unit.
pub const D: usize = 0;
/// Offset of the rows' fifth bits in a unit.
pub const QH: usize = 16;
/// Offset of the rows' nibble bytes in a unit.
pub const QS: usize = 48;
/// Bytes of one row side by side in a unit.
pub const LANE: usize = 4;

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d * d8 * isum` from `-0.0`, with exact integer sums, in
/// the order and roundings of the portable `Q5_0` kernels.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q5_0_group(group: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    // SAFETY: this module compiles only under `cfg(all(target_arch =
    // "aarch64", target_feature = "dotprod"))`, so the build target
    // guarantees that every CPU running this binary has NEON and dotprod.
    unsafe { body::group(group, inputs, out) }
}

#[cfg(test)]
mod tests;
