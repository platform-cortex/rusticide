// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q6_K` weights against `Q8_K` activations, eight rows at a time.
//!
//! The kernel reads units: the `k`-th 210-byte block of eight rows, 1680
//! bytes, interleaved so that one 16-byte load holds the same four low-bit
//! (or high-bit) bytes of four rows and one `vdotq_laneq_s32` accumulates
//! a lane per row. Row `r`'s block bytes sit at:
//!
//! - its 128 low-bit bytes `4c..4c + 4` (`c < 32`): `QL + 32c + 4r`.
//! - its 64 high-bit bytes `4c..4c + 4` (`c < 16`): `QH + 32c + 4r`.
//! - its 16 signed scales, scale `s`: `SCALES + 8s + r`, so scale `s` of
//!   all eight rows is one 8-byte lane.
//! - `d` (f16): `D + 2r`.

#[cfg(any(miri, test))]
mod scalar;
#[cfg(not(miri))]
mod vector;

use crate::Dotprod;
use crate::group::{Q8Superblocks, ROWS};

#[cfg(miri)]
use scalar as body;
#[cfg(not(miri))]
use vector as body;

/// Bytes per unit: one 210-byte block of each of eight rows.
pub const UNIT: usize = 1680;
/// Offset of the rows' low-bit bytes in a unit.
pub const QL: usize = 0;
/// Offset of the rows' high-bit bytes in a unit.
pub const QH: usize = 1024;
/// Offset of the rows' scales in a unit.
pub const SCALES: usize = 1536;
/// Offset of the rows' `d` values in a unit.
pub const D: usize = 1664;
/// Bytes of one row side by side in a unit.
pub const LANE: usize = 4;

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d * d8 * isum` from `-0.0`, with exact integer sums, in
/// the order and roundings of the portable `Q6_K` kernels.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q6_k_group(
    _token: Dotprod,
    group: &[[u8; UNIT]],
    inputs: &[Q8Superblocks<'_>],
    out: &mut [[f32; ROWS]],
) {
    // SAFETY: a `Dotprod` exists only after `Dotprod::detect` found
    // dotprod on this CPU (NEON is baseline on aarch64), so the CPU running
    // this call has every feature the body enables.
    unsafe { body::group(group, inputs, out) }
}

#[cfg(test)]
mod tests;
