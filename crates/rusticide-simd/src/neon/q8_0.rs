// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q8_0` weights against `Q8_0` activations, eight rows at a time.
//!
//! The kernel reads units: the `k`-th 34-byte block of eight rows, 272
//! bytes, interleaved so that one 16-byte load holds the same four weight
//! bytes of four rows and one `vdotq_laneq_s32` accumulates a lane per row.
//! Row `r`'s block bytes sit at:
//!
//! - `d` (f16): `D + 2r`.
//! - weight bytes `4c..4c + 4` (`c < 8`): `QS + 32c + 4r`.

#[cfg(any(miri, test))]
mod scalar;
#[cfg(not(miri))]
mod vector;

use crate::Dotprod;
use crate::group::{Q8Blocks, ROWS};

#[cfg(miri)]
use scalar as body;
#[cfg(not(miri))]
use vector as body;

/// Bytes per unit: one 34-byte block of each of eight rows.
pub const UNIT: usize = 272;
/// Offset of the rows' `d` values in a unit.
pub const D: usize = 0;
/// Offset of the rows' weight bytes in a unit.
pub const QS: usize = 16;
/// Weight bytes of one row side by side in a unit.
pub const LANE: usize = 4;

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d * d8 * isum` from `-0.0`, with exact integer sums, in
/// the order and roundings of the portable `Q8_0` kernel.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q8_0_group(
    _token: Dotprod,
    group: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
) {
    // SAFETY: a `Dotprod` exists only after `Dotprod::detect` found
    // dotprod on this CPU (NEON is baseline on aarch64), so the CPU running
    // this call has every feature the body enables.
    unsafe { body::group(group, inputs, out) }
}

#[cfg(test)]
mod tests;
