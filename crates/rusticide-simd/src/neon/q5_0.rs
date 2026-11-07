// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q5_0` weights against `Q8_0` activations, eight rows at a time.
//!
//! The kernel reads the units `crate::group::q5_0` lays out; one
//! `vdotq_laneq_s32` accumulates four of its rows' nibble bytes, a lane
//! per row.

#[cfg(not(miri))]
mod vector;

pub use crate::group::q5_0::UNIT;

use crate::Dotprod;
use crate::group::{Q8Blocks, ROWS};

#[cfg(miri)]
use crate::group::q5_0::scalar as body;
#[cfg(not(miri))]
use vector as body;

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d * d8 * isum` from `-0.0`, with exact integer sums, in
/// the order and roundings of the portable `Q5_0` kernels.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q5_0_group(
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
mod tests {
    crate::test_support::group_body_tests!(
        super::q5_0_group,
        crate::group::q5_0::samples,
        0x0005_00a5,
        the_largest_sums_are_exact,
    );
}
