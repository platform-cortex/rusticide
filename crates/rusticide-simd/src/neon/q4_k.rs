// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q4_K` weights against `Q8_K` activations, eight rows at a time.
//!
//! The kernel reads the units `crate::group::q4_k` lays out; one
//! `vdotq_laneq_s32` accumulates four of its rows' quant bytes, a lane
//! per row.

#[cfg(not(miri))]
mod vector;

pub use crate::group::q4_k::UNIT;

use crate::Dotprod;
use crate::group::{Q8Superblocks, ROWS};

#[cfg(miri)]
use crate::group::q4_k::scalar as body;
#[cfg(not(miri))]
use vector as body;

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d8 * (d * isum - dmin * msum)` from `-0.0`, with exact
/// integer sums, in the order and roundings of the portable `Q4_K` kernels.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q4_k_group(
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
