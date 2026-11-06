// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q4_K` weights against `Q8_K` activations, eight rows at a time.
//!
//! The kernel reads the units `crate::group::q4_k` lays out, the byte
//! order the NEON kernel reads too: one 32-byte load holds four quant
//! bytes of each of the eight rows, so `vpmaddubsw` against four
//! broadcast activation bytes, then `vpmaddwd` by the run scales, leaves
//! one `i32` lane per row and no horizontal sum.

#[cfg(not(miri))]
mod vector;

pub use crate::group::q4_k::UNIT;

use crate::Avx2;
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
    _token: Avx2,
    group: &[[u8; UNIT]],
    inputs: &[Q8Superblocks<'_>],
    out: &mut [[f32; ROWS]],
) {
    // SAFETY: an `Avx2` exists only after `Avx2::detect` found avx2 and
    // fma on this CPU, so the CPU running this call has every feature the
    // body enables.
    unsafe { body::group(group, inputs, out) }
}

#[cfg(test)]
mod tests;
