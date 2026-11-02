// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q4_K` weights against `Q8_K` activations, eight rows at a time.
//!
//! The kernel reads units: the `k`-th 144-byte block of eight rows, 1152
//! bytes, interleaved so that one 16-byte load holds the same four quant
//! bytes of four rows and one `vdotq_laneq_s32` accumulates a lane per row.
//! Row `r`'s block bytes sit at:
//!
//! - `d` (f16): `D + 2r`; `dmin` (f16): `DMIN + 2r`.
//! - byte `i` of its 12 packed 6-bit scales and mins: `SCALES + 8i + r`, so
//!   byte `i` of all eight rows is one 8-byte lane.
//! - quant bytes `4c..4c + 4` (`c < 32`): `QS + 32c + 4r`.

#[cfg(any(miri, test))]
mod scalar;
#[cfg(not(miri))]
mod vector;

#[cfg(miri)]
use scalar as body;
#[cfg(not(miri))]
use vector as body;

/// Rows per unit.
pub const ROWS: usize = 8;
/// Bytes per unit: one 144-byte block of each of [`ROWS`] rows.
pub const UNIT: usize = 1152;
/// Offset of the rows' `d` values in a unit.
pub const D: usize = 0;
/// Offset of the rows' `dmin` values in a unit.
pub const DMIN: usize = 16;
/// Offset of the rows' packed scales in a unit.
pub const SCALES: usize = 32;
/// Offset of the rows' quant bytes in a unit.
pub const QS: usize = 128;
/// Quant bytes of one row side by side in a unit.
pub const LANE: usize = 4;
/// Inputs the kernel keeps partial sums for at once.
pub const TOKENS: usize = 16;

/// One token's `Q8_K` activation as whole 256-element super-blocks: each
/// super-block's scale (stored once per 32 elements, all equal), its
/// quants, and the sums of its eight 32-element runs.
#[derive(Debug, Default, Clone, Copy)]
pub struct Q8Superblocks<'a> {
    scales: &'a [[f32; 8]],
    quants: &'a [[i8; 256]],
    sums: &'a [[i32; 8]],
}

impl<'a> Q8Superblocks<'a> {
    /// A view of the super-blocks all three slices cover.
    #[must_use]
    pub fn new(scales: &'a [[f32; 8]], quants: &'a [[i8; 256]], sums: &'a [[i32; 8]]) -> Self {
        let len = scales.len().min(quants.len()).min(sums.len());
        Self {
            scales: scales.get(..len).unwrap_or_default(),
            quants: quants.get(..len).unwrap_or_default(),
            sums: sums.get(..len).unwrap_or_default(),
        }
    }
}

/// The products of the eight rows in `group` (one unit per block) with
/// each input, the rows side by side in the input's slot of `out`.
///
/// The shortest of `group` and an input bounds the blocks summed. Each
/// value is `sum_k d8 * (d * isum - dmin * msum)` from `-0.0`, with exact
/// integer sums, in the order and roundings of the portable `Q4_K` kernels.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn q4_k_group(group: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    // SAFETY: this module compiles only under `cfg(all(target_arch =
    // "aarch64", target_feature = "dotprod"))`, so the build target
    // guarantees that every CPU running this binary has NEON and dotprod.
    unsafe { body::group(group, inputs, out) }
}

#[cfg(test)]
mod tests;
