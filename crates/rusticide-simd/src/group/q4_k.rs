// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q4_K` unit every group kernel reads.
//!
//! A unit is the `k`-th 144-byte block of eight rows, 1152 bytes,
//! interleaved so that 32 bytes hold the same four quant bytes of all eight
//! rows. A 16-byte NEON load holds them for four rows, a 32-byte AVX2 load
//! for all eight, and either accumulates a lane per row. Row `r`'s block
//! bytes sit at:
//!
//! - `d` (f16): `D + 2r`; `dmin` (f16): `DMIN + 2r`.
//! - byte `i` of its 12 packed 6-bit scales and mins: `SCALES + 8i + r`, so
//!   byte `i` of all eight rows is one 8-byte lane.
//! - quant bytes `4c..4c + 4` (`c < 32`): `QS + 32c + 4r`.

#[cfg(test)]
pub(crate) mod samples;
#[cfg(any(miri, test))]
pub(crate) mod scalar;

/// Bytes per unit: one 144-byte block of each of eight rows.
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
