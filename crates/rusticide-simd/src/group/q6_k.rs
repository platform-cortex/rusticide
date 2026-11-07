// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q6_K` unit every group kernel reads.
//!
//! A unit is the `k`-th 210-byte block of eight rows, 1680 bytes,
//! interleaved so that 32 bytes hold the same four low-bit (or high-bit)
//! bytes of all eight rows. A 16-byte NEON load holds them for four rows,
//! a 32-byte AVX2 load for all eight, and either accumulates a lane per
//! row. Row `r`'s block bytes sit at:
//!
//! - its 128 low-bit bytes `4c..4c + 4` (`c < 32`): `QL + 32c + 4r`.
//! - its 64 high-bit bytes `4c..4c + 4` (`c < 16`): `QH + 32c + 4r`.
//! - its 16 signed scales, scale `s`: `SCALES + 8s + r`, so scale `s` of
//!   all eight rows is one 8-byte lane.
//! - `d` (f16): `D + 2r`.

#[cfg(test)]
pub(crate) mod samples;
#[cfg(any(miri, test))]
pub(crate) mod scalar;

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
