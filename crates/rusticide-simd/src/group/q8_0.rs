// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q8_0` unit every group kernel reads.
//!
//! A unit is the `k`-th 34-byte block of eight rows, 272 bytes,
//! interleaved so that 32 bytes hold the same four weight bytes of all
//! eight rows. A 16-byte NEON load holds them for four rows, a 32-byte AVX2
//! load for all eight, and either accumulates a lane per row. Row `r`'s
//! block bytes sit at:
//!
//! - `d` (f16): `D + 2r`.
//! - weight bytes `4c..4c + 4` (`c < 8`): `QS + 32c + 4r`.

#[cfg(test)]
pub(crate) mod samples;
#[cfg(any(miri, test))]
pub(crate) mod scalar;

/// Bytes per unit: one 34-byte block of each of eight rows.
pub const UNIT: usize = 272;
/// Offset of the rows' `d` values in a unit.
pub const D: usize = 0;
/// Offset of the rows' weight bytes in a unit.
pub const QS: usize = 16;
/// Weight bytes of one row side by side in a unit.
pub const LANE: usize = 4;
