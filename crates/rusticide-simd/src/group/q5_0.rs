// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q5_0` unit every group kernel reads.
//!
//! A unit is the `k`-th 22-byte block of eight rows, 176 bytes,
//! interleaved so that 32 bytes hold the same four nibble bytes of all
//! eight rows. A 16-byte NEON load holds them for four rows, a 32-byte AVX2
//! load for all eight, and either accumulates a lane per row. Row `r`'s
//! block bytes sit at:
//!
//! - `d` (f16): `D + 2r`.
//! - its four fifth-bit bytes (bit `e` of the little-endian word is
//!   weight `e`'s): `QH + 4r`, so the eight rows' words are 32 bytes.
//! - nibble bytes `4c..4c + 4` (`c < 4`): `QS + 32c + 4r`; byte `i`
//!   holds weight `i` in its low nibble and weight `i + 16` in its high.

#[cfg(test)]
pub(crate) mod samples;
#[cfg(any(miri, test))]
pub(crate) mod scalar;

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
