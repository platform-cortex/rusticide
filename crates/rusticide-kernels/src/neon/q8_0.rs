// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q8_0` group layout: each unit interleaves the `k`-th 34-byte
//! block of eight rows in the byte order `rusticide_simd::neon::q8_0`
//! documents.

use rusticide_simd::group::ROWS;
use rusticide_simd::neon::q8_0::{D, LANE, QS, UNIT};

use crate::group_layout;

/// Bytes per `Q8_0` block.
pub(super) const BLOCK: usize = 34;

/// Where row `row`'s block bytes go in a unit: `(block offset, unit
/// offset, length)` for each run that stays contiguous.
fn runs(row: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let quants =
        (0..8).map(move |lane| (2 + LANE * lane, QS + ROWS * LANE * lane + LANE * row, LANE));
    core::iter::once((0, D + 2 * row, 2)).chain(quants)
}

pub(super) fn prepare(data: &mut [u8], row_bytes: usize) -> bool {
    group_layout::prepare::<BLOCK, UNIT, _>(data, row_bytes, runs)
}

pub(super) fn read_row(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    group_layout::read_row::<BLOCK, UNIT, _>(data, row_bytes, index, out, runs)
}

#[cfg(test)]
#[path = "q8_0_tests.rs"]
mod tests;
