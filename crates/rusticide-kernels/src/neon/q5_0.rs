// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q5_0` group layout: each unit interleaves the `k`-th 22-byte
//! block of eight rows in the byte order `rusticide_simd::neon::q5_0`
//! documents.

use rusticide_simd::neon::group::ROWS;
use rusticide_simd::neon::q5_0::{D, LANE, QH, QS, UNIT};

use super::group_layout;

/// Bytes per `Q5_0` block.
pub(super) const BLOCK: usize = 22;

/// Where row `row`'s block bytes go in a unit: `(block offset, unit
/// offset, length)` for each run that stays contiguous.
fn runs(row: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let quants =
        (0..4).map(move |lane| (6 + LANE * lane, QS + ROWS * LANE * lane + LANE * row, LANE));
    [(0, D + 2 * row, 2), (2, QH + LANE * row, LANE)]
        .into_iter()
        .chain(quants)
}

pub(super) fn prepare(data: &mut [u8], row_bytes: usize) -> bool {
    group_layout::prepare::<BLOCK, UNIT, _>(data, row_bytes, runs)
}

pub(super) fn read_row(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    group_layout::read_row::<BLOCK, UNIT, _>(data, row_bytes, index, out, runs)
}

#[cfg(test)]
#[path = "q5_0_tests.rs"]
mod tests;
