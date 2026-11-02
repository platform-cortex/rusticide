// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q4_K` group layout: each unit interleaves the `k`-th 144-byte
//! block of eight rows in the byte order `rusticide_simd::neon::q4_k`
//! documents.

use rusticide_simd::neon::group::ROWS;
use rusticide_simd::neon::q4_k::{D, DMIN, LANE, QS, SCALES, UNIT};

use super::group_layout;

/// Bytes per `Q4_K` block.
pub(super) const BLOCK: usize = 144;

/// Where row `row`'s block bytes go in a unit: `(block offset, unit
/// offset, length)` for each run that stays contiguous.
fn runs(row: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let floats = [(0, D + 2 * row, 2), (2, DMIN + 2 * row, 2)];
    let scales = (0..12).map(move |byte| (4 + byte, SCALES + ROWS * byte + row, 1));
    let quants =
        (0..32).map(move |lane| (16 + LANE * lane, QS + ROWS * LANE * lane + LANE * row, LANE));
    floats.into_iter().chain(scales).chain(quants)
}

pub(super) fn prepare(data: &mut [u8], row_bytes: usize) -> bool {
    group_layout::prepare::<BLOCK, UNIT, _>(data, row_bytes, runs)
}

pub(super) fn read_row(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    group_layout::read_row::<BLOCK, UNIT, _>(data, row_bytes, index, out, runs)
}

#[cfg(test)]
#[path = "q4_k_tests.rs"]
mod tests;
