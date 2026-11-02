// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q6_K` group layout: each unit interleaves the `k`-th 210-byte
//! block of eight rows in the byte order `rusticide_simd::neon::q6_k`
//! documents.

use rusticide_simd::group::ROWS;
use rusticide_simd::neon::q6_k::{D, LANE, QH, QL, SCALES, UNIT};

use super::group_layout;

/// Bytes per `Q6_K` block.
pub(super) const BLOCK: usize = 210;

/// Where row `row`'s block bytes go in a unit: `(block offset, unit
/// offset, length)` for each run that stays contiguous.
fn runs(row: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let lanes = |from: usize, base: usize, lanes: usize| {
        (0..lanes).map(move |lane| {
            (
                from + LANE * lane,
                base + ROWS * LANE * lane + LANE * row,
                LANE,
            )
        })
    };
    let scales = (0..16).map(move |scale| (192 + scale, SCALES + ROWS * scale + row, 1));
    lanes(0, QL, 32)
        .chain(lanes(128, QH, 16))
        .chain(scales)
        .chain([(208, D + 2 * row, 2)])
}

pub(super) fn prepare(data: &mut [u8], row_bytes: usize) -> bool {
    group_layout::prepare::<BLOCK, UNIT, _>(data, row_bytes, runs)
}

pub(super) fn read_row(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    group_layout::read_row::<BLOCK, UNIT, _>(data, row_bytes, index, out, runs)
}

#[cfg(test)]
#[path = "q6_k_tests.rs"]
mod tests;
