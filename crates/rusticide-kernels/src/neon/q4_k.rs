// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q4_K` layout the NEON kernel reads: groups of eight rows, each
//! group stored as one unit per block column, the unit holding the `k`-th
//! 144-byte block of all eight rows interleaved (the byte order is
//! documented in `rusticide_simd::neon::q4_k`). Repacking moves bytes and
//! nothing else, so `read_row` restores every original block exactly.

use rusticide_simd::neon::q4_k::{D, DMIN, LANE, QS, ROWS, SCALES, UNIT};

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

/// The bytes of a group of eight rows; `None` unless the rows are whole
/// blocks and the group fits in `usize`.
fn group_bytes(row_bytes: usize) -> Option<usize> {
    let whole = row_bytes > 0 && row_bytes.is_multiple_of(BLOCK);
    row_bytes.checked_mul(ROWS).filter(|_| whole)
}

pub(super) fn prepare(data: &mut [u8], row_bytes: usize) -> bool {
    let Some(group) = group_bytes(row_bytes).filter(|&group| data.len().is_multiple_of(group))
    else {
        return false;
    };
    let mut original = Vec::new();
    for rows in data.chunks_exact_mut(group) {
        original.clear();
        original.extend_from_slice(rows);
        let (units, _) = rows.as_chunks_mut::<UNIT>();
        for (row, blocks) in original.chunks_exact(row_bytes).enumerate() {
            let (blocks, _) = blocks.as_chunks::<BLOCK>();
            for (unit, block) in units.iter_mut().zip(blocks) {
                for (from, to, len) in runs(row) {
                    unit[to..to + len].copy_from_slice(&block[from..from + len]);
                }
            }
        }
    }
    true
}

pub(super) fn read_row(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    let rows = group_bytes(row_bytes)
        .filter(|_| out.len() == row_bytes)
        .and_then(|group| {
            let start = (index / ROWS).checked_mul(group)?;
            data.get(start..start.checked_add(group)?)
        });
    let Some(rows) = rows else {
        return false;
    };
    let (units, _) = rows.as_chunks::<UNIT>();
    let (blocks, _) = out.as_chunks_mut::<BLOCK>();
    for (unit, block) in units.iter().zip(blocks) {
        for (from, to, len) in runs(index % ROWS) {
            block[from..from + len].copy_from_slice(&unit[to..to + len]);
        }
    }
    true
}

#[cfg(test)]
#[path = "q4_k_tests.rs"]
mod tests;
