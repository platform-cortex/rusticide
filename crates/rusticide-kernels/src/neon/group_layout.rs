// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The layout the NEON group kernels read: groups of eight rows, each
//! group stored as one `UNIT`-byte unit per block column, the unit holding
//! the `k`-th `BLOCK`-byte block of all eight rows. A format's `runs(row)`
//! gives `(block offset, unit offset, length)` for each contiguous run of
//! row `row`'s block bytes. Repacking moves bytes and nothing else, so
//! `read_row` restores every original block exactly.

use rusticide_simd::neon::group::ROWS;

/// The bytes of a group of eight rows; `None` unless the rows are whole
/// blocks and the group fits in `usize`.
fn group_bytes<const BLOCK: usize>(row_bytes: usize) -> Option<usize> {
    let whole = row_bytes > 0 && row_bytes.is_multiple_of(BLOCK);
    row_bytes.checked_mul(ROWS).filter(|_| whole)
}

pub(super) fn prepare<const BLOCK: usize, const UNIT: usize, I>(
    data: &mut [u8],
    row_bytes: usize,
    runs: impl Fn(usize) -> I,
) -> bool
where
    I: Iterator<Item = (usize, usize, usize)>,
{
    let group = group_bytes::<BLOCK>(row_bytes).filter(|&group| data.len().is_multiple_of(group));
    let Some(group) = group else {
        return false;
    };
    // Unit `k` overwrites bytes other rows have not been read from yet, so
    // each group is first copied aside.
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

pub(super) fn read_row<const BLOCK: usize, const UNIT: usize, I>(
    data: &[u8],
    row_bytes: usize,
    index: usize,
    out: &mut [u8],
    runs: impl Fn(usize) -> I,
) -> bool
where
    I: Iterator<Item = (usize, usize, usize)>,
{
    let rows = group_bytes::<BLOCK>(row_bytes)
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
