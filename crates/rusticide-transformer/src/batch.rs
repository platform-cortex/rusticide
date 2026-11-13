// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-token f32 work over a batch, forked on the pool once it is large
//! enough to repay the fork-join.

use rayon::prelude::*;

/// The smallest batch whose per-token work runs on the pool; a single
/// decode token always stays on the calling thread.
pub const PARALLEL_MIN_TOKENS: usize = 2;

/// Calls `op(token, row)` for each `width`-wide row of `rows`.
pub fn for_each_row(
    rows: &mut [f32],
    width: usize,
    tokens: usize,
    op: impl Fn(usize, &mut [f32]) + Sync,
) {
    if tokens >= PARALLEL_MIN_TOKENS {
        rows.par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(token, row)| op(token, row));
    } else {
        rows.chunks_exact_mut(width)
            .enumerate()
            .for_each(|(token, row)| op(token, row));
    }
}

/// Calls `op(row, matching)` for each `width`-wide row of `rows` and the
/// row of `other` at the same index.
pub fn zip_rows(
    rows: &mut [f32],
    other: &[f32],
    width: usize,
    tokens: usize,
    op: impl Fn(&mut [f32], &[f32]) + Sync,
) {
    if tokens >= PARALLEL_MIN_TOKENS {
        rows.par_chunks_exact_mut(width)
            .zip(other.par_chunks_exact(width))
            .for_each(|(row, matching)| op(row, matching));
    } else {
        rows.chunks_exact_mut(width)
            .zip(other.chunks_exact(width))
            .for_each(|(row, matching)| op(row, matching));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{for_each_row, zip_rows};

    #[test]
    fn for_each_row_visits_every_row_once_with_its_index() {
        for tokens in [1, 2, 9] {
            let width = 3;
            let mut rows = vec![0.0; width * tokens];
            let seen = Mutex::new(Vec::new());
            for_each_row(&mut rows, width, tokens, |token, row| {
                seen.lock().unwrap().push(token);
                row.fill(f32::from(u8::try_from(token).unwrap()));
            });
            let mut seen = seen.into_inner().unwrap();
            seen.sort_unstable();
            assert_eq!(seen, (0..tokens).collect::<Vec<_>>());
            for (token, row) in rows.chunks_exact(width).enumerate() {
                assert_eq!(row, vec![f32::from(u8::try_from(token).unwrap()); width]);
            }
        }
    }

    #[test]
    fn zip_rows_pairs_matching_rows() {
        for tokens in [1, 2, 9] {
            let width = 4;
            let other: Vec<f32> = (0..width * tokens)
                .map(|i| f32::from(u16::try_from(i).unwrap()))
                .collect();
            let mut rows = vec![0.0; width * tokens];
            zip_rows(&mut rows, &other, width, tokens, |row, matching| {
                row.copy_from_slice(matching);
                row[0] += 100.0;
            });
            for (row, matching) in rows.chunks_exact(width).zip(other.chunks_exact(width)) {
                assert_eq!(row[0], matching[0] + 100.0);
                assert_eq!(row[1..], matching[1..]);
            }
        }
    }
}
