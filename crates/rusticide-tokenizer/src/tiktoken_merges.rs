// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The merge list a tiktoken vocabulary was trained with, recovered from its
//! ranks the way tiktoken-to-merges conversions do it.

use std::ops::Range;

use crate::tokenizer::{MergeLookup, PieceLookup};

/// Runs BPE over `token`'s own bytes with only the merges ranked below it;
/// training made it from the two parts left at the end. tiktoken notes that
/// merging by the rank of a concatenation "is equivalent" to this list as
/// long as the vocabulary was trained the usual way, and the list is a
/// fraction of every-split table: far fewer cache misses per lookup.
pub(crate) fn canonical_merges(ids: &PieceLookup) -> MergeLookup {
    let mut merges = MergeLookup::default();
    for (token, &rank) in ids.iter().filter(|(token, _)| token.len() > 1) {
        if let [left, right] = parts_below(token, rank, ids).as_slice() {
            if let (Some(&left), Some(&right)) = (
                ids.get(&token[left.clone()]),
                ids.get(&token[right.clone()]),
            ) {
                merges.insert((left, right), (rank, rank));
            }
        }
    }
    merges
}

/// The parts of `token` after merging, lowest rank first and leftmost on
/// ties, every adjacent pair whose concatenation ranks below `limit`.
fn parts_below(token: &[u8], limit: u32, ids: &PieceLookup) -> Vec<Range<usize>> {
    let mut parts: Vec<Range<usize>> = (0..token.len()).map(|at| at..at + 1).collect();
    loop {
        let best = parts
            .windows(2)
            .enumerate()
            .filter_map(|(left, pair)| {
                let rank = *ids.get(&token[pair[0].start..pair[1].end])?;
                (rank < limit).then_some((rank, left))
            })
            .min();
        let Some((_, left)) = best else {
            return parts;
        };
        parts[left].end = parts[left + 1].end;
        parts.remove(left + 1);
    }
}

#[cfg(test)]
#[path = "tiktoken_merges_tests.rs"]
mod tests;
