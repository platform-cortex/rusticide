// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::successors;

use crate::tokenizer::MergeLookup;

/// A queued merge of the pair `left`, `right`. The derived order compares
/// `rank`, then `left`: under `Reverse` the lowest rank pops first and the
/// leftmost pair wins ties, as llama.cpp's `llm_bigram_bpe` orders them.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    rank: usize,
    left: usize,
    right: usize,
    pair: (u32, u32),
    merged: u32,
}

/// The ids of a word as a linked list; a merged-away symbol has no `next`.
struct Symbols {
    ids: Vec<u32>,
    prev: Vec<Option<usize>>,
    next: Vec<Option<usize>>,
}

impl Symbols {
    fn new(ids: Vec<u32>) -> Self {
        let len = ids.len();
        Self {
            ids,
            prev: (0..len).map(|index| index.checked_sub(1)).collect(),
            next: (0..len)
                .map(|index| (index + 1 < len).then_some(index + 1))
                .collect(),
        }
    }

    /// False once either symbol changed or left the list after queueing.
    fn is_current(&self, candidate: &Candidate) -> bool {
        self.next[candidate.left] == Some(candidate.right)
            && (self.ids[candidate.left], self.ids[candidate.right]) == candidate.pair
    }

    fn merge(&mut self, candidate: &Candidate) {
        self.ids[candidate.left] = candidate.merged;
        let after = self.next[candidate.right].take();
        self.next[candidate.left] = after;
        if let Some(after) = after {
            self.prev[after] = Some(candidate.left);
        }
    }

    fn into_ids(self) -> Vec<u32> {
        let first = (!self.ids.is_empty()).then_some(0);
        successors(first, |&index| self.next[index])
            .map(|index| self.ids[index])
            .collect()
    }
}

fn queue_pair(
    queue: &mut BinaryHeap<Reverse<Candidate>>,
    merges: &MergeLookup,
    symbols: &Symbols,
    left: usize,
    right: usize,
) {
    let pair = (symbols.ids[left], symbols.ids[right]);
    if let Some(&(rank, merged)) = merges.get(&pair) {
        queue.push(Reverse(Candidate {
            rank,
            left,
            right,
            pair,
            merged,
        }));
    }
}

/// At or below this many symbols a linear scan beats the heap's overhead;
/// almost every pre-tokenized word is this short.
const SHORT_WORD: usize = 16;

/// Repeatedly merges the lowest-rank adjacent pair, leftmost on ties.
pub(crate) fn merge_ids(ids: Vec<u32>, merges: &MergeLookup) -> Vec<u32> {
    if ids.len() <= SHORT_WORD {
        merge_short(ids, merges)
    } else {
        merge_long(ids, merges)
    }
}

/// O(n^2) for tiny n: each pair's rank is cached, and a merge looks up only
/// the two pairs around the merged symbol.
fn merge_short(mut ids: Vec<u32>, merges: &MergeLookup) -> Vec<u32> {
    let rank_of = |ids: &[u32], left: usize| merges.get(&(ids[left], ids[left + 1])).copied();
    let mut ranks: Vec<Option<(usize, u32)>> = (0..ids.len().saturating_sub(1))
        .map(|left| rank_of(&ids, left))
        .collect();
    while let Some((left, merged)) = ranks
        .iter()
        .enumerate()
        .filter_map(|(left, found)| found.map(|(rank, merged)| (rank, left, merged)))
        .min()
        .map(|(_, left, merged)| (left, merged))
    {
        ids[left] = merged;
        ids.remove(left + 1);
        ranks.remove(left);
        if left > 0 {
            ranks[left - 1] = rank_of(&ids, left - 1);
        }
        if left < ranks.len() {
            ranks[left] = rank_of(&ids, left);
        }
    }
    ids
}

/// O(n log n): a heap of candidate pairs, skipping those gone stale.
fn merge_long(ids: Vec<u32>, merges: &MergeLookup) -> Vec<u32> {
    let mut symbols = Symbols::new(ids);
    let mut queue = BinaryHeap::new();
    for right in 1..symbols.ids.len() {
        queue_pair(&mut queue, merges, &symbols, right - 1, right);
    }
    while let Some(Reverse(candidate)) = queue.pop() {
        if !symbols.is_current(&candidate) {
            continue;
        }
        symbols.merge(&candidate);
        let left = candidate.left;
        if let Some(prev) = symbols.prev[left] {
            queue_pair(&mut queue, merges, &symbols, prev, left);
        }
        if let Some(next) = symbols.next[left] {
            queue_pair(&mut queue, merges, &symbols, left, next);
        }
    }
    symbols.into_ids()
}

#[cfg(test)]
#[path = "bpe_merge_tests.rs"]
mod tests;
