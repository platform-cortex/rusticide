// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::tokenizer::MergeLookup;

/// A symbol with no mergeable pair, or one merged away.
const NO_RANK: usize = usize::MAX;

/// A long word as a linked list of symbols, with the rank and result of the
/// pair each symbol starts. A heap entry `(rank, left)` is current while
/// `rank[left]` still holds it, as in tiktoken's `_byte_pair_merge_large`:
/// two words per entry instead of a copy of the whole pair.
struct Chain<'m> {
    merges: &'m MergeLookup,
    ids: Vec<u32>,
    next: Vec<usize>,
    prev: Vec<usize>,
    rank: Vec<usize>,
    merged: Vec<u32>,
}

impl<'m> Chain<'m> {
    fn new(ids: Vec<u32>, merges: &'m MergeLookup) -> Self {
        let len = ids.len();
        Self {
            merges,
            next: (1..=len).collect(),
            prev: (0..len).map(|index| index.wrapping_sub(1)).collect(),
            rank: vec![NO_RANK; len],
            merged: vec![0; len],
            ids,
        }
    }

    /// Looks the pair starting at `left` up again and queues it if it merges.
    fn refresh(&mut self, left: usize, heap: &mut BinaryHeap<Reverse<(usize, usize)>>) {
        let right = self.next[left];
        let found = self
            .ids
            .get(right)
            .and_then(|&right_id| self.merges.get(&(self.ids[left], right_id)));
        if let Some(&(rank, merged)) = found {
            self.rank[left] = rank;
            self.merged[left] = merged;
            heap.push(Reverse((rank, left)));
        } else {
            self.rank[left] = NO_RANK;
        }
    }

    fn merge(&mut self, left: usize, heap: &mut BinaryHeap<Reverse<(usize, usize)>>) {
        let right = self.next[left];
        self.ids[left] = self.merged[left];
        self.next[left] = self.next[right];
        if let Some(prev) = self.prev.get_mut(self.next[right]) {
            *prev = left;
        }
        self.rank[right] = NO_RANK;
        self.refresh(left, heap);
        if let Some(&before) = self.prev.get(left).filter(|&&before| before != usize::MAX) {
            self.refresh(before, heap);
        }
    }

    fn into_ids(self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.ids.len());
        let mut index = 0;
        while let Some(&id) = self.ids.get(index) {
            out.push(id);
            index = self.next[index];
        }
        out
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

/// O(n log n): a heap of `(rank, left)` pairs, skipping those gone stale.
fn merge_long(ids: Vec<u32>, merges: &MergeLookup) -> Vec<u32> {
    let len = ids.len();
    let mut chain = Chain::new(ids, merges);
    let mut heap = BinaryHeap::with_capacity(len);
    for left in 0..len {
        chain.refresh(left, &mut heap);
    }
    while let Some(Reverse((rank, left))) = heap.pop() {
        if chain.rank[left] == rank {
            chain.merge(left, &mut heap);
        }
    }
    chain.into_ids()
}

#[cfg(test)]
#[path = "bpe_merge_tests.rs"]
mod tests;
