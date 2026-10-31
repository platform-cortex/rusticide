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

/// At or below this many symbols a linear scan over stack arrays beats the
/// heap; tiktoken draws the same line at 100 bytes.
pub(crate) const SHORT_WORD: usize = 64;

/// Merges `ids` (lowest rank first, leftmost on ties) and appends the
/// result to `out`.
pub(crate) fn merge_into(ids: &mut [u32], merges: &MergeLookup, out: &mut Vec<u32>) {
    if ids.len() <= SHORT_WORD {
        let len = merge_short(ids, merges);
        out.extend_from_slice(&ids[..len]);
    } else {
        out.extend(merge_long(ids.to_vec(), merges));
    }
}

/// O(n^2) for small n, allocation-free: each pair's rank is cached, and a
/// merge looks up only the two pairs around the merged symbol. Returns how
/// many symbols remain at the front of `ids`.
fn merge_short(ids: &mut [u32], merges: &MergeLookup) -> usize {
    let rank_of = |ids: &[u32], left: usize| {
        merges
            .get(&(ids[left], ids[left + 1]))
            .copied()
            .unwrap_or((NO_RANK, 0))
    };
    let mut len = ids.len();
    let mut ranks = [(NO_RANK, 0); SHORT_WORD];
    for (left, slot) in ranks.iter_mut().enumerate().take(len.saturating_sub(1)) {
        *slot = rank_of(ids, left);
    }
    while let Some((left, (rank, merged))) = ranks[..len.saturating_sub(1)]
        .iter()
        .copied()
        .enumerate()
        .min_by_key(|&(_, (rank, _))| rank)
    {
        if rank == NO_RANK {
            break;
        }
        ids[left] = merged;
        ids.copy_within(left + 2..len, left + 1);
        ranks.copy_within(left + 1..len - 1, left);
        len -= 1;
        if left > 0 {
            ranks[left - 1] = rank_of(&ids[..len], left - 1);
        }
        if left + 1 < len {
            ranks[left] = rank_of(&ids[..len], left);
        }
    }
    len
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
