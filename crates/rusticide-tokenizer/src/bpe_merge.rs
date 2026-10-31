// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::tokenizer::MergeLookup;

/// A symbol with no mergeable pair, or one merged away.
const NO_RANK: u32 = u32::MAX;
/// No neighbour on that side.
const NONE: u32 = u32::MAX;
/// Long words merge in slices of at most this many symbols so every index
/// fits a `u32`; only a single word over 4 GiB is ever split.
const MAX_LONG_WORD: usize = u32::MAX as usize - 1;

/// `(rank, left)`: eight bytes, so a heap sift moves half as much memory.
type Entry = Reverse<(u32, u32)>;

fn to_u32(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(NONE)
}

/// A long word as a linked list of symbols, with the rank and result of the
/// pair each symbol starts. A heap entry `(rank, left)` is current while
/// `rank[left]` still holds it, as in tiktoken's `_byte_pair_merge_large`.
struct Chain<'m> {
    merges: &'m MergeLookup,
    ids: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    rank: Vec<u32>,
    merged: Vec<u32>,
}

impl<'m> Chain<'m> {
    fn new(ids: Vec<u32>, merges: &'m MergeLookup) -> Self {
        let len = ids.len();
        Self {
            merges,
            next: (1..=len).map(to_u32).collect(),
            prev: (0..len)
                .map(|index| index.checked_sub(1).map_or(NONE, to_u32))
                .collect(),
            rank: vec![NO_RANK; len],
            merged: vec![0; len],
            ids,
        }
    }

    /// Looks the pair starting at `left` up again; the heap entry to queue
    /// when it merges.
    fn refresh(&mut self, left: u32) -> Option<Entry> {
        let at = left as usize;
        let found = self
            .ids
            .get(self.next[at] as usize)
            .and_then(|&right_id| self.merges.get(&(self.ids[at], right_id)));
        if let Some(&(rank, merged)) = found {
            self.rank[at] = rank;
            self.merged[at] = merged;
            Some(Reverse((rank, left)))
        } else {
            self.rank[at] = NO_RANK;
            None
        }
    }

    fn merge(&mut self, left: u32, heap: &mut BinaryHeap<Entry>) {
        let at = left as usize;
        let right = self.next[at] as usize;
        self.ids[at] = self.merged[at];
        self.next[at] = self.next[right];
        if let Some(prev) = self.prev.get_mut(self.next[right] as usize) {
            *prev = left;
        }
        self.rank[right] = NO_RANK;
        heap.extend(self.refresh(left));
        let before = self.prev[at];
        if before != NONE {
            heap.extend(self.refresh(before));
        }
    }

    fn into_ids(self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.ids.len());
        let mut index = 0;
        while let Some(&id) = self.ids.get(index as usize) {
            out.push(id);
            index = self.next[index as usize];
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
        for slice in ids.chunks(MAX_LONG_WORD) {
            out.extend(merge_long(slice.to_vec(), merges));
        }
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

/// O(n log n): a heap of `(rank, left)` pairs, built in O(n) from the first
/// pass and skipping entries gone stale.
fn merge_long(ids: Vec<u32>, merges: &MergeLookup) -> Vec<u32> {
    let len = ids.len();
    let mut chain = Chain::new(ids, merges);
    let initial: Vec<_> = (0..len)
        .filter_map(|left| chain.refresh(to_u32(left)))
        .collect();
    let mut heap = BinaryHeap::from(initial);
    while let Some(Reverse((rank, left))) = heap.pop() {
        if chain.rank[left as usize] == rank {
            chain.merge(left, &mut heap);
        }
    }
    chain.into_ids()
}

#[cfg(test)]
#[path = "bpe_merge_tests.rs"]
mod tests;
