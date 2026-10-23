// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::tokenizer::Tokenizer;

/// A run of input bytes; `len == 0` once merged into its left neighbour.
struct Symbol {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Highest score pops first; equal scores pop the leftmost pair first.
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl Bigram {
    /// False once either side changed since this pair was queued. A right
    /// side merged away needs no check: only this pair's left absorbs it, and
    /// that grows the left past `size`.
    const fn is_current(&self, symbols: &[Symbol]) -> bool {
        let (left, right) = (symbols[self.left].len, symbols[self.right].len);
        left != 0 && left + right == self.size
    }
}

fn char_symbols(text: &str) -> Vec<Symbol> {
    let count = text.chars().count();
    text.char_indices()
        .enumerate()
        .map(|(index, (start, character))| Symbol {
            start,
            len: character.len_utf8(),
            prev: index.checked_sub(1),
            next: (index + 1 < count).then_some(index + 1),
        })
        .collect()
}

const fn merge(symbols: &mut [Symbol], left: usize, right: usize) {
    symbols[left].len += symbols[right].len;
    symbols[right].len = 0;
    let next = symbols[right].next;
    symbols[left].next = next;
    if let Some(next) = next {
        symbols[next].prev = Some(left);
    }
}

fn spans<'t>(bytes: &'t [u8], symbols: &[Symbol]) -> Vec<&'t [u8]> {
    let mut result = Vec::new();
    let mut cursor = Some(0);
    while let Some(symbol) = cursor.and_then(|index| symbols.get(index)) {
        result.push(&bytes[symbol.start..symbol.start + symbol.len]);
        cursor = symbol.next;
    }
    result
}

impl Tokenizer {
    /// The token whose piece is exactly `piece`; the last duplicate wins.
    pub(crate) fn piece_id(&self, piece: &[u8]) -> Option<u32> {
        let matches = self.pieces.prefixes(piece, 0, &self.scores);
        matches
            .last()
            .filter(|&&(_, len, _)| len == piece.len())
            .map(|&(token, _, _)| token)
    }

    /// llama.cpp's `SentencePiece` merge: start from one symbol per character
    /// and repeatedly join the adjacent pair whose union is the
    /// highest-scoring vocabulary piece. Returns the final symbols in order.
    pub(crate) fn merge_symbols<'t>(&self, text: &'t str) -> Vec<&'t [u8]> {
        let bytes = text.as_bytes();
        let mut symbols = char_symbols(text);
        let mut queue = BinaryHeap::new();
        for right in 1..symbols.len() {
            self.queue_pair(&mut queue, bytes, &symbols, right - 1, right);
        }
        while let Some(bigram) = queue.pop() {
            if !bigram.is_current(&symbols) {
                continue;
            }
            merge(&mut symbols, bigram.left, bigram.right);
            if let Some(prev) = symbols[bigram.left].prev {
                self.queue_pair(&mut queue, bytes, &symbols, prev, bigram.left);
            }
            if let Some(next) = symbols[bigram.left].next {
                self.queue_pair(&mut queue, bytes, &symbols, bigram.left, next);
            }
        }
        spans(bytes, &symbols)
    }

    fn queue_pair(
        &self,
        queue: &mut BinaryHeap<Bigram>,
        bytes: &[u8],
        symbols: &[Symbol],
        left: usize,
        right: usize,
    ) {
        let start = symbols[left].start;
        let size = symbols[left].len + symbols[right].len;
        if let Some(token) = self.piece_id(&bytes[start..start + size]) {
            queue.push(Bigram {
                score: self.scores[token as usize],
                left,
                right,
                size,
            });
        }
    }
}

#[cfg(test)]
#[path = "spm_merge_tests.rs"]
mod tests;
