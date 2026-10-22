// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

/// Length-bounded piece lookup: memory is proportional to the vocabulary's
/// total bytes, and a probe never looks past the longest piece.
#[derive(Debug, Default)]
pub(crate) struct Trie {
    pieces: HashMap<Vec<u8>, u32>,
    longest: usize,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_vocab(vocab: &[Vec<u8>]) -> Self {
        let mut trie = Self::new();
        for (id, bytes) in vocab.iter().enumerate() {
            trie.insert(bytes, u32::try_from(id).unwrap_or(u32::MAX));
        }
        trie
    }

    pub(crate) fn insert(&mut self, bytes: &[u8], token_id: u32) {
        if bytes.is_empty() {
            return;
        }
        self.longest = self.longest.max(bytes.len());
        self.pieces.insert(bytes.to_vec(), token_id);
    }

    #[cfg(test)]
    pub(crate) const fn longest(&self) -> usize {
        self.longest
    }

    /// Returns all (`token_id`, `byte_length`, score) that match at `start`.
    pub(crate) fn prefixes(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
    ) -> Vec<(u32, usize, f32)> {
        let rest = &bytes[start..];
        (1..=rest.len().min(self.longest))
            .filter_map(|len| {
                let tid = *self.pieces.get(&rest[..len])?;
                Some((tid, len, scores[tid as usize]))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
