// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Trie;

type Piece = (Vec<u8>, u32);

impl Trie {
    pub(crate) fn new() -> Self {
        Self::from_vocab(&[])
    }

    /// Rebuilds the whole trie with one more piece: O(size), tests only.
    pub(crate) fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut pieces = Vec::new();
        self.collect(0, &mut Vec::new(), &mut pieces);
        pieces.push((bytes.to_vec(), token_id));
        let borrowed = pieces.iter().map(|(b, id)| (b.as_slice(), *id)).collect();
        *self = Self::from_pieces(borrowed);
    }

    fn collect(&self, node: usize, path: &mut Vec<u8>, pieces: &mut Vec<Piece>) {
        if let Some(token_id) = self.tokens[node] {
            pieces.push((path.clone(), token_id));
        }
        for edge in self.edge_starts[node] as usize..self.edge_starts[node + 1] as usize {
            path.push(self.labels[edge]);
            self.collect(self.targets[edge] as usize, path, pieces);
            path.pop();
        }
    }

    /// [`Self::prefixes`], counting every byte comparison into `comparisons`.
    pub(crate) fn walk(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
        comparisons: &mut usize,
    ) -> Vec<(u32, usize, f32)> {
        self.walk_with(bytes, start, scores, &mut || *comparisons += 1)
    }
}
