// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

const NONE: u32 = u32::MAX;

/// One arena slot: 24 bytes, no per-node allocation. Children form a
/// singly linked sibling list starting at `first_child`.
#[derive(Debug)]
struct Node {
    byte: u8,
    first_child: u32,
    next_sibling: u32,
    token_id: Option<u32>,
}

impl Node {
    const fn new(byte: u8) -> Self {
        Self {
            byte,
            first_child: NONE,
            next_sibling: NONE,
            token_id: None,
        }
    }
}

/// Byte trie in a single arena, built once at construction. Memory is
/// proportional to the vocabulary's total bytes; a prefix walk descends one
/// node per input byte and stops at the first missing child.
#[derive(Debug)]
pub(crate) struct Trie {
    nodes: Vec<Node>,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![Node::new(0)],
        }
    }

    pub(crate) fn from_vocab(vocab: &[Vec<u8>]) -> Self {
        let mut trie = Self::new();
        for (id, bytes) in vocab.iter().enumerate() {
            trie.insert(bytes, u32::try_from(id).unwrap_or(u32::MAX));
        }
        trie
    }

    fn child(&self, node: usize, byte: u8, visits: &mut usize) -> Option<usize> {
        let mut cursor = self.nodes[node].first_child;
        while cursor != NONE {
            *visits += 1;
            let candidate = &self.nodes[cursor as usize];
            if candidate.byte == byte {
                return Some(cursor as usize);
            }
            cursor = candidate.next_sibling;
        }
        None
    }

    pub(crate) fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut idx = 0usize;
        for &b in bytes {
            if let Some(next) = self.child(idx, b, &mut 0) {
                idx = next;
            } else {
                let new_idx = u32::try_from(self.nodes.len()).unwrap_or(NONE);
                let mut node = Node::new(b);
                node.next_sibling = self.nodes[idx].first_child;
                self.nodes.push(node);
                self.nodes[idx].first_child = new_idx;
                idx = new_idx as usize;
            }
        }
        if idx != 0 {
            self.nodes[idx].token_id = Some(token_id);
        }
    }

    /// Returns all (`token_id`, `byte_length`, score) that match at `start`.
    pub(crate) fn prefixes(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
    ) -> Vec<(u32, usize, f32)> {
        self.walk(bytes, start, scores, &mut 0)
    }

    /// [`Self::prefixes`], counting every node examined into `visits`.
    pub(crate) fn walk(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
        visits: &mut usize,
    ) -> Vec<(u32, usize, f32)> {
        let mut result = Vec::new();
        let mut idx = 0usize;
        for (len, &b) in bytes[start..].iter().enumerate() {
            let Some(next) = self.child(idx, b, visits) else {
                break;
            };
            idx = next;
            if let Some(tid) = self.nodes[idx].token_id {
                result.push((tid, len + 1, scores[tid as usize]));
            }
        }
        result
    }
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "trie_walk_tests.rs"]
mod walk_tests;
