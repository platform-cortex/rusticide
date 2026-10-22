// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

struct TrieNode {
    children: Vec<(u8, usize)>,
    token_id: Option<u32>,
}

pub(crate) struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                children: Vec::new(),
                token_id: None,
            }],
        }
    }

    fn child(&self, node: usize, byte: u8) -> Option<usize> {
        self.nodes[node]
            .children
            .iter()
            .find(|(candidate, _)| *candidate == byte)
            .map(|&(_, index)| index)
    }

    pub(crate) fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut idx = 0usize;
        for &b in bytes {
            if let Some(next) = self.child(idx, b) {
                idx = next;
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode {
                    children: Vec::new(),
                    token_id: None,
                });
                self.nodes[idx].children.push((b, new_idx));
                idx = new_idx;
            }
        }
        self.nodes[idx].token_id = Some(token_id);
    }

    /// Returns all (`token_id`, `byte_length`, score) that match at `start`.
    pub(crate) fn prefixes(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
    ) -> Vec<(u32, usize, f32)> {
        let mut result = Vec::new();
        let mut idx = 0usize;
        for (len, &b) in bytes[start..].iter().enumerate() {
            let Some(next) = self.child(idx, b) else {
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
