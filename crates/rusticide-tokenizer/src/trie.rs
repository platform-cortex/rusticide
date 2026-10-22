// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Byte trie in compressed sparse row form, built once at construction.
///
/// Node `n` owns the edges `edge_starts[n]..edge_starts[n + 1]`, sorted by
/// byte, so a child lookup is a binary search: at most 9 comparisons at
/// fan-out 256. A node costs 12 bytes (`edge_starts` + `tokens`) and an edge
/// 5 bytes (`labels` + `targets`); every node but the root has one incoming
/// edge, so memory is about 17 bytes per distinct vocabulary prefix.
#[derive(Debug)]
pub(crate) struct Trie {
    edge_starts: Vec<u32>,
    labels: Vec<u8>,
    targets: Vec<u32>,
    tokens: Vec<Option<u32>>,
}

impl Trie {
    pub(crate) fn from_vocab(vocab: &[Vec<u8>]) -> Self {
        let pieces = vocab
            .iter()
            .enumerate()
            .map(|(index, bytes)| (bytes.as_slice(), u32::try_from(index).unwrap_or(u32::MAX)))
            .collect();
        Self::from_pieces(pieces)
    }

    /// Sorts the pieces (stably, so the last duplicate wins), then creates
    /// each node once by sharing the longest common prefix with the previous
    /// piece: O(sort + bytes).
    fn from_pieces(mut pieces: Vec<(&[u8], u32)>) -> Self {
        pieces.sort_by(|left, right| left.0.cmp(right.0));
        let mut builder = Builder::new();
        let mut previous: &[u8] = &[];
        for (bytes, token_id) in pieces {
            builder.push(previous, bytes, token_id);
            previous = bytes;
        }
        builder.finish()
    }

    fn child(&self, node: usize, byte: u8, probe: &mut impl FnMut()) -> Option<usize> {
        let first = self.edge_starts[node] as usize;
        let last = self.edge_starts[node + 1] as usize;
        find_label(&self.labels[first..last], byte, probe)
            .map(|offset| self.targets[first + offset] as usize)
    }

    /// Returns all (`token_id`, `byte_length`, score) that match at `start`.
    pub(crate) fn prefixes(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
    ) -> Vec<(u32, usize, f32)> {
        self.walk_with(bytes, start, scores, &mut || ())
    }

    fn walk_with(
        &self,
        bytes: &[u8],
        start: usize,
        scores: &[f32],
        probe: &mut impl FnMut(),
    ) -> Vec<(u32, usize, f32)> {
        let mut result = Vec::new();
        let mut node = 0usize;
        for (len, &byte) in bytes[start..].iter().enumerate() {
            let Some(next) = self.child(node, byte, probe) else {
                break;
            };
            node = next;
            if let Some(tid) = self.tokens[node] {
                result.push((tid, len + 1, scores[tid as usize]));
            }
        }
        result
    }
}

/// Binary search over sorted labels: `ceil(log2(len)) + 1` comparisons.
/// Hand-rolled because `binary_search_by` measured about 2x slower here.
fn find_label(labels: &[u8], byte: u8, probe: &mut impl FnMut()) -> Option<usize> {
    if labels.is_empty() {
        return None;
    }
    let mut low = 0;
    let mut size = labels.len();
    while size > 1 {
        let half = size / 2;
        probe();
        if labels[low + half] <= byte {
            low += half;
        }
        size -= half;
    }
    probe();
    (labels[low] == byte).then_some(low)
}

/// Nodes are created in depth-first order, so each parent's children arrive
/// in increasing byte order; `finish` only has to group them by parent.
struct Builder {
    parents: Vec<u32>,
    labels: Vec<u8>,
    tokens: Vec<Option<u32>>,
    path: Vec<u32>,
}

impl Builder {
    fn new() -> Self {
        Self {
            parents: vec![0],
            labels: vec![0],
            tokens: vec![None],
            path: vec![0],
        }
    }

    fn push(&mut self, previous: &[u8], bytes: &[u8], token_id: u32) {
        let shared = previous
            .iter()
            .zip(bytes)
            .take_while(|(left, right)| left == right)
            .count();
        self.path.truncate(shared + 1);
        for &byte in &bytes[shared..] {
            let node = u32::try_from(self.labels.len()).unwrap_or(u32::MAX);
            self.parents.push(self.path[self.path.len() - 1]);
            self.labels.push(byte);
            self.tokens.push(None);
            self.path.push(node);
        }
        self.tokens[self.path[self.path.len() - 1] as usize] = Some(token_id);
    }

    fn finish(self) -> Trie {
        let mut edge_starts = vec![0u32; self.labels.len() + 1];
        for &parent in &self.parents[1..] {
            edge_starts[parent as usize + 1] += 1;
        }
        for node in 1..edge_starts.len() {
            edge_starts[node] += edge_starts[node - 1];
        }
        let mut cursor = edge_starts.clone();
        let mut labels = vec![0u8; self.labels.len() - 1];
        let mut targets = vec![0u32; self.labels.len() - 1];
        for (node, &parent) in self.parents.iter().enumerate().skip(1) {
            let slot = cursor[parent as usize] as usize;
            cursor[parent as usize] += 1;
            labels[slot] = self.labels[node];
            targets[slot] = u32::try_from(node).unwrap_or(u32::MAX);
        }
        Trie {
            edge_starts,
            labels,
            targets,
            tokens: self.tokens,
        }
    }
}

#[cfg(test)]
#[path = "trie_test_support.rs"]
mod test_support;

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "trie_walk_tests.rs"]
mod walk_tests;
