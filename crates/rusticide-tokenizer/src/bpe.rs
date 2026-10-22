// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::tokenizer::Tokenizer;

fn gpt2_pre_tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

impl Tokenizer {
    /// GPT-2 byte-level BPE encoding.
    pub(crate) fn encode_gpt2(&self, text: &str) -> Vec<u32> {
        gpt2_pre_tokenize(text)
            .flat_map(|word| self.bpe_encode_word(word))
            .collect()
    }

    /// Lowest-rank merge as (position, merged id); the leftmost wins ties.
    fn best_merge(&self, ids: &[u32]) -> Option<(usize, u32)> {
        ids.windows(2)
            .enumerate()
            .filter_map(|(position, pair)| {
                let &(rank, merged) = self.merge_info.get(&(pair[0], pair[1]))?;
                Some((rank, position, merged))
            })
            .min()
            .map(|(_, position, merged)| (position, merged))
    }

    /// Starts from one id per byte (the byte value) and merges by rank.
    ///
    /// Each merge removes one id, so the loop is bounded by the word length;
    /// cost is quadratic in the word length.
    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = word.bytes().map(u32::from).collect();
        while let Some((position, merged)) = self.best_merge(&ids) {
            ids[position] = merged;
            ids.remove(position + 1);
        }
        ids
    }
}

#[cfg(test)]
#[path = "bpe_tests.rs"]
mod tests;
