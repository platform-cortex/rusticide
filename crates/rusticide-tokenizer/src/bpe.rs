// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{Result, TokenizerError};
use crate::pre_tokenizer::PreTokenizer;
use crate::tokenizer::Tokenizer;

impl Tokenizer {
    /// GPT-2 byte-level BPE encoding.
    pub(crate) fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        let pre_tokenizer = self.pre.unwrap_or(PreTokenizer::GPT2);
        let mut tokens = Vec::new();
        for word in pre_tokenizer.split(text)? {
            tokens.extend(self.bpe_encode_word(word)?);
        }
        Ok(tokens)
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

    /// Starts from the byte token of each byte and merges by rank.
    ///
    /// Each merge removes one id, so the loop is bounded by the word length;
    /// cost is quadratic in the word length.
    fn bpe_encode_word(&self, word: &str) -> Result<Vec<u32>> {
        let mut ids = word
            .bytes()
            .map(|byte| {
                self.byte_tokens[usize::from(byte)].ok_or_else(|| {
                    TokenizerError::EncodeError(format!("no token for byte 0x{byte:02X}"))
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        while let Some((position, merged)) = self.best_merge(&ids) {
            ids[position] = merged;
            ids.remove(position + 1);
        }
        Ok(ids)
    }
}

#[cfg(test)]
#[path = "bpe_tests.rs"]
mod tests;
