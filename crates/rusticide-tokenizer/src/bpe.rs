// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bpe_merge::merge_ids;
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

    /// Starts from the byte token of each byte and merges by rank.
    fn bpe_encode_word(&self, word: &str) -> Result<Vec<u32>> {
        let ids = word
            .bytes()
            .map(|byte| {
                self.byte_tokens[usize::from(byte)].ok_or_else(|| {
                    TokenizerError::EncodeError(format!("no token for byte 0x{byte:02X}"))
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok(merge_ids(ids, &self.merge_info))
    }
}

#[cfg(test)]
#[path = "bpe_tests.rs"]
mod tests;
