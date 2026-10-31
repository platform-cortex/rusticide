// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::ops::Range;

use crate::bpe_merge::{SHORT_WORD, merge_into};
use crate::error::{Result, TokenizerError};
use crate::pre_tokenizer::PreTokenizer;
use crate::tokenizer::Tokenizer;

impl Tokenizer {
    /// GPT-2 byte-level BPE encoding.
    pub(crate) fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        let pre_tokenizer = self.pre.unwrap_or(PreTokenizer::GPT2);
        let mut tokens = Vec::new();
        let mut seen: HashMap<&str, Range<usize>> = HashMap::new();
        for word in pre_tokenizer.split(text)? {
            if let Some(range) = seen.get(word) {
                tokens.extend_from_within(range.clone());
                continue;
            }
            if let Some(&id) = self
                .whole_pieces
                .as_ref()
                .and_then(|ids| ids.get(word.as_bytes()))
            {
                tokens.push(id);
                continue;
            }
            let start = tokens.len();
            self.bpe_encode_word(word, &mut tokens)?;
            seen.insert(word, start..tokens.len());
        }
        Ok(tokens)
    }

    /// Starts from the byte token of each byte, merges by rank, and appends
    /// the ids to `out`; short words never touch the heap.
    fn bpe_encode_word(&self, word: &str, out: &mut Vec<u32>) -> Result<()> {
        let byte_id = |byte: u8| {
            self.byte_tokens[usize::from(byte)].ok_or_else(|| {
                TokenizerError::EncodeError(format!("no token for byte 0x{byte:02X}"))
            })
        };
        if word.len() <= SHORT_WORD {
            let mut ids = [0; SHORT_WORD];
            for (slot, byte) in ids.iter_mut().zip(word.bytes()) {
                *slot = byte_id(byte)?;
            }
            merge_into(&mut ids[..word.len()], &self.merge_info, out);
        } else {
            let mut ids = word.bytes().map(byte_id).collect::<Result<Vec<u32>>>()?;
            merge_into(&mut ids, &self.merge_info, out);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "bpe_tests.rs"]
mod tests;
