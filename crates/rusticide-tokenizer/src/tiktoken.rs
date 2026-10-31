// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! tiktoken vocabularies: one `base64(token bytes) rank` per line,
//! ranks are ids, and special tokens come from the encoding.

use std::cmp::Reverse;

use crate::end_of_generation::end_of_generation_ids;
use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::pre_tokenizer::PreTokenizer;
use crate::tokenizer::{MergeLookup, PieceLookup, Tokenizer, byte_tokens_for, pad_scores};
use crate::trie::Trie;

/// Ranks above this are rejected before any allocation sized by them.
const MAX_RANK: u32 = 1 << 22;

/// A tiktoken encoding: its pre-tokenizer pattern and special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TiktokenEncoding {
    /// GPT-4 and GPT-3.5's `cl100k_base`.
    Cl100kBase,
    /// GPT-4o and later's `o200k_base`.
    O200kBase,
}

impl TiktokenEncoding {
    const fn pre_tokenizer(self) -> PreTokenizer {
        match self {
            Self::Cl100kBase => PreTokenizer::CL100K,
            Self::O200kBase => PreTokenizer::O200K,
        }
    }

    /// tiktoken-rs 0.12.1, `openai_public.rs`.
    const fn specials(self) -> &'static [(&'static str, u32)] {
        match self {
            Self::Cl100kBase => &[
                ("<|endoftext|>", 100_257),
                ("<|fim_prefix|>", 100_258),
                ("<|fim_middle|>", 100_259),
                ("<|fim_suffix|>", 100_260),
                ("<|endofprompt|>", 100_276),
            ],
            Self::O200kBase => &[("<|endoftext|>", 199_999), ("<|endofprompt|>", 200_018)],
        }
    }
}

const fn sextet(symbol: u8) -> Option<u32> {
    match symbol {
        b'A'..=b'Z' => Some((symbol - b'A') as u32),
        b'a'..=b'z' => Some((symbol - b'a') as u32 + 26),
        b'0'..=b'9' => Some((symbol - b'0') as u32 + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Standard padded base64; `None` on any malformed input.
pub(crate) fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let (quads, _) = bytes.as_chunks::<4>();
    for quad in quads {
        let padding = quad.iter().rev().take_while(|&&b| b == b'=').count();
        if padding > 2 {
            return None;
        }
        let mut word = 0u32;
        for &symbol in &quad[..4 - padding] {
            word = (word << 6) | sextet(symbol)?;
        }
        word <<= 6 * padding;
        out.extend_from_slice(&word.to_be_bytes()[1..4 - padding]);
    }
    Some(out)
}

fn parse_line(line: &str) -> Option<(Vec<u8>, u32)> {
    let (token, rank) = line.split_once(' ')?;
    let rank: u32 = rank.parse().ok()?;
    (rank <= MAX_RANK).then_some(())?;
    Some((decode_base64(token)?, rank))
}

/// Every split `T = A + B` of a token into two tokens merges at `T`'s rank:
/// tiktoken merges the pair whose concatenation ranks lowest, which this
/// table hands to the same engine the GPT-2 path uses.
fn rank_merges(ids: &PieceLookup) -> MergeLookup {
    let mut merges = MergeLookup::default();
    for (token, &token_id) in ids.iter().filter(|(token, _)| token.len() > 1) {
        for split in 1..token.len() {
            if let (Some(&left), Some(&right)) =
                (ids.get(&token[..split]), ids.get(&token[split..]))
            {
                merges.insert((left, right), (token_id as usize, token_id));
            }
        }
    }
    merges
}

impl Tokenizer {
    /// Builds a tokenizer from a `.tiktoken` vocabulary file's text.
    ///
    /// # Errors
    ///
    /// `InvalidTiktoken` for a malformed line or an absurd rank, and
    /// `MissingByteToken` when one of the 256 byte tokens is absent.
    pub fn from_tiktoken(ranks: &str, encoding: TiktokenEncoding) -> Result<Self> {
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut ids = PieceLookup::default();
        for (index, line) in ranks.lines().enumerate() {
            let (token, rank) =
                parse_line(line).ok_or(TokenizerError::InvalidTiktoken { line: index + 1 })?;
            let slot = rank as usize;
            if slot >= vocab.len() {
                vocab.resize(slot + 1, Vec::new());
            }
            ids.insert(token.clone(), rank);
            vocab[slot] = token;
        }
        let byte_tokens = byte_tokens_for(TokenizerModel::Tiktoken, &vocab);
        if let Some(byte) = (0..=255u8).find(|&b| byte_tokens[usize::from(b)].is_none()) {
            return Err(TokenizerError::MissingByteToken { byte });
        }
        let merge_info = rank_merges(&ids);
        let mut special_tokens: Vec<(String, u32)> = encoding
            .specials()
            .iter()
            .map(|&(text, id)| (text.to_owned(), id))
            .collect();
        special_tokens.sort_by_key(|(text, _)| Reverse(text.len()));
        for &(text, id) in encoding.specials() {
            let slot = id as usize;
            if slot >= vocab.len() {
                vocab.resize(slot + 1, Vec::new());
            }
            vocab[slot] = text.as_bytes().to_vec();
        }
        let eos_id = encoding.specials()[0].1;
        Ok(Self {
            model: TokenizerModel::Tiktoken,
            scores: pad_scores(Vec::new(), vocab.len()),
            pieces: Trie::from_vocab(&[]),
            end_of_generation: end_of_generation_ids(&vocab, eos_id, None),
            vocab,
            bos_id: eos_id,
            eos_id,
            add_bos: false,
            add_space_prefix: false,
            merge_info,
            byte_tokens,
            pre: Some(encoding.pre_tokenizer()),
            special_tokens,
            whole_pieces: Some(ids),
        })
    }
}

#[cfg(test)]
#[path = "tiktoken_tests.rs"]
mod tests;
