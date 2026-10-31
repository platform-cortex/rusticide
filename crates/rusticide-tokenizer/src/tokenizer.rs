// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use crate::byte_unicode::byte_token_table;
use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::pre_tokenizer::PreTokenizer;
use crate::trie::Trie;

pub(crate) type MergeLookup = HashMap<(u32, u32), (usize, u32)>;

/// Scores are indexed by token id, so they must cover the whole vocabulary.
pub(crate) fn pad_scores(mut scores: Vec<f32>, vocab_len: usize) -> Vec<f32> {
    scores.resize(vocab_len, 0.0);
    scores
}

/// Only GPT-2 spells bytes as characters; `SentencePiece` has its own table.
pub(crate) fn byte_tokens_for(model: TokenizerModel, vocab: &[Vec<u8>]) -> [Option<u32>; 256] {
    match model {
        TokenizerModel::GPT2 => byte_token_table(vocab),
        TokenizerModel::LLaMA => [None; 256],
    }
}

pub(crate) fn token_id(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| TokenizerError::VocabOverflow)
}

#[derive(Debug)]
pub struct Tokenizer {
    pub(crate) model: TokenizerModel,
    pub(crate) vocab: Vec<Vec<u8>>,
    pub(crate) scores: Vec<f32>,
    pub(crate) bos_id: u32,
    pub(crate) eos_id: u32,
    pub(crate) add_bos: bool,
    pub(crate) add_space_prefix: bool,
    pub(crate) merge_info: MergeLookup,
    pub(crate) pieces: Trie,
    pub(crate) byte_tokens: [Option<u32>; 256],
    pub(crate) pre: Option<PreTokenizer>,
    pub(crate) end_of_generation: Vec<u32>,
}

impl Tokenizer {
    /// Construct a tokenizer directly from values (useful for testing).
    ///
    /// This is a trusted API: it validates nothing. Use `from_gguf` for
    /// untrusted input, which rejects what this accepts:
    ///
    /// - `scores` must be finite. Non-finite scores give unspecified but
    ///   non-panicking tokenization.
    /// - `from_gguf` rejects any token longer than 1024 bytes
    ///   ([`TokenizerError::TokenTooLong`]); `new` does not enforce that cap.
    ///   `SentencePiece` encoding costs O(input bytes x longest token), so a
    ///   long token here makes every such encode proportionally slower.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        model: TokenizerModel,
        vocab: Vec<&str>,
        scores: Option<Vec<f32>>,
        bos_id: u32,
        eos_id: u32,
        add_bos: bool,
    ) -> Self {
        let vocab_bytes: Vec<Vec<u8>> = vocab.iter().map(|s| s.as_bytes().to_vec()).collect();
        let scores_vec = pad_scores(scores.unwrap_or_default(), vocab_bytes.len());
        let add_space_prefix = model == TokenizerModel::LLaMA;
        let byte_tokens = byte_tokens_for(model, &vocab_bytes);
        let pre = (model == TokenizerModel::GPT2).then_some(PreTokenizer::GPT2);
        Self {
            pre,
            model,
            byte_tokens,
            pieces: Trie::from_vocab(&vocab_bytes),
            vocab: vocab_bytes,
            scores: scores_vec,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info: HashMap::new(),
            end_of_generation: vec![eos_id],
        }
    }

    pub(crate) fn vocab_index(&self, tid: u32) -> Result<usize> {
        let index = usize::try_from(tid).unwrap_or(usize::MAX);
        if index < self.vocab.len() {
            Ok(index)
        } else {
            Err(TokenizerError::TokenOutOfRange(tid, self.vocab.len()))
        }
    }

    #[must_use]
    pub const fn model_type(&self) -> TokenizerModel {
        self.model
    }

    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[must_use]
    pub const fn bos_id(&self) -> u32 {
        self.bos_id
    }

    #[must_use]
    pub const fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// True for the EOS token, the EOT token and every vocabulary token
    /// llama.cpp treats as end of generation.
    #[must_use]
    pub fn is_end_of_generation(&self, token: u32) -> bool {
        self.end_of_generation.binary_search(&token).is_ok()
    }
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "end_of_generation_tests.rs"]
mod end_of_generation_tests;
