// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;

pub(crate) type MergeLookup = HashMap<(u32, u32), (usize, u32)>;

/// Scores are indexed by token id, so they must cover the whole vocabulary.
pub(crate) fn pad_scores(mut scores: Vec<f32>, vocab_len: usize) -> Vec<f32> {
    scores.resize(vocab_len, 0.0);
    scores
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
}

impl Tokenizer {
    /// Construct a tokenizer directly from values (useful for testing).
    ///
    /// `scores` must be finite. Non-finite scores are not validated here (use
    /// `from_gguf` for untrusted input, which rejects them); they give
    /// unspecified but non-panicking tokenization.
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
        Self {
            model,
            vocab: vocab_bytes,
            scores: scores_vec,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info: HashMap::new(),
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
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;
