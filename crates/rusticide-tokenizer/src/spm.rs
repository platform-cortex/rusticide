// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::tokenizer::Tokenizer;

const SPACE_MARKER: &str = "\u{2581}";

struct Lattice {
    cost: Vec<f32>,
    back: Vec<(u32, usize)>,
}

impl Lattice {
    fn new(len: usize) -> Self {
        let mut cost = vec![f32::INFINITY; len + 1];
        cost[0] = 0.0;
        Self {
            cost,
            back: vec![(0, 0); len + 1],
        }
    }

    fn relax(&mut self, from: usize, span: usize, token: u32, score: f32) {
        let candidate = self.cost[from] - score;
        if candidate < self.cost[from + span] {
            self.cost[from + span] = candidate;
            self.back[from + span] = (token, span);
        }
    }

    fn backtrack(&self) -> Result<Vec<u32>> {
        let mut pos = self.cost.len() - 1;
        if !self.cost[pos].is_finite() {
            return Err(TokenizerError::EncodeError(
                "Viterbi could not find a valid tokenization path".to_string(),
            ));
        }
        let mut result = Vec::new();
        while pos > 0 {
            let (token, span) = self.back[pos];
            result.push(token);
            pos -= span;
        }
        result.reverse();
        Ok(result)
    }
}

impl Tokenizer {
    /// Encode text to token IDs.
    ///
    /// # Errors
    ///
    /// Returns `EncodeError` if the text cannot be encoded.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = if text.is_empty() {
            Vec::new()
        } else {
            match self.model {
                TokenizerModel::LLaMA => self.encode_llama(text)?,
                TokenizerModel::GPT2 => self.encode_gpt2(text),
            }
        };
        if self.add_bos {
            let mut result = vec![self.bos_id];
            result.extend(tokens);
            Ok(result)
        } else {
            Ok(tokens)
        }
    }

    /// `LLaMA` / `SentencePiece` unigram encoding via Viterbi DP.
    ///
    /// Cost is O(input bytes x longest vocabulary entry); the piece lookup is
    /// built once at construction.
    fn encode_llama(&self, text: &str) -> Result<Vec<u32>> {
        let prefix = if self.add_space_prefix { " " } else { "" };
        let normalized = format!("{prefix}{text}").replace(' ', SPACE_MARKER);
        let bytes = normalized.as_bytes();
        let byte_fallback = self.build_byte_fallback();
        let mut lattice = Lattice::new(bytes.len());
        for start in 0..bytes.len() {
            let matches = self.pieces.prefixes(bytes, start, &self.scores);
            if matches.is_empty() {
                if let Some(id) = byte_fallback[bytes[start] as usize] {
                    lattice.relax(start, 1, id, self.scores[id as usize]);
                }
            }
            for (token, span, score) in matches {
                lattice.relax(start, span, token, score);
            }
        }
        lattice.backtrack()
    }

    fn build_byte_fallback(&self) -> [Option<u32>; 256] {
        let mut fallback = [None; 256];
        for (id, bytes) in self.vocab.iter().enumerate() {
            if let [b'<', b'0', b'x', high, low, b'>'] = bytes[..] {
                if let (Some(h), Some(l)) = (hex_nibble(high), hex_nibble(low)) {
                    fallback[usize::from((h << 4) | l)] =
                        Some(u32::try_from(id).unwrap_or(u32::MAX));
                }
            }
        }
        fallback
    }
}

const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[path = "spm_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "spm_ranking_tests.rs"]
mod ranking_tests;
