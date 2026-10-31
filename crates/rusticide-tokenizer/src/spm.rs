// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::tokenizer::Tokenizer;

const SPACE_MARKER: &str = "\u{2581}";

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
                TokenizerModel::GPT2 => self.encode_gpt2(text)?,
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

    /// `LLaMA` / `SentencePiece` encoding as llama.cpp does it: merge pieces
    /// by score, then spell any symbol that is not a piece as byte pieces.
    fn encode_llama(&self, text: &str) -> Result<Vec<u32>> {
        let prefix = if self.add_space_prefix { " " } else { "" };
        let normalized = format!("{prefix}{text}").replace(' ', SPACE_MARKER);
        let byte_fallback = self.build_byte_fallback();
        let mut tokens = Vec::new();
        for piece in self.merge_symbols(&normalized) {
            if let Some(token) = self.piece_id(piece) {
                tokens.push(token);
                continue;
            }
            for &byte in piece {
                let token = byte_fallback[usize::from(byte)].ok_or_else(|| {
                    TokenizerError::EncodeError(format!("no piece for byte 0x{byte:02X}"))
                })?;
                tokens.push(token);
            }
        }
        Ok(tokens)
    }

    fn build_byte_fallback(&self) -> [Option<u32>; 256] {
        let mut fallback = [None; 256];
        for (id, bytes) in self.vocab.iter().enumerate() {
            if let Some(byte) = byte_piece(bytes) {
                fallback[usize::from(byte)] = Some(u32::try_from(id).unwrap_or(u32::MAX));
            }
        }
        fallback
    }
}

/// The byte a `SentencePiece` byte-fallback piece (`<0xHH>`) stands for.
pub(crate) const fn byte_piece(piece: &[u8]) -> Option<u8> {
    if let [b'<', b'0', b'x', high, low, b'>'] = *piece {
        if let (Some(h), Some(l)) = (hex_nibble(high), hex_nibble(low)) {
            return Some((h << 4) | l);
        }
    }
    None
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

#[cfg(test)]
#[path = "spm_byte_tests.rs"]
mod byte_tests;
