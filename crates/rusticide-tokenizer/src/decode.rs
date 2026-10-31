// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::byte_unicode::char_to_byte;
use crate::error::Result;
use crate::model::TokenizerModel;
use crate::spm::byte_piece;
use crate::tokenizer::Tokenizer;

const SPM_MARKER: [u8; 3] = [0xE2, 0x96, 0x81];

impl Tokenizer {
    /// Decode token IDs back to text by joining their byte sequences.
    ///
    /// # Errors
    ///
    /// Returns `TokenOutOfRange` if any token ID exceeds the vocabulary.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self.decode_to_bytes(tokens)?;
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Decode token IDs but filter special tokens (BOS/EOS) and join with
    /// appropriate spacing.
    ///
    /// # Errors
    ///
    /// Returns `TokenOutOfRange` if any token ID exceeds the vocabulary.
    pub fn decode_detokenized(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &tid in tokens {
            if tid == self.bos_id || tid == self.eos_id {
                continue;
            }
            let piece = self.piece(tid)?;
            if !bytes.is_empty() && add_space_before(piece) {
                bytes.push(b' ');
            }
            bytes.extend_from_slice(piece);
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    fn decode_to_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &tid in tokens {
            let piece = self.piece(tid)?;
            match self.model {
                TokenizerModel::LLaMA => push_spm_text(&mut bytes, piece),
                TokenizerModel::GPT2 => push_gpt2_text(&mut bytes, piece),
            }
        }
        Ok(bytes)
    }

    fn piece(&self, tid: u32) -> Result<&[u8]> {
        Ok(&self.vocab[self.vocab_index(tid)?])
    }
}

fn add_space_before(piece: &[u8]) -> bool {
    piece.starts_with(&SPM_MARKER)
}

/// llama.cpp's piece-to-text for `SentencePiece`: a byte piece is its byte,
/// any other piece has each space marker turned back into a space.
fn push_spm_text(bytes: &mut Vec<u8>, piece: &[u8]) {
    if let Some(byte) = byte_piece(piece) {
        bytes.push(byte);
        return;
    }
    let mut rest = piece;
    while let Some(index) = find_marker(rest) {
        bytes.extend_from_slice(&rest[..index]);
        bytes.push(b' ');
        rest = &rest[index + SPM_MARKER.len()..];
    }
    bytes.extend_from_slice(rest);
}

/// Each character of a GPT-2 piece stands for one byte; a character outside
/// the table (a special token's text) is kept as its own UTF-8 bytes.
fn push_gpt2_text(bytes: &mut Vec<u8>, piece: &[u8]) {
    let text = String::from_utf8_lossy(piece);
    for c in text.chars() {
        match char_to_byte(c) {
            Some(byte) => bytes.push(byte),
            None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
}

fn find_marker(piece: &[u8]) -> Option<usize> {
    piece
        .windows(SPM_MARKER.len())
        .position(|window| window == SPM_MARKER)
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
