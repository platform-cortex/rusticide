// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BPE and `SentencePiece` tokenizers built from GGUF metadata.

mod bpe;
mod byte_unicode;
mod decode;
mod error;
mod from_gguf;
mod model;
mod spm;
mod spm_merge;
mod tokenizer;
mod trie;
mod value_ext;

#[cfg(test)]
mod crafted_tests;
#[cfg(test)]
mod test_support;

pub use error::{Result, TokenizerError};
pub use model::TokenizerModel;
pub use tokenizer::Tokenizer;
