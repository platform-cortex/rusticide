// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::MAX_TOKEN_BYTES;
use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{build, error_of, llama_fixture, tokenizer_meta};
use crate::tokenizer::Tokenizer;

#[test]
fn from_gguf_rejects_a_token_over_the_length_cap() {
    let long = "a".repeat(MAX_TOKEN_BYTES + 1);
    let entries = tokenizer_meta("llama", &["<s>", "ok", &long]);
    assert!(matches!(
        error_of(&entries),
        TokenizerError::TokenTooLong {
            index: 2,
            len,
            max: MAX_TOKEN_BYTES,
        } if len == MAX_TOKEN_BYTES + 1
    ));
}

#[test]
fn from_gguf_accepts_a_token_at_the_length_cap() {
    let at_cap = "a".repeat(MAX_TOKEN_BYTES);
    let tok = build(&tokenizer_meta("llama", &["<s>", &at_cap]));
    assert_eq!(tok.vocab_size(), 2);
}

#[test]
fn the_piece_lookup_is_built_at_construction() {
    let tok = Tokenizer::from_gguf(&llama_fixture()).unwrap();
    assert_eq!(tok.pieces.longest(), "hello".len().max(" world".len()));
    let direct = Tokenizer::new(TokenizerModel::LLaMA, vec!["ab", "abc"], None, 0, 1, false);
    assert_eq!(direct.pieces.longest(), 3);
}
