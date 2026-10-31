// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::byte_unicode::byte_to_char;
use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{error_of, gpt2_tokenizer, gpt2_tokens, tokenizer_meta};
use crate::tokenizer::Tokenizer;

#[test]
fn from_gguf_rejects_a_vocab_missing_a_byte_token() {
    let mut owned = gpt2_tokens(&[]);
    owned.remove(0);
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    assert!(matches!(
        error_of(&tokenizer_meta("gpt2", &refs)),
        TokenizerError::MissingByteToken { byte: 0x00 }
    ));
}

#[test]
fn missing_byte_token_names_the_byte_in_hex() {
    let message = TokenizerError::MissingByteToken { byte: 0x0A }.to_string();
    assert_eq!(message, "gpt2 vocabulary has no token for byte 0x0A");
}

#[test]
fn the_byte_table_holds_the_id_of_each_byte_character() {
    let tok = gpt2_tokenizer(&["x"], &[]);
    assert_eq!(tok.byte_tokens[usize::from(b' ')], Some(32));
    assert!(tok.byte_tokens.iter().all(Option::is_some));
}

#[test]
fn a_repeated_byte_spelling_keeps_the_lowest_id() {
    let mut owned = gpt2_tokens(&[]);
    owned.push(byte_to_char(b'a').to_string());
    let vocab = owned.iter().map(String::as_str).collect();
    let tok = Tokenizer::new(TokenizerModel::GPT2, vocab, None, 0, 1, false);
    assert_eq!(tok.byte_tokens[usize::from(b'a')], Some(97));
}

#[test]
fn llama_has_no_byte_table() {
    let owned = gpt2_tokens(&[]);
    let vocab = owned.iter().map(String::as_str).collect();
    let tok = Tokenizer::new(TokenizerModel::LLaMA, vocab, None, 0, 1, false);
    assert!(tok.byte_tokens.iter().all(Option::is_none));
}
