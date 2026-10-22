// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers built from hostile metadata must fail with typed errors, never
//! panic.

use rusticide_io::GGUFValue;

use crate::error::TokenizerError;
use crate::test_support::{build, char_byte_tokens, float_array, text_array, tokenizer_meta, with};

#[test]
fn gpt2_ids_outside_a_small_vocabulary_decode_to_a_typed_error() {
    let tok = build(&tokenizer_meta("gpt2", &["a"]));
    let encoded = tok.encode("b").unwrap();
    assert_eq!(encoded, vec![98]);
    assert!(matches!(
        tok.decode(&encoded),
        Err(TokenizerError::TokenOutOfRange(98, 1))
    ));
}

#[test]
fn out_of_range_special_ids_are_never_used_as_indices() {
    let entries = with(
        tokenizer_meta("llama", &["\u{2581}a"]),
        "tokenizer.ggml.bos_token_id",
        GGUFValue::Uint32(900),
    );
    let entries = with(
        entries,
        "tokenizer.ggml.eos_token_id",
        GGUFValue::Uint32(901),
    );
    let tok = build(&entries);
    let encoded = tok.encode("a").unwrap();
    assert_eq!(encoded, vec![900, 0]);
    assert_eq!(tok.decode_detokenized(&encoded).unwrap(), "\u{2581}a");
    assert!(matches!(
        tok.decode(&encoded),
        Err(TokenizerError::TokenOutOfRange(900, 1))
    ));
}

#[test]
fn scores_shorter_than_the_vocabulary_cannot_panic_the_byte_fallback() {
    let entries = with(
        tokenizer_meta("llama", &["<0xE2>", "<0x96>", "<0x81>", "<0x61>"]),
        "tokenizer.ggml.scores",
        float_array(&[]),
    );
    let tok = build(&entries);
    assert_eq!(tok.encode("a").unwrap(), vec![1, 0, 1, 2, 3]);
}

#[test]
fn merges_naming_unknown_tokens_are_skipped_not_indexed() {
    let entries = with(
        tokenizer_meta("gpt2", &["a", "b"]),
        "tokenizer.ggml.merges",
        text_array(&["a b", "zzz zzz", " ", "a  b"]),
    );
    let tok = build(&entries);
    assert!(tok.merge_info.is_empty());
    assert_eq!(tok.encode("ab").unwrap(), vec![97, 98]);
}

#[test]
fn self_referential_merges_terminate() {
    let mut tokens = char_byte_tokens();
    tokens.push("aa".to_string());
    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let entries = with(
        tokenizer_meta("gpt2", &refs),
        "tokenizer.ggml.merges",
        text_array(&["a a"]),
    );
    let tok = build(&entries);
    assert!(tok.merge_info.contains_key(&(97, 97)));
    let word = "a".repeat(64);
    assert_eq!(tok.encode(&word).unwrap(), vec![256; 32]);
}
