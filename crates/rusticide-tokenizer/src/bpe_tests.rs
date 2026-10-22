// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::model::TokenizerModel;
use crate::test_support::{char_byte_tokens, meta, text_array, tokenizer_meta, with};
use crate::tokenizer::Tokenizer;

fn gpt2_byte_tokenizer(scores_len: usize) -> Tokenizer {
    let owned = char_byte_tokens();
    let vocab = owned.iter().map(String::as_str).collect();
    Tokenizer::new(
        TokenizerModel::GPT2,
        vocab,
        Some(vec![0.0; scores_len]),
        0,
        1,
        false,
    )
}

#[test]
fn test_encode_gpt2_unaffected_by_spm_normalization() {
    let tok = gpt2_byte_tokenizer(256);
    assert_eq!(tok.encode("a b").unwrap(), &[97, 98]);
}

#[test]
fn test_bpe_encode_simple_word() {
    let tok = gpt2_byte_tokenizer(257);
    assert_eq!(tok.encode("a").unwrap(), &[97]);
}

#[test]
fn test_bpe_with_merges() {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|b| String::from_utf8_lossy(&[b]).to_string())
        .collect();
    tokens.push("ab".to_string());
    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let entries = tokenizer_meta("gpt2", &refs);
    let entries = with(entries, "tokenizer.ggml.merges", text_array(&["a b"]));
    let entries = with(
        entries,
        "tokenizer.ggml.add_bos_token",
        GGUFValue::Bool(false),
    );
    let tok = Tokenizer::from_gguf(&meta(&entries)).unwrap();
    assert_eq!(tok.encode("ab").unwrap(), &[256]);
}

fn merge_tokenizer(tokens: &[&str], merges: &[&str]) -> Tokenizer {
    let entries = with(
        tokenizer_meta("gpt2", tokens),
        "tokenizer.ggml.merges",
        text_array(merges),
    );
    Tokenizer::from_gguf(&meta(&entries)).unwrap()
}

fn byte_tokens_plus(extra: &[&str]) -> Vec<String> {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|b| String::from_utf8_lossy(&[b]).to_string())
        .collect();
    tokens.extend(extra.iter().map(|s| (*s).to_string()));
    tokens
}

#[test]
fn bpe_applies_the_lowest_rank_merge_first() {
    let owned = byte_tokens_plus(&["bc", "ab", "abc"]);
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let tok = merge_tokenizer(&refs, &["b c", "a b", "a bc"]);
    assert_eq!(tok.encode("abc").unwrap(), &[258]);
    let tok = merge_tokenizer(&refs, &["a b", "b c"]);
    assert_eq!(tok.encode("abc").unwrap(), &[257, 99]);
}

#[test]
fn bpe_breaks_rank_ties_at_the_leftmost_pair() {
    let owned = byte_tokens_plus(&["aa"]);
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let tok = merge_tokenizer(&refs, &["a a"]);
    assert_eq!(tok.encode("aaa").unwrap(), &[256, 97]);
}

#[test]
fn bpe_encodes_each_whitespace_separated_word_alone() {
    let owned = byte_tokens_plus(&["ab"]);
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let tok = merge_tokenizer(&refs, &["a b"]);
    assert_eq!(tok.encode("ab  a\tb\nab").unwrap(), &[256, 97, 98, 256]);
}

#[test]
fn bpe_of_only_whitespace_is_empty() {
    let tok = gpt2_byte_tokenizer(256);
    assert_eq!(tok.encode(" \n\t").unwrap(), Vec::<u32>::new());
}
