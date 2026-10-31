// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{byte_id, gpt2_tokenizer, gpt2_tokens};
use crate::tokenizer::Tokenizer;

#[test]
fn encode_maps_bytes_through_the_table() {
    let tok = gpt2_tokenizer(&[], &[]);
    assert_eq!(tok.encode("hi").unwrap(), &[byte_id(b'h'), byte_id(b'i')]);
}

#[test]
fn encode_uses_the_token_id_of_the_shifted_character() {
    let mut owned = gpt2_tokens(&[]);
    owned.swap(0x21, 0x41);
    let vocab = owned.iter().map(String::as_str).collect();
    let tok = Tokenizer::new(TokenizerModel::GPT2, vocab, None, 0, 1, false);
    assert_eq!(tok.encode("!A").unwrap(), &[0x41, 0x21]);
}

#[test]
fn encode_merges_by_rank() {
    let tok = gpt2_tokenizer(&["hi"], &["h i"]);
    assert_eq!(tok.encode("hi").unwrap(), &[256]);
}

#[test]
fn encode_without_a_byte_token_is_a_typed_error() {
    let tok = Tokenizer::new(TokenizerModel::GPT2, vec!["a"], None, 0, 1, false);
    assert!(matches!(
        tok.encode("b"),
        Err(TokenizerError::EncodeError(message)) if message == "no token for byte 0x62"
    ));
}

#[test]
fn encode_ignores_spm_normalization() {
    let tok = gpt2_tokenizer(&[], &[]);
    assert_eq!(
        tok.encode("a b").unwrap(),
        &[byte_id(b'a'), byte_id(b' '), byte_id(b'b')]
    );
}

#[test]
fn bpe_applies_the_lowest_rank_merge_first() {
    let extra = ["bc", "ab", "abc"];
    let tok = gpt2_tokenizer(&extra, &["b c", "a b", "a bc"]);
    assert_eq!(tok.encode("abc").unwrap(), &[258]);
    let tok = gpt2_tokenizer(&extra, &["a b", "b c"]);
    assert_eq!(tok.encode("abc").unwrap(), &[257, byte_id(b'c')]);
}

#[test]
fn bpe_breaks_rank_ties_at_the_leftmost_pair() {
    let tok = gpt2_tokenizer(&["aa"], &["a a"]);
    assert_eq!(tok.encode("aaa").unwrap(), &[256, byte_id(b'a')]);
}

#[test]
fn bpe_encodes_each_pre_token_alone() {
    let tok = gpt2_tokenizer(&["ab"], &["a b"]);
    assert_eq!(
        tok.encode("ab  a").unwrap(),
        &[256, byte_id(b' '), byte_id(b' '), byte_id(b'a')]
    );
}

#[test]
fn encode_keeps_whitespace() {
    let tok = gpt2_tokenizer(&[], &[]);
    let ids = tok.encode("a  b\n").unwrap();
    assert_eq!(tok.decode(&ids).unwrap(), "a  b\n");
}
