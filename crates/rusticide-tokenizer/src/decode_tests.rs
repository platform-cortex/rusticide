// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{byte_id, gpt2_tokenizer};
use crate::tokenizer::Tokenizer;

fn llama(vocab: Vec<&str>, scores: Option<Vec<f32>>, add_bos: bool) -> Tokenizer {
    Tokenizer::new(TokenizerModel::LLaMA, vocab, scores, 0, 1, add_bos)
}

#[test]
fn test_decode_empty() {
    let tok = llama(vec!["<s>", "</s>"], Some(vec![0.0; 2]), false);
    assert_eq!(tok.decode(&[]).unwrap(), "");
}

#[test]
fn test_decode_out_of_range() {
    let tok = llama(vec!["a", "b"], Some(vec![0.0; 2]), false);
    let err = tok.decode(&[99]).unwrap_err();
    assert!(matches!(err, TokenizerError::TokenOutOfRange(99, 2)));
}

#[test]
fn test_decode_detokenized_skips_special() {
    let tok = llama(vec!["<s>", "</s>", "hello"], Some(vec![0.0; 3]), true);
    assert_eq!(tok.decode_detokenized(&[0, 2, 1]).unwrap(), "hello");
}

#[test]
fn test_decode_detokenized_filters_bos_eos() {
    let tok = llama(vec!["<s>", "</s>", "hello"], None, true);
    assert_eq!(tok.decode_detokenized(&[0, 2, 1]).unwrap(), "hello");
}

#[test]
fn decode_detokenized_spaces_before_marker_pieces_after_the_first() {
    let tok = llama(
        vec!["<s>", "</s>", "hello", "\u{2581}world", "!"],
        None,
        false,
    );
    assert_eq!(
        tok.decode_detokenized(&[2, 3, 4]).unwrap(),
        "hello \u{2581}world!"
    );
}

#[test]
fn decode_detokenized_does_not_space_the_first_piece() {
    let tok = llama(vec!["<s>", "</s>", "\u{2581}hi"], None, false);
    assert_eq!(tok.decode_detokenized(&[2]).unwrap(), "\u{2581}hi");
}

#[test]
fn decode_detokenized_out_of_range() {
    let tok = llama(vec!["<s>", "</s>"], None, false);
    assert!(matches!(
        tok.decode_detokenized(&[7]),
        Err(TokenizerError::TokenOutOfRange(7, 2))
    ));
}

#[test]
fn llama_decode_turns_the_space_marker_into_a_space() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}there", "\u{2581}was"],
        None,
        true,
    );
    assert_eq!(tok.decode(&[2, 3]).unwrap(), " there was");
}

#[test]
fn llama_decode_turns_byte_pieces_into_their_bytes() {
    let tok = llama(
        vec!["<s>", "</s>", "<0x0A>", "<0xC3>", "<0xA9>"],
        None,
        true,
    );
    assert_eq!(tok.decode(&[2, 3, 4]).unwrap(), "\n\u{e9}");
}

fn gpt2_with(extra: &[&str]) -> Tokenizer {
    gpt2_tokenizer(extra, &[])
}

#[test]
fn decode_reverses_the_byte_mapping() {
    let tok = gpt2_with(&["\u{120}ol\u{c3}\u{a1}"]);
    assert_eq!(tok.decode(&[256]).unwrap(), " ol\u{e1}");
}

#[test]
fn decode_joins_pieces_before_reading_utf8() {
    let tok = gpt2_with(&[]);
    let split = [byte_id(0xC3), byte_id(0xA9)];
    assert_eq!(tok.decode(&split).unwrap(), "\u{e9}");
}

#[test]
fn decode_passes_special_tokens_through() {
    let tok = gpt2_with(&["<|im_end|>", "\u{2581}"]);
    assert_eq!(tok.decode(&[256]).unwrap(), "<|im_end|>");
    assert_eq!(tok.decode(&[257]).unwrap(), "\u{2581}");
}

#[test]
fn encode_then_decode_round_trips_utf8() {
    let tok = gpt2_with(&[]);
    let ids = tok.encode("caf\u{e9}\u{4e16}").unwrap();
    assert_eq!(tok.decode(&ids).unwrap(), "caf\u{e9}\u{4e16}");
}
