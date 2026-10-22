// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
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
