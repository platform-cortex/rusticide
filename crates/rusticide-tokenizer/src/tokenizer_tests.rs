// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::tokenizer::{Tokenizer, token_id};

#[test]
fn accessors_report_the_constructor_arguments() {
    let tok = Tokenizer::new(TokenizerModel::GPT2, vec!["a"], None, 3, 4, false);
    assert_eq!(tok.model_type(), TokenizerModel::GPT2);
    assert_eq!(tok.vocab_size(), 1);
    assert_eq!(tok.bos_id(), 3);
    assert_eq!(tok.eos_id(), 4);
}

#[test]
fn new_with_short_scores_is_padded() {
    let tok = Tokenizer::new(
        TokenizerModel::LLaMA,
        vec!["<s>", "</s>", "\u{2581}hi", "\u{2581}h"],
        Some(vec![0.5]),
        0,
        1,
        false,
    );
    assert_eq!(tok.scores, vec![0.5, 0.0, 0.0, 0.0]);
    assert_eq!(tok.encode("hi").unwrap(), vec![2]);
}

#[test]
fn vocab_index_out_of_range() {
    let tok = Tokenizer::new(TokenizerModel::LLaMA, vec!["a", "b"], None, 0, 1, false);
    assert!(matches!(
        tok.vocab_index(2),
        Err(TokenizerError::TokenOutOfRange(2, 2))
    ));
    assert!(matches!(
        tok.vocab_index(u32::MAX),
        Err(TokenizerError::TokenOutOfRange(u32::MAX, 2))
    ));
}

#[test]
fn vocab_index_in_range_is_the_token_id() {
    let tok = Tokenizer::new(TokenizerModel::LLaMA, vec!["a", "b"], None, 0, 1, false);
    assert_eq!(tok.vocab_index(1).unwrap(), 1);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn token_id_overflow_is_vocab_overflow() {
    let overflowing = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    assert!(matches!(
        token_id(overflowing),
        Err(TokenizerError::VocabOverflow)
    ));
}

#[test]
fn token_id_keeps_the_index() {
    assert_eq!(token_id(7).unwrap(), 7);
}
