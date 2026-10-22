// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::model::TokenizerModel;
use crate::tokenizer::Tokenizer;

fn llama(vocab: Vec<&str>, scores: Vec<f32>) -> Tokenizer {
    Tokenizer::new(TokenizerModel::LLaMA, vocab, Some(scores), 0, 1, false)
}

#[test]
fn equal_cost_paths_keep_the_first_one_found() {
    let tok = llama(vec!["\u{2581}ab", "\u{2581}a", "b"], vec![-2.0, -1.0, -1.0]);
    assert_eq!(tok.encode("ab").unwrap(), vec![0]);
}

#[test]
fn a_matched_piece_beats_the_byte_fallback() {
    let mut tok = llama(vec!["a", "<0x61>"], vec![-5.0, -1.0]);
    tok.add_space_prefix = false;
    assert_eq!(tok.encode("a").unwrap(), vec![0]);
}

#[test]
fn byte_fallback_is_charged_its_own_score() {
    let tok = llama(
        vec!["\u{2581}", "\u{2581}a", "<0x61>"],
        vec![-1.0, -1.5, -1.0],
    );
    assert_eq!(tok.encode("a").unwrap(), vec![1]);
}
