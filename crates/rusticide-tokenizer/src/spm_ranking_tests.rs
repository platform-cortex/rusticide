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

#[test]
fn the_highest_scoring_pair_merges_first_even_when_a_better_total_exists() {
    let tok = llama(
        vec!["\u{2581}", "a", "b", "\u{2581}a", "ab"],
        vec![-10.0, -0.1, -0.1, -1.0, -0.5],
    );
    assert_eq!(tok.encode("ab").unwrap(), vec![0, 4]);
}

#[test]
fn equal_scoring_pairs_merge_leftmost_first() {
    let mut tok = llama(vec!["a", "aa"], vec![-1.0, -1.0]);
    tok.add_space_prefix = false;
    assert_eq!(tok.encode("aaa").unwrap(), vec![1, 0]);
}

#[test]
fn a_pair_whose_symbol_already_merged_is_skipped() {
    let tok = llama(
        vec!["\u{2581}", "a", "b", "c", "\u{2581}a", "bc", "\u{2581}ab"],
        vec![0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0],
    );
    assert_eq!(tok.encode("abc").unwrap(), vec![4, 5]);
}

#[test]
fn only_merges_through_pieces_in_the_vocabulary() {
    let tok = llama(
        vec!["\u{2581}", "a", "b", "\u{2581}ab"],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(tok.encode("ab").unwrap(), vec![0, 1, 2]);
}

#[test]
fn a_merged_pair_can_merge_again_with_its_left_neighbour() {
    let mut tok = llama(
        vec!["a", "b", "c", "bc", "abc"],
        vec![0.0, 0.0, 0.0, -1.0, -2.0],
    );
    tok.add_space_prefix = false;
    assert_eq!(tok.encode("abc").unwrap(), vec![4]);
}

#[test]
fn a_queued_pair_is_dropped_once_its_left_side_merged_away() {
    let mut tok = llama(
        vec!["x", "a", "b", "c", "xa", "ab", "bc"],
        vec![0.0, 0.0, 0.0, 0.0, -1.0, -5.0, -1.0],
    );
    tok.add_space_prefix = false;
    assert_eq!(tok.encode("xabc").unwrap(), vec![4, 6]);
}
