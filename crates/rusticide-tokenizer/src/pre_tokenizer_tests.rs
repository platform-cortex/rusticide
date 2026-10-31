// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::pre_tokenizer::PreTokenizer;

fn split(name: Option<&str>, text: &str) -> Vec<String> {
    let pre = PreTokenizer::from_name(name).unwrap();
    pre.split(text)
        .unwrap()
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn qwen2(text: &str) -> Vec<String> {
    split(Some("qwen2"), text)
}

#[test]
fn qwen2_splits_words_with_their_leading_space() {
    assert_eq!(qwen2("Hello world"), ["Hello", " world"]);
}

#[test]
fn qwen2_splits_a_space_run_before_a_word() {
    assert_eq!(qwen2("a  b"), ["a", " ", " b"]);
}

#[test]
fn qwen2_splits_newline_runs() {
    assert_eq!(qwen2("x\n\ny"), ["x", "\n\n", "y"]);
}

#[test]
fn qwen2_splits_digits_one_by_one() {
    assert_eq!(qwen2("2024"), ["2", "0", "2", "4"]);
}

#[test]
fn qwen2_splits_contractions() {
    assert_eq!(qwen2("don't"), ["don", "'t"]);
}

#[test]
fn qwen2_keeps_accented_letters_together() {
    assert_eq!(qwen2("Ol\u{e1}, meu"), ["Ol\u{e1}", ",", " meu"]);
}

#[test]
fn gpt2_keeps_digit_runs() {
    assert_eq!(split(Some("gpt2"), "2024"), ["2024"]);
}

#[test]
fn gpt2_is_also_known_by_the_name_llama_cpp_reads() {
    assert_eq!(split(Some("gpt-2"), "a 12"), ["a", " 12"]);
}

#[test]
fn absent_pre_uses_the_default_patterns() {
    assert_eq!(split(None, "a  b"), split(Some("gpt2"), "a  b"));
    assert_eq!(split(None, "a  b"), ["a", " ", " b"]);
}

#[test]
fn the_default_patterns_split_digits_in_threes() {
    assert_eq!(split(None, "12345"), ["123", "45"]);
}

#[test]
fn empty_text_has_no_pieces() {
    assert_eq!(qwen2(""), Vec::<String>::new());
}

#[test]
fn unknown_pre_is_unsupported() {
    assert!(matches!(
        PreTokenizer::from_name(Some("mystery")),
        Err(TokenizerError::UnsupportedPreTokenizer(name)) if name == "mystery"
    ));
}
