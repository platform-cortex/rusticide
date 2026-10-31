// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::error::TokenizerError;
use crate::test_support::{build, byte_id, error_of, gpt2_meta, with};
use crate::tokenizer::Tokenizer;

const CONTROL: i32 = 3;

/// A GPT-2 tokenizer whose `extra` tokens (ids from 256) have `types`.
fn gpt2_typed(extra: &[&str], types: &[i32]) -> Tokenizer {
    let mut all = vec![1; 256];
    all.extend_from_slice(types);
    let values = all.into_iter().map(GGUFValue::Int32).collect();
    build(&with(
        gpt2_meta(extra, &[]),
        "tokenizer.ggml.token_type",
        GGUFValue::Array(values),
    ))
}

fn chat() -> Tokenizer {
    gpt2_typed(&["<|im_start|>", "<|im_end|>"], &[CONTROL, CONTROL])
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(byte_id).collect()
}

#[test]
fn control_tokens_in_a_gpt2_prompt_become_their_ids() {
    let mut expected = vec![256];
    expected.extend(bytes("user\nhi"));
    expected.push(257);
    let ids = chat().encode("<|im_start|>user\nhi<|im_end|>").unwrap();
    assert_eq!(ids, expected);
}

#[test]
fn adjacent_special_tokens_are_each_emitted() {
    let ids = chat().encode("<|im_start|><|im_end|><|im_end|>").unwrap();
    assert_eq!(ids, [256, 257, 257]);
}

#[test]
fn special_tokens_are_found_at_both_ends() {
    let tok = chat();
    assert_eq!(tok.encode("<|im_end|>a").unwrap(), [257, byte_id(b'a')]);
    assert_eq!(tok.encode("a<|im_start|>").unwrap(), [byte_id(b'a'), 256]);
}

#[test]
fn the_longer_special_token_is_matched_first() {
    let tok = gpt2_typed(&["ab", "abc"], &[CONTROL, CONTROL]);
    assert_eq!(tok.encode("abcab").unwrap(), [257, 256]);
}

#[test]
fn each_special_token_claims_the_whole_text_before_shorter_ones() {
    let tok = gpt2_typed(&["ab", "bcd"], &[CONTROL, CONTROL]);
    assert_eq!(tok.encode("abcd").unwrap(), [byte_id(b'a'), 257]);
}

#[test]
fn unknown_and_user_defined_tokens_are_special_too() {
    let tok = gpt2_typed(&["<u>", "<d>"], &[2, 4]);
    assert_eq!(tok.encode("<d><u>").unwrap(), [257, 256]);
}

#[test]
fn ordinary_unused_and_byte_tokens_are_not_special() {
    for kind in [0, 1, 5, 6] {
        let tok = gpt2_typed(&["<|im_start|>"], &[kind]);
        assert_eq!(tok.encode("<|im_start|>").unwrap(), bytes("<|im_start|>"));
    }
}

#[test]
fn an_empty_special_token_matches_nothing() {
    let tok = gpt2_typed(&[""], &[CONTROL]);
    assert_eq!(tok.encode("ab").unwrap(), bytes("ab"));
}

#[test]
fn token_types_that_are_not_an_array_are_rejected() {
    let entries = with(
        gpt2_meta(&[], &[]),
        "tokenizer.ggml.token_type",
        GGUFValue::Int32(CONTROL),
    );
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected array value type")
    ));
}

#[test]
fn a_non_integer_token_type_is_an_ordinary_token() {
    let mut values = vec![GGUFValue::Int32(1); 256];
    values.push(GGUFValue::String("3".to_string()));
    let tok = build(&with(
        gpt2_meta(&["<x>"], &[]),
        "tokenizer.ggml.token_type",
        GGUFValue::Array(values),
    ));
    assert_eq!(tok.encode("<x>").unwrap(), bytes("<x>"));
}
