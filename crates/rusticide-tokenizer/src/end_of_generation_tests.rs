// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::model::TokenizerModel;
use crate::test_support::{build, gpt2_base, with};
use crate::tokenizer::Tokenizer;

const ENDOFTEXT: u32 = 256;
const IM_END: u32 = 257;
const PLAIN: u32 = 258;
const OTHER: u32 = 259;

fn extra_tokens() -> [&'static str; 4] {
    ["<|endoftext|>", "<|im_end|>", "plain", "<|other|>"]
}

fn with_ids(eos: i32, eot: Option<i32>) -> Tokenizer {
    let entries = with(
        gpt2_base(&extra_tokens()),
        "tokenizer.ggml.eos_token_id",
        GGUFValue::Int32(eos),
    );
    let entries = match eot {
        Some(id) => with(entries, "tokenizer.ggml.eot_token_id", GGUFValue::Int32(id)),
        None => entries,
    };
    build(&entries)
}

#[test]
fn eos_ends_generation() {
    let tok = with_ids(i32::try_from(PLAIN).unwrap(), None);
    assert!(tok.is_end_of_generation(PLAIN));
}

#[test]
fn endoftext_ends_generation() {
    let tok = with_ids(i32::try_from(PLAIN).unwrap(), None);
    assert!(tok.is_end_of_generation(ENDOFTEXT));
    assert!(tok.is_end_of_generation(IM_END));
}

#[test]
fn eot_token_id_ends_generation() {
    let tok = with_ids(
        i32::try_from(PLAIN).unwrap(),
        Some(i32::try_from(OTHER).unwrap()),
    );
    assert!(tok.is_end_of_generation(OTHER));
}

#[test]
fn ordinary_tokens_do_not_end_generation() {
    let tok = with_ids(i32::try_from(PLAIN).unwrap(), None);
    assert!(!tok.is_end_of_generation(OTHER));
    assert!(!tok.is_end_of_generation(0x61));
    assert!(!tok.is_end_of_generation(u32::MAX));
}

#[test]
fn a_repeated_end_token_is_stored_once() {
    let tok = with_ids(
        i32::try_from(ENDOFTEXT).unwrap(),
        Some(i32::try_from(ENDOFTEXT).unwrap()),
    );
    assert_eq!(tok.end_of_generation, [ENDOFTEXT, IM_END]);
}

#[test]
fn new_ends_generation_only_at_eos() {
    let tok = Tokenizer::new(
        TokenizerModel::LLaMA,
        vec!["<s>", "</s>", "x"],
        None,
        0,
        1,
        false,
    );
    assert_eq!(tok.end_of_generation, [1]);
    assert!(tok.is_end_of_generation(1));
    assert!(!tok.is_end_of_generation(2));
}
