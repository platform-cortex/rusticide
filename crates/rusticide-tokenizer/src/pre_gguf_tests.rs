// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{build, error_of, gpt2_meta, gpt2_tokens, text, with};
use crate::tokenizer::Tokenizer;

const MERGED: u32 = 256;

fn with_pre(name: &str) -> Vec<(&'static str, rusticide_io::GGUFValue)> {
    with(
        gpt2_meta(&["12"], &["1 2"]),
        "tokenizer.ggml.pre",
        text(name),
    )
}

#[test]
fn from_gguf_reads_the_qwen2_pre_tokenizer() {
    let tok = build(&with_pre("qwen2"));
    assert_eq!(tok.encode("12").unwrap(), &[0x31, 0x32]);
}

#[test]
fn from_gguf_reads_the_gpt2_pre_tokenizer() {
    let tok = build(&with_pre("gpt2"));
    assert_eq!(tok.encode("12").unwrap(), &[MERGED]);
}

#[test]
fn from_gguf_without_pre_uses_the_default_patterns() {
    let tok = build(&gpt2_meta(&["12"], &["1 2"]));
    assert_eq!(tok.encode("12").unwrap(), &[MERGED]);
}

#[test]
fn from_gguf_rejects_an_unsupported_pre_tokenizer() {
    assert!(matches!(
        error_of(&with_pre("mystery")),
        TokenizerError::UnsupportedPreTokenizer(name) if name == "mystery"
    ));
}

#[test]
fn unsupported_pre_tokenizer_names_the_pre_type() {
    let message = TokenizerError::UnsupportedPreTokenizer("x".to_string()).to_string();
    assert_eq!(message, "unsupported pre-tokenizer: x");
}

#[test]
fn new_uses_the_gpt2_pre_tokenizer() {
    let owned = gpt2_tokens(&[]);
    let vocab = owned.iter().map(String::as_str).collect();
    let tok = Tokenizer::new(TokenizerModel::GPT2, vocab, None, 0, 1, false);
    assert_eq!(tok.encode("a b").unwrap(), &[0x61, 0x20, 0x62]);
}
