// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};

use crate::byte_unicode::byte_to_char;
use crate::error::TokenizerError;
use crate::tokenizer::Tokenizer;

pub(crate) fn meta(entries: &[(&str, GGUFValue)]) -> Metadata {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

pub(crate) fn text(value: &str) -> GGUFValue {
    GGUFValue::String(value.to_string())
}

pub(crate) fn text_array(items: &[&str]) -> GGUFValue {
    GGUFValue::Array(items.iter().map(|item| text(item)).collect())
}

pub(crate) fn float_array(items: &[f32]) -> GGUFValue {
    GGUFValue::Array(items.iter().copied().map(GGUFValue::Float32).collect())
}

pub(crate) fn tokenizer_meta(model: &str, tokens: &[&str]) -> Vec<(&'static str, GGUFValue)> {
    vec![
        ("tokenizer.ggml.model", text(model)),
        ("tokenizer.ggml.tokens", text_array(tokens)),
    ]
}

pub(crate) fn with(
    mut entries: Vec<(&'static str, GGUFValue)>,
    key: &'static str,
    value: GGUFValue,
) -> Vec<(&'static str, GGUFValue)> {
    entries.push((key, value));
    entries
}

pub(crate) fn llama_fixture() -> Metadata {
    let entries = tokenizer_meta(
        "llama",
        &["<s>", "</s>", "<unk>", "hello", " world", "hell", "o", "lo"],
    );
    let entries = with(
        entries,
        "tokenizer.ggml.scores",
        float_array(&[0.0, 0.0, 0.0, -2.0, -1.5, -1.0, -0.5, -0.8]),
    );
    let entries = with(entries, "tokenizer.ggml.bos_token_id", GGUFValue::Int32(0));
    let entries = with(entries, "tokenizer.ggml.eos_token_id", GGUFValue::Int32(1));
    let entries = with(
        entries,
        "tokenizer.ggml.add_bos_token",
        GGUFValue::Bool(true),
    );
    meta(&entries)
}

/// The 256 byte tokens of a GPT-2 vocabulary (id = byte), then `extra`.
pub(crate) fn gpt2_tokens(extra: &[&str]) -> Vec<String> {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|byte| byte_to_char(byte).to_string())
        .collect();
    tokens.extend(extra.iter().map(|token| (*token).to_string()));
    tokens
}

/// The id of the byte token for `byte` in a `gpt2_tokens` vocabulary.
pub(crate) const fn byte_id(byte: u8) -> u32 {
    byte as u32
}

/// Metadata for a GPT-2 tokenizer: byte tokens, `extra` tokens and `merges`.
pub(crate) fn gpt2_meta(extra: &[&str], merges: &[&str]) -> Vec<(&'static str, GGUFValue)> {
    with(
        gpt2_base(extra),
        "tokenizer.ggml.merges",
        text_array(merges),
    )
}

/// Like `gpt2_meta` but with no merges key at all.
pub(crate) fn gpt2_base(extra: &[&str]) -> Vec<(&'static str, GGUFValue)> {
    let owned = gpt2_tokens(extra);
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    tokenizer_meta("gpt2", &refs)
}

pub(crate) fn gpt2_tokenizer(extra: &[&str], merges: &[&str]) -> Tokenizer {
    build(&gpt2_meta(extra, merges))
}

pub(crate) fn build(entries: &[(&str, GGUFValue)]) -> Tokenizer {
    Tokenizer::from_gguf(&meta(entries)).unwrap()
}

pub(crate) fn error_of(entries: &[(&str, GGUFValue)]) -> TokenizerError {
    Tokenizer::from_gguf(&meta(entries)).unwrap_err()
}
