// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{
    build, error_of, float_array, llama_fixture, meta, text, text_array, tokenizer_meta, with,
};
use crate::tokenizer::Tokenizer;

#[test]
fn from_gguf_reads_real_bos_eos_token_id_keys() {
    let entries = tokenizer_meta("llama", &["<s>", "</s>", "<unk>"]);
    let entries = with(entries, "tokenizer.ggml.bos_token_id", GGUFValue::Uint32(1));
    let entries = with(entries, "tokenizer.ggml.eos_token_id", GGUFValue::Uint32(2));
    let tok = build(&entries);
    assert_eq!(tok.bos_id(), 1);
    assert_eq!(tok.eos_id(), 2);
}

#[test]
fn test_from_gguf_llama() {
    let tok = Tokenizer::from_gguf(&llama_fixture()).unwrap();
    assert_eq!(tok.model_type(), TokenizerModel::LLaMA);
    assert_eq!(tok.vocab_size(), 8);
    assert_eq!(tok.bos_id(), 0);
    assert_eq!(tok.eos_id(), 1);
    assert!(tok.add_bos);
}

#[test]
fn test_from_gguf_gpt2() {
    let entries = tokenizer_meta("gpt2", &["!", "\"", "#", "$", "%"]);
    let entries = with(
        entries,
        "tokenizer.ggml.add_bos_token",
        GGUFValue::Bool(false),
    );
    let tok = build(&entries);
    assert_eq!(tok.model_type(), TokenizerModel::GPT2);
    assert_eq!(tok.vocab_size(), 5);
    assert!(!tok.add_bos);
}

#[test]
fn test_from_gguf_unknown_model() {
    let err = Tokenizer::from_gguf(&meta(&tokenizer_meta("unknown", &["a"]))).unwrap_err();
    assert!(matches!(err, TokenizerError::UnknownModel(name) if name == "unknown"));
}

#[test]
fn test_from_gguf_missing_required() {
    let err = Tokenizer::from_gguf(&Metadata::default()).unwrap_err();
    assert!(matches!(
        err,
        TokenizerError::MissingMetadata("tokenizer.ggml.model")
    ));
}

#[test]
fn test_from_gguf_add_bos_default_llama() {
    assert!(build(&tokenizer_meta("llama", &["a", "b"])).add_bos);
}

#[test]
fn test_from_gguf_add_bos_default_gpt2() {
    assert!(!build(&tokenizer_meta("gpt2", &["a", "b"])).add_bos);
}

#[test]
fn test_from_gguf_bos_eos_default_llama() {
    let tok = build(&tokenizer_meta("llama", &["<s>", "</s>", "<unk>"]));
    assert_eq!(tok.bos_id(), 1);
    assert_eq!(tok.eos_id(), 2);
}

#[test]
fn test_from_gguf_bos_eos_default_gpt2() {
    let tok = build(&tokenizer_meta("gpt2", &["!", "\"", "#"]));
    assert_eq!(tok.bos_id(), 0);
    assert_eq!(tok.eos_id(), 2);
}

#[test]
fn from_gguf_short_scores_are_padded() {
    let entries = tokenizer_meta("llama", &["<s>", "</s>", "\u{2581}hi", "x"]);
    let entries = with(entries, "tokenizer.ggml.scores", float_array(&[0.5]));
    let tok = build(&entries);
    assert_eq!(tok.scores, vec![0.5, 0.0, 0.0, 0.0]);
    assert_eq!(tok.encode("hi").unwrap(), vec![1, 2]);
}

#[test]
fn from_gguf_missing_model_key() {
    let entries = vec![("tokenizer.ggml.tokens", text_array(&["a"]))];
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("tokenizer.ggml.model")
    ));
}

#[test]
fn from_gguf_missing_tokens_key() {
    let entries = vec![("tokenizer.ggml.model", text("llama"))];
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("tokenizer.ggml.tokens")
    ));
}

#[test]
fn from_gguf_model_not_a_string() {
    let entries = vec![("tokenizer.ggml.model", GGUFValue::Uint32(1))];
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected string value type")
    ));
}

#[test]
fn from_gguf_tokens_not_an_array() {
    let entries = vec![
        ("tokenizer.ggml.model", text("llama")),
        ("tokenizer.ggml.tokens", text("a")),
    ];
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected array value type")
    ));
}

#[test]
fn from_gguf_token_not_a_string() {
    let entries = vec![
        ("tokenizer.ggml.model", text("llama")),
        (
            "tokenizer.ggml.tokens",
            GGUFValue::Array(vec![GGUFValue::Uint32(1)]),
        ),
    ];
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected string value type")
    ));
}

#[test]
fn from_gguf_scores_not_an_array() {
    let entries = with(
        tokenizer_meta("llama", &["a"]),
        "tokenizer.ggml.scores",
        GGUFValue::Float32(1.0),
    );
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected array value type")
    ));
}

#[test]
fn from_gguf_non_float_score_entries_become_zero() {
    let scores = GGUFValue::Array(vec![GGUFValue::Uint32(7), GGUFValue::Float32(2.0)]);
    let entries = with(
        tokenizer_meta("llama", &["a", "b"]),
        "tokenizer.ggml.scores",
        scores,
    );
    assert_eq!(build(&entries).scores, vec![0.0, 2.0]);
}

#[test]
fn from_gguf_bad_bos_id_type() {
    let entries = with(
        tokenizer_meta("llama", &["a"]),
        "tokenizer.ggml.bos_token_id",
        text("1"),
    );
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected integer id value type")
    ));
}

#[test]
fn from_gguf_negative_eos_id_is_invalid() {
    let entries = with(
        tokenizer_meta("llama", &["a"]),
        "tokenizer.ggml.eos_token_id",
        GGUFValue::Int32(-4),
    );
    assert!(matches!(error_of(&entries), TokenizerError::InvalidId(-4)));
}

#[test]
fn from_gguf_add_bos_wrong_type_uses_default() {
    let wrong = || GGUFValue::Uint32(0);
    let llama = with(
        tokenizer_meta("llama", &["a"]),
        "tokenizer.ggml.add_bos_token",
        wrong(),
    );
    let gpt2 = with(
        tokenizer_meta("gpt2", &["a"]),
        "tokenizer.ggml.add_bos_token",
        wrong(),
    );
    assert!(build(&llama).add_bos);
    assert!(!build(&gpt2).add_bos);
}

#[test]
fn from_gguf_add_space_prefix_wrong_type_uses_default() {
    let wrong = || GGUFValue::Uint32(0);
    let key = "tokenizer.ggml.add_space_prefix";
    let llama = with(tokenizer_meta("llama", &["a"]), key, wrong());
    let gpt2 = with(tokenizer_meta("gpt2", &["a"]), key, wrong());
    assert!(build(&llama).add_space_prefix);
    assert!(!build(&gpt2).add_space_prefix);
}

#[test]
fn from_gguf_gpt2_without_merges() {
    let tok = build(&tokenizer_meta("gpt2", &["a", "b"]));
    assert!(tok.merge_info.is_empty());
}

#[test]
fn from_gguf_llama_ignores_merges() {
    let entries = with(
        tokenizer_meta("llama", &["a", "b", "ab"]),
        "tokenizer.ggml.merges",
        text_array(&["a b"]),
    );
    assert!(build(&entries).merge_info.is_empty());
}

#[test]
fn from_gguf_absent_scores_are_all_zero() {
    let tok = build(&tokenizer_meta("llama", &["a", "b"]));
    assert_eq!(tok.scores, vec![0.0, 0.0]);
}
