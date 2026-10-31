// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use rusticide_io::GGUFValue;

use crate::error::TokenizerError;
use crate::test_support::{build, error_of, gpt2_base, gpt2_meta, text, with};
use crate::tokenizer::Tokenizer;

#[test]
fn from_gguf_gpt2_merges_not_an_array() {
    let entries = with(gpt2_base(&[]), "tokenizer.ggml.merges", text("a b"));
    assert!(matches!(
        error_of(&entries),
        TokenizerError::MissingMetadata("expected array value type")
    ));
}

#[test]
fn from_gguf_gpt2_merge_ranks_follow_entry_order() {
    let tok = build(&gpt2_meta(&["ab", "abc"], &["a b", "ab c"]));
    assert_eq!(tok.merge_info.get(&(97, 98)), Some(&(0, 256)));
    assert_eq!(tok.merge_info.get(&(256, 99)), Some(&(1, 257)));
}

#[test]
fn merge_lookup_skips_malformed_pairs() {
    let ids: HashMap<String, u32> = [("a", 0), ("b", 1), ("ab", 2), ("c", 3), ("ac", 4)]
        .into_iter()
        .map(|(token, id)| (token.to_string(), id))
        .collect();
    let merges = [
        text("nospace"),
        text("a x"),
        text("x b"),
        text("b c"),
        text("a c"),
    ];
    let lookup = Tokenizer::build_merge_lookup(&ids, &merges).unwrap();
    assert_eq!(lookup.len(), 1);
    assert_eq!(lookup.get(&(0, 3)), Some(&(4, 4)));
}

#[test]
fn merge_entry_not_a_string_is_an_error() {
    let ids = HashMap::new();
    let merges = [GGUFValue::Uint32(1)];
    assert!(matches!(
        Tokenizer::build_merge_lookup(&ids, &merges),
        Err(TokenizerError::MissingMetadata(
            "expected string value type"
        ))
    ));
}
