// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::test_support::{float_array, meta, tokenizer_meta, with};
use crate::tokenizer::Tokenizer;

fn llama(vocab: Vec<&str>, scores: Vec<f32>, add_bos: bool) -> Tokenizer {
    Tokenizer::new(TokenizerModel::LLaMA, vocab, Some(scores), 0, 1, add_bos)
}

#[test]
fn test_encode_decode_llama_basic() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}hell", "o", "\u{2581}world"],
        vec![0.0; 5],
        false,
    );
    let encoded = tok.encode("hello world").unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, "\u{2581}hello\u{2581}world");
}

#[test]
fn test_encode_decode_llama_with_bos() {
    let tok = llama(vec!["<s>", "</s>", "\u{2581}hello"], vec![0.0; 3], true);
    let encoded = tok.encode("hello").unwrap();
    assert_eq!(encoded.first(), Some(&0));
    assert!(tok.decode(&encoded).unwrap().contains("hello"));
}

#[test]
fn test_encode_without_bos() {
    let tok = llama(vec!["<s>", "</s>", "\u{2581}hello"], vec![0.0; 3], false);
    assert_eq!(tok.encode("hello").unwrap(), &[2]);
}

#[test]
fn test_encode_llama_applies_spm_space_normalization() {
    let tok = llama(
        vec![
            "<s>",
            "</s>",
            "\u{2581}hello",
            "\u{2581}world",
            "h",
            "e",
            "l",
            "o",
            "w",
            "r",
            "d",
            "<0x20>",
        ],
        vec![
            0.0, 0.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ],
        false,
    );
    assert_eq!(tok.encode("hello world").unwrap(), &[2, 3]);
}

fn space_prefix_tokenizer(add_space_prefix: Option<bool>) -> Tokenizer {
    let entries = tokenizer_meta("llama", &["<s>", "</s>", "\u{2581}hello", "hello"]);
    let entries = with(entries, "tokenizer.ggml.scores", float_array(&[0.0; 4]));
    let entries = with(
        entries,
        "tokenizer.ggml.add_bos_token",
        GGUFValue::Bool(false),
    );
    let entries = match add_space_prefix {
        Some(flag) => with(
            entries,
            "tokenizer.ggml.add_space_prefix",
            GGUFValue::Bool(flag),
        ),
        None => entries,
    };
    Tokenizer::from_gguf(&meta(&entries)).unwrap()
}

#[test]
fn test_encode_llama_add_space_prefix_false_from_gguf_skips_leading_space() {
    let tok = space_prefix_tokenizer(Some(false));
    assert_eq!(tok.encode("hello").unwrap(), &[3]);
}

#[test]
fn test_encode_llama_add_space_prefix_true_from_gguf_prepends_leading_space() {
    let tok = space_prefix_tokenizer(None);
    assert_eq!(tok.encode("hello").unwrap(), &[2]);
}

#[test]
fn test_encode_empty_no_bos() {
    let tok = llama(vec!["<s>", "</s>"], vec![0.0; 2], false);
    assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
}

#[test]
fn test_encode_empty_with_bos() {
    let tok = llama(vec!["<s>", "</s>"], vec![0.0; 2], true);
    assert_eq!(tok.encode("").unwrap(), &[0]);
}

#[test]
fn test_encode_decode_utf8() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}caf", "\u{00E9}"],
        vec![0.0; 4],
        false,
    );
    let encoded = tok.encode("caf\u{00E9}").unwrap();
    assert_eq!(tok.decode(&encoded).unwrap(), "\u{2581}caf\u{00E9}");
}

#[test]
fn test_viterbi_chooses_optimal_segmentation() {
    let tok = llama(
        vec!["\u{2581}ab", "\u{2581}a", "b", "cd"],
        vec![-10.0, -1.0, -1.0, -10.0],
        false,
    );
    assert_eq!(tok.encode("ab").unwrap(), &[1, 2]);
    assert_eq!(tok.encode("abcd").unwrap(), &[1, 2, 3]);
}

#[test]
fn test_viterbi_equal_scores() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}hello", "\u{2581}hell", "o"],
        vec![0.0; 5],
        false,
    );
    let encoded = tok.encode("hello").unwrap();
    assert_eq!(tok.decode(&encoded).unwrap(), "\u{2581}hello");
}

#[test]
fn test_byte_fallback_unknown_bytes() {
    let tok = llama(
        vec![
            "<s>", "</s>", "<0xE2>", "<0x96>", "<0x81>", "<0x61>", "<0x62>",
        ],
        vec![0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0],
        false,
    );
    let encoded = tok.encode("ab").unwrap();
    assert_eq!(encoded, &[2, 3, 4, 5, 6]);
    assert_eq!(
        tok.decode(&encoded).unwrap(),
        "<0xE2><0x96><0x81><0x61><0x62>"
    );
}

#[test]
fn test_new_tokenizer_roundtrip() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}hello", "\u{2581}world"],
        vec![0.0; 4],
        false,
    );
    let encoded = tok.encode("hello world").unwrap();
    assert_eq!(tok.decode(&encoded).unwrap(), "\u{2581}hello\u{2581}world");
}

#[test]
fn test_non_ascii_roundtrip() {
    let tok = llama(
        vec!["<s>", "</s>", "\u{2581}\u{00E9}", "\u{2603}"],
        vec![0.0; 4],
        false,
    );
    let encoded = tok.encode("\u{00E9}\u{2603}").unwrap();
    assert_eq!(tok.decode(&encoded).unwrap(), "\u{2581}\u{00E9}\u{2603}");
}

#[test]
fn test_hex_nibble_valid() {
    assert_eq!(super::hex_nibble(b'0'), Some(0));
    assert_eq!(super::hex_nibble(b'9'), Some(9));
    assert_eq!(super::hex_nibble(b'a'), Some(10));
    assert_eq!(super::hex_nibble(b'f'), Some(15));
    assert_eq!(super::hex_nibble(b'A'), Some(10));
    assert_eq!(super::hex_nibble(b'F'), Some(15));
}

#[test]
fn test_hex_nibble_invalid() {
    assert_eq!(super::hex_nibble(b'g'), None);
    assert_eq!(super::hex_nibble(b'z'), None);
    assert_eq!(super::hex_nibble(b'@'), None);
    assert_eq!(super::hex_nibble(b' '), None);
}

#[test]
fn encode_error_variant_is_reachable_through_unmatched_input() {
    let tok = llama(vec!["a"], vec![0.0], false);
    let err = tok.encode("z").unwrap_err();
    assert!(matches!(err, TokenizerError::EncodeError(_)));
}

#[test]
fn byte_fallback_ignores_invalid_hex() {
    let tok = llama(
        vec!["<0xZZ>", "<0x6g>", "<0x6>", "[0x61]", "<0x62>"],
        vec![0.0; 5],
        false,
    );
    let table = tok.build_byte_fallback();
    assert_eq!(table.iter().flatten().count(), 1);
    assert_eq!(table[0x62], Some(4));
}

#[test]
fn byte_fallback_reads_mixed_case_hex() {
    let tok = llama(vec!["<0xaB>", "<0xCd>"], vec![0.0; 2], false);
    let table = tok.build_byte_fallback();
    assert_eq!(table[0xAB], Some(0));
    assert_eq!(table[0xCD], Some(1));
}

#[test]
fn empty_vocabulary_entries_never_match() {
    let tok = llama(vec!["", "\u{2581}a"], vec![0.0; 2], false);
    assert_eq!(tok.encode("a").unwrap(), &[1]);
}

#[test]
fn nan_scores_make_the_input_unencodable_not_a_panic() {
    let tok = llama(vec!["\u{2581}a"], vec![f32::NAN], false);
    assert!(matches!(
        tok.encode("a"),
        Err(TokenizerError::EncodeError(_))
    ));
}

#[test]
fn infinite_scores_are_an_encode_error_not_a_panic() {
    let tok = llama(
        vec!["\u{2581}a", "b", "<0x62>"],
        vec![f32::INFINITY, 0.0, 0.0],
        false,
    );
    assert!(matches!(
        tok.encode("ab"),
        Err(TokenizerError::EncodeError(_))
    ));
}
