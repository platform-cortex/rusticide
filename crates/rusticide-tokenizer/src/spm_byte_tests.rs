// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::model::TokenizerModel;
use crate::tokenizer::Tokenizer;

fn llama(vocab: Vec<&str>, scores: Vec<f32>, add_bos: bool) -> Tokenizer {
    Tokenizer::new(TokenizerModel::LLaMA, vocab, Some(scores), 0, 1, add_bos)
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
fn an_unknown_character_falls_back_to_each_of_its_bytes() {
    let tok = llama(
        vec!["\u{2581}", "<0xC3>", "<0xA9>", "\u{2581}a"],
        vec![0.0; 4],
        false,
    );
    assert_eq!(tok.encode("\u{e9}").unwrap(), &[0, 1, 2]);
}
