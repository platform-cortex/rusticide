// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{TiktokenEncoding, decode_base64};
use crate::error::TokenizerError;
use crate::model::TokenizerModel;
use crate::tokenizer::Tokenizer;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(bytes: &[u8]) -> String {
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &b)| acc | (u32::from(b) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[((word >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The 256 byte tokens (rank = byte value), then `extra` from rank 256.
fn ranks(extra: &[&[u8]]) -> String {
    (0..=255u8)
        .map(|b| vec![b])
        .chain(extra.iter().map(|t| t.to_vec()))
        .enumerate()
        .map(|(rank, token)| format!("{} {rank}\n", base64(&token)))
        .collect::<Vec<_>>()
        .concat()
}

fn cl100k(extra: &[&[u8]]) -> Tokenizer {
    Tokenizer::from_tiktoken(&ranks(extra), TiktokenEncoding::Cl100kBase).unwrap()
}

#[test]
fn base64_decodes_padded_and_unpadded_tokens() {
    assert_eq!(decode_base64("IQ==").unwrap(), b"!");
    assert_eq!(decode_base64("YWI=").unwrap(), b"ab");
    assert_eq!(decode_base64("YWJj").unwrap(), b"abc");
    assert_eq!(decode_base64("/+8=").unwrap(), [0xff, 0xef]);
    assert_eq!(decode_base64("").unwrap(), b"");
}

#[test]
fn base64_rejects_a_bad_character_or_length() {
    assert_eq!(decode_base64("YW*j"), None);
    assert_eq!(decode_base64("YWJ"), None);
    assert_eq!(decode_base64("Y==="), None);
}

#[test]
fn bytes_are_their_own_tokens() {
    let tok = cl100k(&[]);
    assert_eq!(tok.model_type(), TokenizerModel::Tiktoken);
    assert_eq!(
        tok.encode("hi").unwrap(),
        [u32::from(b'h'), u32::from(b'i')]
    );
}

#[test]
fn pairs_merge_by_the_rank_of_their_concatenation() {
    let tok = cl100k(&[b"bc", b"ab", b"abc"]);
    assert_eq!(tok.encode("abc").unwrap(), [258]);
    assert_eq!(tok.encode("abd").unwrap(), [257, u32::from(b'd')]);
    assert_eq!(tok.encode("bcd").unwrap(), [256, u32::from(b'd')]);
}

#[test]
fn the_lowest_ranked_merge_wins() {
    let tok = cl100k(&[b"bc", b"ab"]);
    assert_eq!(tok.encode("abc").unwrap(), [u32::from(b'a'), 256]);
}

#[test]
fn decode_returns_the_raw_bytes() {
    let tok = cl100k(&[b"\xc3\xa1"]);
    assert_eq!(tok.encode("\u{e1}").unwrap(), [256]);
    assert_eq!(tok.decode(&[256, u32::from(b'!')]).unwrap(), "\u{e1}!");
}

#[test]
fn special_tokens_come_from_the_encoding() {
    let tok = cl100k(&[]);
    assert_eq!(tok.encode("<|endoftext|>").unwrap(), [100_257]);
    assert!(tok.is_end_of_generation(100_257));
    let o200k = Tokenizer::from_tiktoken(&ranks(&[]), TiktokenEncoding::O200kBase).unwrap();
    assert_eq!(o200k.encode("<|endofprompt|>").unwrap(), [200_018]);
}

#[test]
fn a_malformed_line_is_rejected_with_its_number() {
    let text = format!("{}YWI= notanumber\n", ranks(&[]));
    let error = Tokenizer::from_tiktoken(&text, TiktokenEncoding::Cl100kBase).unwrap_err();
    assert!(matches!(
        error,
        TokenizerError::InvalidTiktoken { line: 257 }
    ));
    let error =
        Tokenizer::from_tiktoken("not base64! 0\n", TiktokenEncoding::Cl100kBase).unwrap_err();
    assert!(matches!(error, TokenizerError::InvalidTiktoken { line: 1 }));
}

#[test]
fn an_absurd_rank_is_rejected_before_allocating() {
    let text = format!("YWI= {}\n", u32::MAX);
    let error = Tokenizer::from_tiktoken(&text, TiktokenEncoding::Cl100kBase).unwrap_err();
    assert!(matches!(error, TokenizerError::InvalidTiktoken { line: 1 }));
}

#[test]
fn a_vocabulary_missing_a_byte_is_rejected() {
    let full = ranks(&[]);
    let text = full.split_once('\n').unwrap().1;
    let error = Tokenizer::from_tiktoken(text, TiktokenEncoding::Cl100kBase).unwrap_err();
    assert!(matches!(
        error,
        TokenizerError::MissingByteToken { byte: 0 }
    ));
}
