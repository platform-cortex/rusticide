// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pre-tokenizer must split exactly like llama.cpp's patterns run on a
//! backtracking engine, look-ahead included.

use fancy_regex::Regex;

use crate::pre_tokenizer::{CL100K, DEFAULT, GPT2, O200K, PreTokenizer, QWEN2};

const ALPHABET: &[&str] = &[
    " ",
    " ",
    " ",
    "\n",
    "\r",
    "\t",
    "a",
    "b",
    "Z",
    "\u{e1}",
    "\u{e9}",
    "1",
    "7",
    "'",
    "s",
    ",",
    ".",
    "-",
    "<",
    "\u{a0}",
    "\u{3000}",
    "\u{4e2d}",
    "\u{1f600}",
    "\u{2028}",
    "\u{436}",
    "\u{628}",
    "\u{301}",
    "\u{1c5}",
    "\u{2b0}",
    "A",
    "\u{c9}",
];

fn reference<'t>(patterns: &[&str], text: &'t str) -> Vec<&'t str> {
    let mut pieces = vec![text];
    for pattern in patterns {
        let regex = Regex::new(pattern).unwrap();
        let mut next = Vec::new();
        for piece in pieces {
            let mut last = 0;
            for found in regex.find_iter(piece) {
                let found = found.unwrap();
                next.push(&piece[last..found.start()]);
                next.push(found.as_str());
                last = found.end();
            }
            next.push(&piece[last..]);
        }
        next.retain(|piece| !piece.is_empty());
        pieces = next;
    }
    pieces
}

fn texts() -> impl Iterator<Item = String> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    (0..4000).map(move |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let len = usize::try_from(state >> 59).unwrap() + 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ALPHABET[usize::try_from(state >> 33).unwrap() % ALPHABET.len()]
            })
            .collect()
    })
}

#[test]
fn every_pre_tokenizer_splits_like_its_reference_patterns() {
    for (name, patterns) in [
        (None, DEFAULT),
        (Some("gpt2"), GPT2),
        (Some("qwen2"), QWEN2),
    ] {
        let pre = PreTokenizer::from_name(name).unwrap();
        for text in texts() {
            assert_eq!(
                pre.split(&text).unwrap(),
                reference(patterns, &text),
                "{name:?} {text:?}"
            );
        }
    }
}

#[test]
fn tiktoken_pre_tokenizers_split_like_their_possessive_originals() {
    for (pre, patterns) in [(PreTokenizer::CL100K, CL100K), (PreTokenizer::O200K, O200K)] {
        for text in texts() {
            assert_eq!(
                pre.split(&text).unwrap(),
                reference(patterns, &text),
                "{patterns:?} {text:?}"
            );
        }
    }
}
