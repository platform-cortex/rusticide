// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Reverse;
use std::ops::RangeInclusive;

use rusticide_io::GGUFValue;

use crate::value_ext::GGUFValueExt;

/// The token types llama.cpp partitions prompts on: UNKNOWN, CONTROL and
/// `USER_DEFINED` (`llama-vocab.cpp` at 8642541b2, lines 3409-3420).
const SPECIAL_TYPES: RangeInclusive<u32> = 2..=4;

/// A piece of a prompt: text still to encode, or a special token's id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fragment<'t> {
    Text(&'t str),
    Token(u32),
}

/// The special tokens' texts and ids, longest text first. The sort is stable,
/// so equal lengths stay in id order.
pub(crate) fn special_tokens(vocab: &[String], types: &[GGUFValue]) -> Vec<(String, u32)> {
    let mut specials: Vec<(String, u32)> = vocab
        .iter()
        .zip(types)
        .enumerate()
        .filter(|(_, (token, kind))| !token.is_empty() && is_special_type(kind))
        .filter_map(|(index, (token, _))| Some((token.clone(), u32::try_from(index).ok()?)))
        .collect();
    specials.sort_by_key(|(text, _)| Reverse(text.len()));
    specials
}

fn is_special_type(kind: &GGUFValue) -> bool {
    kind.as_u32_lossy()
        .is_ok_and(|kind| SPECIAL_TYPES.contains(&kind))
}

/// llama.cpp's `tokenizer_st_partition`: each special token in turn, longest
/// first, splits every remaining text fragment at all its occurrences.
pub(crate) fn partition<'t>(text: &'t str, specials: &[(String, u32)]) -> Vec<Fragment<'t>> {
    let mut fragments = Vec::new();
    if !text.is_empty() {
        fragments.push(Fragment::Text(text));
    }
    for (special, id) in specials {
        fragments = fragments
            .into_iter()
            .flat_map(|fragment| split(fragment, special, *id))
            .collect();
    }
    fragments
}

fn split<'t>(fragment: Fragment<'t>, special: &str, id: u32) -> Vec<Fragment<'t>> {
    let Fragment::Text(text) = fragment else {
        return vec![fragment];
    };
    let mut pieces = Vec::new();
    let mut start = 0;
    for (at, _) in text.match_indices(special) {
        if at > start {
            pieces.push(Fragment::Text(&text[start..at]));
        }
        pieces.push(Fragment::Token(id));
        start = at + special.len();
    }
    if start < text.len() {
        pieces.push(Fragment::Text(&text[start..]));
    }
    pieces
}

#[cfg(test)]
#[path = "special_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "special_spm_tests.rs"]
mod spm_tests;
