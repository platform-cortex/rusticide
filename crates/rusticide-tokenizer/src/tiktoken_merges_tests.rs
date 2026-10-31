// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::canonical_merges;
use crate::tokenizer::PieceLookup;

/// The 256 byte tokens (rank = byte value), then `extra` from rank 256.
fn ranks(extra: &[&[u8]]) -> PieceLookup {
    (0..=255u8)
        .map(|byte| vec![byte])
        .chain(extra.iter().map(|token| token.to_vec()))
        .zip(0..)
        .collect()
}

fn id(byte: u8) -> u32 {
    u32::from(byte)
}

#[test]
fn only_the_split_training_produced_is_kept() {
    let merges = canonical_merges(&ranks(&[b"ab", b"bc", b"abc"]));
    assert_eq!(merges.get(&(256, id(b'c'))), Some(&(258, 258)));
    assert_eq!(merges.get(&(id(b'a'), 257)), None);
    assert_eq!(merges.get(&(id(b'a'), id(b'b'))), Some(&(256, 256)));
}

#[test]
fn the_lower_ranked_inner_merge_decides_the_split() {
    let merges = canonical_merges(&ranks(&[b"bc", b"ab", b"abc"]));
    assert_eq!(merges.get(&(id(b'a'), 256)), Some(&(258, 258)));
    assert_eq!(merges.get(&(257, id(b'c'))), None);
}

#[test]
fn a_token_its_lower_ranks_cannot_build_in_two_has_no_merge() {
    let merges = canonical_merges(&ranks(&[b"xyz"]));
    assert!(merges.values().all(|&(rank, _)| rank != 256));
    assert_eq!(merges.len(), 0);
}
