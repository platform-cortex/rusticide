// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hash::{BuildHasher, Hasher};

use super::PairHasher;
use crate::tokenizer::MergeLookup;

fn hash(pair: (u32, u32)) -> u64 {
    MergeLookup::default().hasher().hash_one(pair)
}

#[test]
fn pairs_hash_by_order() {
    assert_ne!(hash((1, 2)), hash((2, 1)));
    assert_eq!(hash((7, 9)), hash((7, 9)));
}

#[test]
fn nearby_pairs_spread_over_the_high_bits() {
    let tops: std::collections::HashSet<u64> = (0..64).map(|id| hash((id, id + 1)) >> 57).collect();
    assert!(
        tops.len() > 32,
        "only {} distinct top-7-bit values",
        tops.len()
    );
}

#[test]
fn bytes_hash_like_their_values() {
    let mut bytes = PairHasher::default();
    bytes.write(&[1, 2]);
    let mut words = PairHasher::default();
    words.write_u32(1);
    words.write_u32(2);
    assert_eq!(bytes.finish(), words.finish());
}

#[test]
fn a_lookup_finds_what_was_inserted() {
    let mut lookup = MergeLookup::default();
    lookup.insert((3, 4), (0, 5));
    assert_eq!(lookup.get(&(3, 4)), Some(&(0, 5)));
    assert_eq!(lookup.get(&(4, 3)), None);
}
