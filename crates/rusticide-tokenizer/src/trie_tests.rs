// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Trie;

#[test]
fn prefixes_reports_every_token_along_the_path() {
    let mut trie = Trie::new();
    trie.insert(b"a", 0);
    trie.insert(b"ab", 1);
    let found = trie.prefixes(b"xab", 1, &[-1.0, -2.0]);
    assert_eq!(found, vec![(0, 1, -1.0), (1, 2, -2.0)]);
}

#[test]
fn trie_prefixes_stops_at_a_missing_child() {
    let mut trie = Trie::new();
    trie.insert(b"abc", 0);
    trie.insert(b"a", 1);
    let found = trie.prefixes(b"abx", 0, &[-3.0, -1.0]);
    assert_eq!(found, vec![(1, 1, -1.0)]);
}

#[test]
fn trie_shares_prefixes_between_tokens() {
    let mut trie = Trie::new();
    trie.insert(b"ab", 0);
    trie.insert(b"ac", 1);
    assert_eq!(trie.prefixes(b"ac", 0, &[0.0, -2.0]), vec![(1, 2, -2.0)]);
    assert_eq!(trie.prefixes(b"ab", 0, &[-5.0, 0.0]), vec![(0, 2, -5.0)]);
}

#[test]
fn trie_does_not_skip_over_a_non_matching_byte() {
    let mut trie = Trie::new();
    trie.insert(b"ab", 0);
    let found: Vec<(u32, usize, f32)> = trie.prefixes(b"axb", 0, &[-1.0]);
    assert_eq!(found, Vec::new());
}
