// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Trie;

fn long_token_trie(depth: usize) -> Trie {
    let mut trie = Trie::new();
    trie.insert(b"a", 0);
    trie.insert(&vec![b'a'; depth], 1);
    trie
}

#[test]
fn the_walk_visits_at_most_the_input_times_the_match_depth() {
    let depth = 1024;
    let trie = long_token_trie(depth);
    let input = vec![b'a'; 4096];
    let mut visits = 0;
    for start in 0..input.len() {
        trie.walk(&input, start, &[0.0, 0.0], &mut visits);
    }
    assert!(
        visits <= input.len() * (depth + 1),
        "visited {visits} nodes"
    );
}

#[test]
fn the_walk_stops_at_the_first_missing_child() {
    let trie = long_token_trie(1024);
    let input = vec![b'b'; 4096];
    let mut visits = 0;
    for start in 0..input.len() {
        trie.walk(&input, start, &[0.0, 0.0], &mut visits);
    }
    assert_eq!(visits, input.len());
}

#[test]
fn sibling_edges_are_found_in_any_insertion_order() {
    let mut trie = Trie::new();
    for (index, byte) in b"zamb".iter().copied().enumerate() {
        trie.insert(&[byte], u32::try_from(index).unwrap());
    }
    let scores = [0.0, -1.0, -2.0, -3.0];
    assert_eq!(trie.prefixes(b"m", 0, &scores), vec![(2, 1, -2.0)]);
    assert_eq!(trie.prefixes(b"b", 0, &scores), vec![(3, 1, -3.0)]);
    assert_eq!(trie.prefixes(b"z", 0, &scores), vec![(0, 1, 0.0)]);
}

#[test]
fn an_empty_token_is_never_reported() {
    let mut trie = Trie::new();
    trie.insert(b"", 0);
    assert_eq!(trie.prefixes(b"x", 0, &[0.0]), Vec::new());
}

#[test]
fn reinserting_a_piece_replaces_its_token_id() {
    let mut trie = Trie::new();
    trie.insert(b"ab", 0);
    trie.insert(b"ab", 1);
    assert_eq!(trie.prefixes(b"ab", 0, &[-1.0, -2.0]), vec![(1, 2, -2.0)]);
}

fn wide_trie(depth: usize, fan_out: &[u8]) -> Trie {
    let mut vocab = Vec::new();
    for prefix_len in 1..=depth {
        for &byte in fan_out {
            let mut token = vec![b'a'; prefix_len];
            token.push(byte);
            vocab.push(token);
        }
    }
    Trie::from_vocab(&vocab)
}

#[test]
fn a_wide_node_costs_logarithmic_comparisons_per_byte() {
    let depth = 32;
    let fan_out: Vec<u8> = (0x20..=0xFF).collect();
    let trie = wide_trie(depth, &fan_out);
    let scores = vec![0.0; depth * fan_out.len()];
    let input = vec![b'a'; 256];
    let mut comparisons = 0;
    for start in 0..input.len() {
        trie.walk(&input, start, &scores, &mut comparisons);
    }
    let log2_fan_out = (usize::BITS - (fan_out.len() - 1).leading_zeros()) as usize;
    let bound = input.len() * (depth + 1) * (log2_fan_out + 2);
    assert!(comparisons <= bound, "{comparisons} comparisons > {bound}");
}

#[test]
fn one_lookup_among_224_children_costs_exactly_nine_comparisons() {
    let vocab: Vec<Vec<u8>> = (0x20..=0xFF).map(|byte| vec![byte]).collect();
    let trie = Trie::from_vocab(&vocab);
    let mut comparisons = 0;
    let found = trie.walk(b"a", 0, &vec![0.0; vocab.len()], &mut comparisons);
    assert_eq!(found, vec![(0x61 - 0x20, 1, 0.0)]);
    assert_eq!(comparisons, 9);
}
