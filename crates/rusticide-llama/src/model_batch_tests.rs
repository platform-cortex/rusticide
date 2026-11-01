// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, Model, ModelError};

use super::Llama;
use crate::Variant;
use crate::test_support::tensor;
use crate::testing::{TinySource, tiny_source, tiny_tensor};

const PARTS: [(&str, [usize; 2]); 7] = [
    ("attn_q.weight", [4, 4]),
    ("attn_k.weight", [2, 4]),
    ("attn_v.weight", [2, 4]),
    ("attn_output.weight", [4, 4]),
    ("ffn_gate.weight", [8, 4]),
    ("ffn_up.weight", [8, 4]),
    ("ffn_down.weight", [4, 8]),
];
const BIASES: [(&str, usize); 3] = [("attn_q.bias", 4), ("attn_k.bias", 2), ("attn_v.bias", 2)];
const TOKENS: [u32; 8] = [1, 5, 6, 4, 3, 6, 2, 5];

/// The tiny model with a second block, so that a token's output feeds the
/// keys and values later tokens attend to.
fn two_blocks(source: TinySource, architecture: &str, biases: bool) -> InMemorySource {
    let count = GGUFValue::Uint32(2);
    let mut source = source.with_metadata(&format!("{architecture}.block_count"), count);
    for (salt, (part, shape)) in PARTS.into_iter().enumerate() {
        let weight = tiny_tensor(part, &shape, salt + 3).unwrap();
        source = source.with_tensor(&format!("blk.1.{part}"), weight);
    }
    for (salt, (part, len)) in BIASES.into_iter().enumerate().filter(|_| biases) {
        let bias = tiny_tensor(part, &[len], salt + 1).unwrap();
        source = source.with_tensor(&format!("blk.1.{part}"), bias);
    }
    let norm = tensor(&[4], &[1.5, 0.5, 1.0, 2.0]);
    source
        .with_tensor("blk.1.attn_norm.weight", norm.clone())
        .with_tensor("blk.1.ffn_norm.weight", norm)
        .build()
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

fn variants() -> [(InMemorySource, Variant); 2] {
    [
        (
            two_blocks(TinySource::new(), "llama", false),
            Variant::Llama,
        ),
        (
            two_blocks(TinySource::qwen2(), "qwen2", true),
            Variant::Qwen2,
        ),
    ]
}

#[test]
fn batches_match_one_token_at_a_time_bit_for_bit() {
    for (source, variant) in variants() {
        let mut single = Llama::from_source(&source, variant).unwrap();
        let expected: Vec<Vec<u32>> = (0..)
            .zip(TOKENS)
            .map(|(position, token)| bits(&single.forward(token, position).unwrap()))
            .collect();
        let mut batched = Llama::from_source(&source, variant).unwrap();
        let first = batched.forward_batch(&TOKENS[..3], 0).unwrap();
        assert_eq!(bits(&first), expected[2]);
        let second = batched.forward_batch(&TOKENS[3..7], 3).unwrap();
        assert_eq!(bits(&second), expected[6]);
        assert_eq!(bits(&batched.forward(TOKENS[7], 7).unwrap()), expected[7]);
        assert_ne!(expected[6], expected[7]);
    }
}

fn stock() -> Llama {
    Llama::from_source(&tiny_source(), Variant::Llama).unwrap()
}

#[test]
fn a_bad_token_rejects_the_batch_before_it_reaches_the_cache() {
    let mut llama = stock();
    match llama.forward_batch(&[1, 7, 2], 0) {
        Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "token out of range"),
        other => panic!("unexpected: {other:?}"),
    }
    let after = llama.forward_batch(&[1, 2], 0).unwrap();
    assert_eq!(
        bits(&after),
        bits(&stock().forward_batch(&[1, 2], 0).unwrap())
    );
}

#[test]
fn a_batch_past_the_context_is_context_full_before_it_reaches_the_cache() {
    let mut llama = stock();
    llama.forward_batch(&[4; 10], 0).unwrap();
    for start in [10, usize::MAX] {
        match llama.forward_batch(&[4; 7], start) {
            Err(ModelError::ContextFull { limit }) => assert_eq!(limit, 16),
            other => panic!("unexpected: {other:?}"),
        }
    }
    assert_eq!(llama.forward_batch(&[4; 6], 10).unwrap().len(), 7);
}

#[test]
fn a_batch_at_the_wrong_start_is_cache_position() {
    let mut llama = stock();
    llama.forward_batch(&[4, 5], 0).unwrap();
    match llama.forward_batch(&[4, 5], 3) {
        Err(ModelError::CachePosition { expected, got }) => assert_eq!((expected, got), (2, 3)),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn a_batch_of_no_tokens_is_invalid_config() {
    match stock().forward_batch(&[], 0) {
        Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "no tokens to forward"),
        other => panic!("unexpected: {other:?}"),
    }
}
