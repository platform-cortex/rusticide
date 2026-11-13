// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Model, ModelError};

use super::Decoder;
use crate::config::DecoderConfig;
use crate::rope::RopeStyle;
use crate::test_support::{TINY_VOCAB, patterned, patterned_block, tiny_config, tiny_weights};
use crate::weights::BlockWeights;

const TOKENS: [u32; 8] = [1, 5, 6, 4, 3, 6, 2, 5];

fn bias(len: usize, offset: usize) -> Vec<f32> {
    patterned(&[len], offset).to_vec::<f32>().unwrap()
}

/// Block `offset`, with query, key and value biases when `biases`.
fn block(config: &DecoderConfig, offset: usize, biases: bool) -> BlockWeights {
    let mut block = patterned_block(config, offset);
    if biases {
        let kv_dim = config.kv_dim().unwrap();
        block.attn_q_bias = Some(bias(config.embedding_length, offset + 1));
        block.attn_k_bias = Some(bias(kv_dim, offset + 2));
        block.attn_v_bias = Some(bias(kv_dim, offset + 3));
    }
    block
}

/// The tiny decoder with a second block, so that a token's output feeds the
/// keys and values later tokens attend to.
fn two_blocks(rope: RopeStyle, biases: bool) -> Decoder {
    let config = DecoderConfig {
        block_count: 2,
        ..tiny_config(rope)
    };
    let mut second = block(&config, 3, biases);
    second.attn_norm = vec![1.5, 0.5, 1.0, 2.0];
    second.ffn_norm = vec![1.5, 0.5, 1.0, 2.0];
    let blocks = vec![block(&config, 1, biases), second];
    let weights = tiny_weights(&config, blocks);
    Decoder::new(config, weights).unwrap()
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn batches_match_one_token_at_a_time_bit_for_bit() {
    for (rope, biases) in [(RopeStyle::Adjacent, false), (RopeStyle::HalfSplit, true)] {
        let mut single = two_blocks(rope, biases);
        let expected: Vec<Vec<u32>> = (0..)
            .zip(TOKENS)
            .map(|(position, token)| bits(&single.forward(token, position).unwrap()))
            .collect();
        let mut batched = two_blocks(rope, biases);
        let first = batched.forward_batch(&TOKENS[..3], 0).unwrap();
        assert_eq!(bits(&first), expected[2]);
        let second = batched.forward_batch(&TOKENS[3..7], 3).unwrap();
        assert_eq!(bits(&second), expected[6]);
        assert_eq!(bits(&batched.forward(TOKENS[7], 7).unwrap()), expected[7]);
        assert_ne!(expected[6], expected[7]);
    }
}

pub(super) fn stock() -> Decoder {
    let config = tiny_config(RopeStyle::Adjacent);
    let weights = tiny_weights(&config, vec![patterned_block(&config, 1)]);
    Decoder::new(config, weights).unwrap()
}

#[test]
fn a_bad_token_rejects_the_batch_before_it_reaches_the_cache() {
    let mut decoder = stock();
    match decoder.forward_batch(&[1, 7, 2], 0) {
        Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "token out of range"),
        other => panic!("unexpected: {other:?}"),
    }
    let after = decoder.forward_batch(&[1, 2], 0).unwrap();
    assert_eq!(
        bits(&after),
        bits(&stock().forward_batch(&[1, 2], 0).unwrap())
    );
}

#[test]
fn a_batch_past_the_context_is_context_full_before_it_reaches_the_cache() {
    let mut decoder = stock();
    decoder.forward_batch(&[4; 10], 0).unwrap();
    for start in [10, usize::MAX] {
        match decoder.forward_batch(&[4; 7], start) {
            Err(ModelError::ContextFull { limit }) => assert_eq!(limit, 16),
            other => panic!("unexpected: {other:?}"),
        }
    }
    assert_eq!(
        decoder.forward_batch(&[4; 6], 10).unwrap().len(),
        TINY_VOCAB
    );
}

#[test]
fn a_batch_at_the_wrong_start_is_cache_position() {
    let mut decoder = stock();
    decoder.forward_batch(&[4, 5], 0).unwrap();
    match decoder.forward_batch(&[4, 5], 3) {
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
