// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Model, ModelError};

use super::Decoder;
use super::batch_tests::stock;
use crate::config::DecoderConfig;
use crate::rope::RopeStyle;
use crate::test_support::{TINY_VOCAB, patterned_block, tiny_config, tiny_weights};

#[test]
fn vocab_size_and_context_length_come_from_the_config() {
    let decoder = stock();
    assert_eq!(decoder.vocab_size(), TINY_VOCAB);
    assert_eq!(decoder.context_length(), 16);
}

#[test]
fn forward_out_of_order_is_cache_position() {
    let mut decoder = stock();
    match decoder.forward(1, 1) {
        Err(ModelError::CachePosition { expected, got }) => assert_eq!((expected, got), (0, 1)),
        other => panic!("unexpected: {other:?}"),
    }
    decoder.forward(1, 0).unwrap();
    match decoder.forward(1, 0) {
        Err(ModelError::CachePosition { expected, got }) => assert_eq!((expected, got), (1, 0)),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn forward_past_the_context_is_context_full() {
    let mut decoder = stock();
    for position in 0..16 {
        decoder.forward(4, position).unwrap();
    }
    match decoder.forward(4, 16) {
        Err(ModelError::ContextFull { limit }) => assert_eq!(limit, 16),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn reset_restarts_at_position_zero() {
    let mut decoder = stock();
    let first = decoder.forward(5, 0).unwrap();
    let second = decoder.forward(6, 1).unwrap();
    decoder.reset();
    assert_eq!(decoder.forward(5, 0).unwrap(), first);
    assert_eq!(decoder.forward(6, 1).unwrap(), second);
}

#[test]
fn token_out_of_range_is_invalid_config() {
    let mut decoder = stock();
    for token in [7, u32::MAX] {
        match decoder.forward(token, 0) {
            Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "token out of range"),
            other => panic!("unexpected: {other:?}"),
        }
    }
    assert_eq!(decoder.forward(6, 0).unwrap().len(), TINY_VOCAB);
}

/// `edit` applied to the tiny config, then a decoder over the stock weights.
fn built(edit: impl FnOnce(&mut DecoderConfig)) -> rusticide_model::Result<Decoder> {
    let stock = tiny_config(RopeStyle::Adjacent);
    let weights = tiny_weights(&stock, vec![patterned_block(&stock, 1)]);
    let mut config = stock;
    edit(&mut config);
    Decoder::new(config, weights)
}

type Edit = fn(&mut DecoderConfig);

#[test]
fn new_rejects_a_config_attention_cannot_run_instead_of_panicking() {
    let kv = "kv head count must be non-zero and divide the head count";
    let cases: [(Edit, &str); 5] = [
        (|config| config.head_count_kv = 0, kv),
        (|config| config.head_count_kv = 3, kv),
        (
            |config| config.head_dim = 0,
            "head dimension must be even and non-zero",
        ),
        (
            |config| config.feed_forward_length = 0,
            "block count, feed-forward length and context length must be non-zero",
        ),
        (
            |config| config.embedding_length = 6,
            "embedding length must be divisible by the head count: it must equal the head count times the head dimension",
        ),
    ];
    for (edit, expected) in cases {
        match built(edit) {
            Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, expected),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
