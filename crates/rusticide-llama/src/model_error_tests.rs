// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Loader, Model, ModelError};

use super::{Llama, load};
use crate::testing::{TINY_VOCAB, TinySource, tiny_source};

fn stock() -> Llama {
    Llama::from_source(&tiny_source()).unwrap()
}

#[test]
fn forward_out_of_order_is_cache_position() {
    let mut llama = stock();
    match llama.forward(1, 1) {
        Err(ModelError::CachePosition { expected, got }) => assert_eq!((expected, got), (0, 1)),
        other => panic!("unexpected: {other:?}"),
    }
    llama.forward(1, 0).unwrap();
    match llama.forward(1, 0) {
        Err(ModelError::CachePosition { expected, got }) => assert_eq!((expected, got), (1, 0)),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn forward_past_the_context_is_context_full() {
    let mut llama = stock();
    for position in 0..16 {
        llama.forward(4, position).unwrap();
    }
    match llama.forward(4, 16) {
        Err(ModelError::ContextFull { limit }) => assert_eq!(limit, 16),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn reset_restarts_at_position_zero() {
    let mut llama = stock();
    let first = llama.forward(5, 0).unwrap();
    let second = llama.forward(6, 1).unwrap();
    llama.reset();
    assert_eq!(llama.forward(5, 0).unwrap(), first);
    assert_eq!(llama.forward(6, 1).unwrap(), second);
}

#[test]
fn token_out_of_range_is_invalid_config() {
    let mut llama = stock();
    for token in [7, u32::MAX] {
        match llama.forward(token, 0) {
            Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "token out of range"),
            other => panic!("unexpected: {other:?}"),
        }
    }
    assert_eq!(llama.forward(6, 0).unwrap().len(), TINY_VOCAB);
}

#[test]
fn load_returns_a_boxed_model() {
    let loader: Loader = load;
    let mut model = loader(&tiny_source()).unwrap();
    assert_eq!(model.vocab_size(), TINY_VOCAB);
    assert_eq!(model.context_length(), 16);
    assert_eq!(model.forward(1, 0).unwrap().len(), TINY_VOCAB);
}

#[test]
fn load_reports_a_missing_tensor() {
    let source = TinySource::new()
        .without_tensor("blk.0.attn_q.weight")
        .build();
    match load(&source) {
        Err(ModelError::MissingTensor { name }) => assert_eq!(name, "blk.0.attn_q.weight"),
        other => panic!("unexpected: {:?}", other.map(|_| ())),
    }
}
