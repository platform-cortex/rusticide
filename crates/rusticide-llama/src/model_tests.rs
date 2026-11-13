// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Loader, ModelError};

use super::{load, load_qwen2};
use crate::testing::{TINY_VOCAB, TinySource, tiny_source};

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

#[test]
fn load_qwen2_returns_a_boxed_model() {
    let mut model = load_qwen2(&TinySource::qwen2().build()).unwrap();
    assert_eq!(model.vocab_size(), 258);
    assert_eq!(model.context_length(), 16);
    assert_eq!(model.forward(1, 0).unwrap().len(), 258);
}
