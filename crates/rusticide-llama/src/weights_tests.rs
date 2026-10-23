// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_model::{InMemorySource, ModelError};

use super::LlamaWeights;
use crate::config::LlamaConfig;
use crate::testing::{TINY_VOCAB, TinySource, tiny_source};

fn load(source: &InMemorySource) -> rusticide_model::Result<LlamaWeights> {
    let config = LlamaConfig::from_source(source)?;
    LlamaWeights::load(source, &config)
}

#[test]
fn weights_load_tiny_source() {
    let weights = load(&tiny_source()).unwrap();
    assert_eq!(weights.token_embd.rows(), TINY_VOCAB);
    assert_eq!(weights.token_embd.cols(), 4);
    assert_eq!(weights.blocks.len(), 1);
    let block = &weights.blocks[0];
    assert_eq!(block.attn_norm.len(), 4);
    assert_eq!((block.attn_q.rows(), block.attn_q.cols()), (4, 4));
    assert_eq!((block.attn_k.rows(), block.attn_k.cols()), (2, 4));
    assert_eq!((block.attn_v.rows(), block.attn_v.cols()), (2, 4));
    assert_eq!((block.attn_output.rows(), block.attn_output.cols()), (4, 4));
    assert_eq!(block.ffn_norm.len(), 4);
    assert_eq!((block.ffn_gate.rows(), block.ffn_gate.cols()), (8, 4));
    assert_eq!((block.ffn_up.rows(), block.ffn_up.cols()), (8, 4));
    assert_eq!((block.ffn_down.rows(), block.ffn_down.cols()), (4, 8));
    assert_eq!(weights.output_norm.len(), 4);
    let output = weights.output.as_ref().unwrap();
    assert_eq!((output.rows(), output.cols()), (TINY_VOCAB, 4));
}

#[test]
fn weights_reject_a_mis_shaped_projection() {
    let wrong = Tensor::from_slice(&[0.5_f32; 4], &[1, 4]).unwrap();
    let source = TinySource::new()
        .with_tensor("blk.0.attn_k.weight", wrong)
        .build();
    match load(&source) {
        Err(ModelError::ShapeMismatch {
            name,
            expected,
            got,
        }) => {
            assert_eq!(name, "blk.0.attn_k.weight");
            assert_eq!(expected, vec![2, 4]);
            assert_eq!(got, vec![1, 4]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn missing_output_weight_ties_to_token_embedding() {
    let source = TinySource::new().without_tensor("output.weight").build();
    assert!(load(&source).unwrap().output.is_none());
}

#[test]
fn weights_report_a_missing_block_tensor() {
    let source = TinySource::new()
        .without_tensor("blk.0.ffn_down.weight")
        .build();
    match load(&source) {
        Err(ModelError::MissingTensor { name }) => assert_eq!(name, "blk.0.ffn_down.weight"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn weights_reject_a_mis_shaped_output_norm() {
    let wrong = Tensor::from_slice(&[1.0_f32; 3], &[3]).unwrap();
    let source = TinySource::new()
        .with_tensor("output_norm.weight", wrong)
        .build();
    assert!(matches!(
        load(&source),
        Err(ModelError::ShapeMismatch { name, .. }) if name == "output_norm.weight"
    ));
}

#[test]
fn weights_load_a_huge_block_count_without_allocating_for_it() {
    let source = TinySource::new()
        .with_metadata(
            "llama.block_count",
            rusticide_io::GGUFValue::Uint64(1 << 40),
        )
        .build();
    match load(&source) {
        Err(ModelError::MissingTensor { name }) => assert_eq!(name, "blk.1.attn_norm.weight"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn weights_reject_a_config_whose_kv_dimension_overflows() {
    let mut config = LlamaConfig::from_source(&tiny_source()).unwrap();
    config.head_count_kv = usize::MAX;
    match LlamaWeights::load(&tiny_source(), &config) {
        Err(ModelError::InvalidConfig { reason }) => {
            assert_eq!(reason, "model dimensions are too large");
        }
        other => panic!("unexpected: {other:?}"),
    }
}
