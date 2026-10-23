// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, ModelError};

use super::LlamaConfig;
use crate::testing::{TINY_VOCAB, TinySource, tiny_source};

fn invalid(source: &InMemorySource, expected: &'static str) {
    let result = LlamaConfig::from_source(source);
    assert!(
        matches!(&result, Err(ModelError::InvalidConfig { reason }) if *reason == expected),
        "unexpected: {result:?}"
    );
}

fn with(key: &str, value: u32) -> InMemorySource {
    TinySource::new()
        .with_metadata(key, GGUFValue::Uint32(value))
        .build()
}

#[test]
fn config_reads_tiny_source() {
    let config = LlamaConfig::from_source(&tiny_source()).unwrap();
    assert_eq!(config.vocab_size, TINY_VOCAB);
    assert_eq!(config.embedding_length, 4);
    assert_eq!(config.block_count, 1);
    assert_eq!(config.feed_forward_length, 8);
    assert_eq!((config.head_count, config.head_count_kv), (2, 1));
    assert_eq!(config.head_dim, 2);
    assert_eq!(config.context_length, 16);
}

#[test]
fn config_defaults_rope_base_and_eps() {
    let config = LlamaConfig::from_source(&tiny_source()).unwrap();
    assert!((config.rope_freq_base - 10000.0).abs() < f32::EPSILON);
    assert!((config.rms_norm_eps - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn config_reads_explicit_rope_base_eps_and_kv_heads() {
    let source = TinySource::new()
        .with_metadata("llama.rope.freq_base", GGUFValue::Float32(500.0))
        .with_metadata(
            "llama.attention.layer_norm_rms_epsilon",
            GGUFValue::Float32(1e-6),
        )
        .without_metadata("llama.attention.head_count_kv")
        .build();
    let config = LlamaConfig::from_source(&source).unwrap();
    assert!((config.rope_freq_base - 500.0).abs() < f32::EPSILON);
    assert!((config.rms_norm_eps - 1e-6).abs() < f32::EPSILON);
    assert_eq!(config.head_count_kv, 2);
}

#[test]
fn config_requires_context_length() {
    let source = TinySource::new()
        .without_metadata("llama.context_length")
        .build();
    match LlamaConfig::from_source(&source) {
        Err(ModelError::MissingMetadata { key }) => assert_eq!(key, "llama.context_length"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn config_rejects_zero_heads() {
    invalid(
        &with("llama.attention.head_count", 0),
        "head count must be non-zero",
    );
}

#[test]
fn config_rejects_embedding_not_divisible_by_heads() {
    invalid(
        &with("llama.embedding_length", 5),
        "embedding length must be divisible by the head count",
    );
}

#[test]
fn config_rejects_kv_heads_not_dividing_heads() {
    let reason = "kv head count must be non-zero and divide the head count";
    invalid(&with("llama.attention.head_count_kv", 3), reason);
    invalid(&with("llama.attention.head_count_kv", 0), reason);
}

#[test]
fn config_rejects_odd_or_zero_head_dim() {
    let reason = "head dimension must be even and non-zero";
    invalid(&with("llama.embedding_length", 6), reason);
    invalid(&with("llama.embedding_length", 0), reason);
}

#[test]
fn config_rejects_zero_block_feed_forward_or_context() {
    let reason = "block count, feed-forward length and context length must be non-zero";
    invalid(&with("llama.block_count", 0), reason);
    invalid(&with("llama.feed_forward_length", 0), reason);
    invalid(&with("llama.context_length", 0), reason);
}

#[test]
fn config_rejects_a_mismatched_rope_dimension() {
    let reason = "rope dimension count must equal the head dimension";
    invalid(&with("llama.rope.dimension_count", 4), reason);
    let wide = TinySource::new()
        .with_metadata("llama.rope.dimension_count", GGUFValue::Uint64(4))
        .build();
    invalid(&wide, reason);
}

#[test]
fn config_accepts_a_matching_rope_dimension() {
    assert!(LlamaConfig::from_source(&with("llama.rope.dimension_count", 2)).is_ok());
    let wide = TinySource::new()
        .with_metadata("llama.rope.dimension_count", GGUFValue::Uint64(2))
        .build();
    assert!(LlamaConfig::from_source(&wide).is_ok());
}

#[test]
fn config_rejects_a_non_integer_rope_dimension() {
    let source = TinySource::new()
        .with_metadata("llama.rope.dimension_count", GGUFValue::Bool(true))
        .build();
    invalid(&source, "rope dimension count must be an unsigned integer");
}

#[test]
fn config_rejects_values_that_do_not_fit_usize_arithmetic() {
    let huge = |key: &str| {
        TinySource::new()
            .with_metadata(key, GGUFValue::Uint64(1 << 63))
            .build()
    };
    invalid(
        &huge("llama.embedding_length"),
        "model dimensions are too large",
    );
    let wide = TinySource::new()
        .with_metadata("llama.embedding_length", GGUFValue::Uint64(1 << 62))
        .build();
    invalid(&wide, "model dimensions are too large");
}

fn wide(entries: &[(&str, u64)]) -> InMemorySource {
    entries
        .iter()
        .fold(TinySource::new(), |source, (key, value)| {
            source.with_metadata(key, GGUFValue::Uint64(*value))
        })
        .build()
}

#[test]
fn config_rejects_an_embedding_square_that_overflows() {
    let source = wide(&[
        ("llama.embedding_length", 1 << 32),
        ("llama.attention.head_count", 1 << 16),
        ("llama.attention.head_count_kv", 1),
    ]);
    invalid(&source, "model dimensions are too large");
}

#[test]
fn config_rejects_an_embedding_by_feed_forward_product_that_overflows() {
    let source = wide(&[("llama.feed_forward_length", 1 << 63)]);
    invalid(&source, "model dimensions are too large");
}

#[test]
fn config_rejects_a_mis_shaped_token_embedding() {
    let reason = "token_embd.weight must have shape [vocab, embedding length]";
    let flat = Tensor::from_slice(&[1.0_f32; 4], &[4]).unwrap();
    invalid(
        &TinySource::new()
            .with_tensor("token_embd.weight", flat)
            .build(),
        reason,
    );
    let narrow = Tensor::from_slice(&[1.0_f32; 6], &[2, 3]).unwrap();
    invalid(
        &TinySource::new()
            .with_tensor("token_embd.weight", narrow)
            .build(),
        reason,
    );
    let empty = Tensor::from_slice(&[] as &[f32], &[0, 4]).unwrap();
    invalid(
        &TinySource::new()
            .with_tensor("token_embd.weight", empty)
            .build(),
        reason,
    );
}

#[test]
fn config_requires_the_token_embedding() {
    let source = TinySource::new()
        .without_tensor("token_embd.weight")
        .build();
    match LlamaConfig::from_source(&source) {
        Err(ModelError::MissingTensor { name }) => assert_eq!(name, "token_embd.weight"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn kv_dim_is_kv_heads_times_head_dim() {
    let config = LlamaConfig::from_source(&tiny_source()).unwrap();
    assert_eq!(config.kv_dim().unwrap(), 2);
}

#[test]
fn kv_dim_reports_an_overflow() {
    let mut config = LlamaConfig::from_source(&tiny_source()).unwrap();
    config.head_count_kv = usize::MAX;
    match config.kv_dim() {
        Err(ModelError::InvalidConfig { reason }) => {
            assert_eq!(reason, "model dimensions are too large");
        }
        other => panic!("unexpected: {other:?}"),
    }
}
