// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, ModelError};
use rusticide_transformer::{Activation, DecoderConfig, RopeStyle};

use super::{ARCHITECTURE, config};
use crate::testing::{VOCAB, tiny};

fn invalid(source: &InMemorySource, expected: &'static str) {
    let result = config(source);
    assert!(
        matches!(&result, Err(ModelError::InvalidConfig { reason }) if *reason == expected),
        "unexpected: {result:?}"
    );
}

#[test]
fn the_architecture_is_bitnet_b1_58() {
    assert_eq!(ARCHITECTURE, "bitnet-b1.58");
}

#[test]
fn config_reads_the_bitnet_keys() {
    let source = tiny()
        .with_metadata("bitnet-b1.58.rope.freq_base", GGUFValue::Float32(500_000.0))
        .with_metadata(
            "bitnet-b1.58.attention.layer_norm_rms_epsilon",
            GGUFValue::Float32(1e-6),
        )
        .with_metadata("bitnet-b1.58.rope.dimension_count", GGUFValue::Uint32(2))
        .with_metadata("llama.embedding_length", GGUFValue::Uint32(8))
        .build();
    let expected = DecoderConfig {
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        head_count: 2,
        head_count_kv: 1,
        head_dim: 2,
        context_length: 16,
        vocab_size: VOCAB,
        rope_freq_base: 500_000.0,
        rms_norm_eps: 1e-6,
        rope: RopeStyle::HalfSplit,
        activation: Activation::ReluSquared,
    };
    assert_eq!(config(&source).unwrap(), expected);
}

#[test]
fn config_defaults_rope_base_and_eps() {
    let config = config(&tiny().build()).unwrap();
    assert!((config.rope_freq_base - 10_000.0).abs() < f32::EPSILON);
    assert!((config.rms_norm_eps - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn config_names_the_bitnet_key_when_missing() {
    let source = tiny()
        .without_metadata("bitnet-b1.58.context_length")
        .build();
    match config(&source) {
        Err(ModelError::MissingMetadata { key }) => {
            assert_eq!(key, "bitnet-b1.58.context_length");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn config_rejects_a_mismatched_rope_dimension() {
    let source = tiny()
        .with_metadata("bitnet-b1.58.rope.dimension_count", GGUFValue::Uint32(4))
        .build();
    invalid(
        &source,
        "rope dimension count must equal the head dimension",
    );
}

#[test]
fn config_rejects_zero_heads() {
    let source = tiny()
        .with_metadata("bitnet-b1.58.attention.head_count", GGUFValue::Uint32(0))
        .build();
    invalid(&source, "head count must be non-zero");
}
