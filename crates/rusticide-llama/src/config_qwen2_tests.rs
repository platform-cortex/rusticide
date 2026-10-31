// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, ModelError};

use super::LlamaConfig;
use crate::Variant;
use crate::testing::TinySource;

const KEYS: [(&str, u32); 6] = [
    ("context_length", 16),
    ("embedding_length", 4),
    ("block_count", 1),
    ("feed_forward_length", 8),
    ("attention.head_count", 2),
    ("attention.head_count_kv", 1),
];

fn qwen2_source(dropped: Option<&str>) -> InMemorySource {
    let mut builder = TinySource::new().with_metadata(
        "general.architecture",
        GGUFValue::String("qwen2".to_owned()),
    );
    for (suffix, value) in KEYS {
        builder = builder.without_metadata(&format!("llama.{suffix}"));
        if dropped != Some(suffix) {
            builder = builder.with_metadata(&format!("qwen2.{suffix}"), GGUFValue::Uint32(value));
        }
    }
    builder.build()
}

#[test]
fn qwen2_config_reads_qwen2_keys() {
    let config = LlamaConfig::from_source(&qwen2_source(None), Variant::Qwen2).unwrap();
    assert_eq!(config.variant, Variant::Qwen2);
    assert_eq!(config.embedding_length, 4);
    assert_eq!((config.head_count, config.head_count_kv), (2, 1));
    assert_eq!(config.context_length, 16);
}

#[test]
fn qwen2_config_names_qwen2_keys_when_missing() {
    let source = qwen2_source(Some("context_length"));
    match LlamaConfig::from_source(&source, Variant::Qwen2) {
        Err(ModelError::MissingMetadata { key }) => assert_eq!(key, "qwen2.context_length"),
        other => panic!("unexpected: {other:?}"),
    }
}

fn with_rope_dimension(dimension: u32) -> InMemorySource {
    let mut builder = TinySource::new()
        .with_metadata(
            "general.architecture",
            GGUFValue::String("qwen2".to_owned()),
        )
        .with_metadata("qwen2.rope.dimension_count", GGUFValue::Uint32(dimension));
    for (suffix, value) in KEYS {
        builder = builder
            .without_metadata(&format!("llama.{suffix}"))
            .with_metadata(&format!("qwen2.{suffix}"), GGUFValue::Uint32(value));
    }
    builder.build()
}

#[test]
fn qwen2_config_rejects_a_mismatched_qwen2_rope_dimension() {
    let result = LlamaConfig::from_source(&with_rope_dimension(4), Variant::Qwen2);
    assert!(
        matches!(&result, Err(ModelError::InvalidConfig { reason })
            if *reason == "rope dimension count must equal the head dimension"),
        "unexpected: {result:?}"
    );
}

#[test]
fn qwen2_config_accepts_the_matching_qwen2_rope_dimension() {
    assert!(LlamaConfig::from_source(&with_rope_dimension(2), Variant::Qwen2).is_ok());
}
