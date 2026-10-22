// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Metadata;
use crate::error::GgufError;
use crate::gguf::GGUFValue;

fn metadata(entries: &[(&str, GGUFValue)]) -> Metadata {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

fn with_arch(key: &str, value: GGUFValue) -> Metadata {
    metadata(&[
        ("general.architecture", GGUFValue::String("llama".into())),
        (key, value),
    ])
}

#[test]
fn get_finds_a_key() {
    let meta = with_arch("llama.block_count", GGUFValue::Uint32(2));
    assert_eq!(meta.get("llama.block_count"), Some(&GGUFValue::Uint32(2)));
    assert_eq!(meta.len(), 2);
    assert!(!meta.is_empty());
}

#[test]
fn get_misses() {
    let meta = Metadata::default();
    assert_eq!(meta.get("nope"), None);
    assert!(meta.is_empty());
}

#[test]
fn architecture_requires_a_string() {
    let meta = metadata(&[("general.architecture", GGUFValue::Uint32(1))]);
    assert_eq!(meta.architecture(), None);
    let meta = metadata(&[("general.architecture", GGUFValue::String("llama".into()))]);
    assert_eq!(meta.architecture(), Some("llama"));
}

#[test]
fn each_config_accessor_reads_its_key() {
    let meta = metadata(&[
        ("general.architecture", GGUFValue::String("llama".into())),
        ("llama.context_length", GGUFValue::Uint32(1)),
        ("llama.embedding_length", GGUFValue::Uint32(2)),
        ("llama.block_count", GGUFValue::Uint32(3)),
        ("llama.attention.head_count", GGUFValue::Uint32(4)),
        ("llama.attention.head_count_kv", GGUFValue::Uint32(5)),
        ("llama.feed_forward_length", GGUFValue::Uint32(6)),
        ("llama.rope.freq_base", GGUFValue::Float32(7.5)),
        ("llama.attn_norm_epsilon", GGUFValue::Float32(0.25)),
    ]);
    assert_eq!(meta.context_length(), Some(1));
    assert_eq!(meta.embedding_length(), Some(2));
    assert_eq!(meta.block_count(), Some(3));
    assert_eq!(meta.head_count(), Some(4));
    assert_eq!(meta.head_count_kv(), Some(5));
    assert_eq!(meta.feed_forward_length(), Some(6));
    assert_eq!(meta.rope_freq_base(), Some(7.5));
    assert_eq!(meta.rms_norm_eps(), Some(0.25));
}

#[test]
fn unsigned_helper_accepts_integer_widths() {
    let read = |value| with_arch("llama.block_count", value).block_count();
    assert_eq!(read(GGUFValue::Uint64(u64::MAX)), Some(u64::MAX));
    assert_eq!(read(GGUFValue::Uint32(4)), Some(4));
    assert_eq!(read(GGUFValue::Int32(5)), Some(5));
    assert_eq!(read(GGUFValue::Int64(6)), Some(6));
}

#[test]
fn unsigned_helper_rejects_negative_and_other_types() {
    let read = |value| with_arch("llama.block_count", value).block_count();
    assert_eq!(read(GGUFValue::Int32(-1)), None);
    assert_eq!(read(GGUFValue::Int64(-1)), None);
    assert_eq!(read(GGUFValue::String("3".into())), None);
}

#[test]
fn float_helper_narrows_float64_and_rejects_others() {
    let read = |value| with_arch("llama.rope.freq_base", value).rope_freq_base();
    assert_eq!(read(GGUFValue::Float64(0.5)), Some(0.5));
    assert_eq!(read(GGUFValue::Uint32(1)), None);
}

#[test]
fn accessors_need_an_architecture() {
    let meta = metadata(&[("llama.block_count", GGUFValue::Uint32(2))]);
    assert_eq!(meta.block_count(), None);
    assert_eq!(meta.rope_freq_base(), None);
}

#[test]
fn alignment_defaults_to_32() {
    assert_eq!(Metadata::default().alignment().unwrap(), 32);
}

#[test]
fn alignment_reads_general_alignment() {
    let meta = metadata(&[("general.alignment", GGUFValue::Uint32(64))]);
    assert_eq!(meta.alignment().unwrap(), 64);
}

#[test]
fn zero_alignment_is_rejected() {
    let meta = metadata(&[("general.alignment", GGUFValue::Uint32(0))]);
    assert!(matches!(
        meta.alignment(),
        Err(GgufError::InvalidAlignment(GGUFValue::Uint32(0)))
    ));
}

#[test]
fn non_power_of_two_alignment_is_rejected() {
    let meta = metadata(&[("general.alignment", GGUFValue::Uint32(48))]);
    assert!(matches!(
        meta.alignment(),
        Err(GgufError::InvalidAlignment(GGUFValue::Uint32(48)))
    ));
}

#[test]
fn wrongly_typed_alignment_is_rejected() {
    let meta = metadata(&[("general.alignment", GGUFValue::Uint64(32))]);
    assert!(matches!(
        meta.alignment(),
        Err(GgufError::InvalidAlignment(GGUFValue::Uint64(32)))
    ));
}
