// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::ModelError;

use super::*;

fn config() -> DecoderConfig {
    DecoderConfig {
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        head_count: 2,
        head_count_kv: 1,
        head_dim: 2,
        context_length: 16,
        vocab_size: 7,
        rope_freq_base: 10_000.0,
        rms_norm_eps: 1e-5,
        rope: RopeStyle::Adjacent,
        activation: Activation::Silu,
    }
}

fn invalid(config: &DecoderConfig, expected: &'static str) {
    let result = config.validate();
    assert!(
        matches!(&result, Err(ModelError::InvalidConfig { reason }) if *reason == expected),
        "unexpected: {result:?}"
    );
}

/// `config()` with `head_count`, `head_count_kv` and `head_dim` replaced.
fn heads(head_count: usize, head_count_kv: usize, head_dim: usize) -> DecoderConfig {
    DecoderConfig {
        head_count,
        head_count_kv,
        head_dim,
        ..config()
    }
}

#[test]
fn kv_dim_is_kv_heads_times_head_dim() {
    assert_eq!(config().kv_dim().unwrap(), 2);
}

#[test]
fn kv_dim_reports_an_overflow() {
    let mut config = config();
    config.head_count_kv = usize::MAX;
    match config.kv_dim() {
        Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, DIMENSIONS_TOO_LARGE),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn validate_accepts_a_consistent_config() {
    assert!(config().validate().is_ok());
    assert!(heads(1, 1, 4).validate().is_ok());
}

#[test]
fn validate_rejects_zero_heads() {
    invalid(&heads(0, 1, 2), "head count must be non-zero");
}

#[test]
fn validate_rejects_kv_heads_that_are_zero_or_do_not_divide_the_heads() {
    let reason = "kv head count must be non-zero and divide the head count";
    invalid(&heads(2, 0, 2), reason);
    invalid(&heads(2, 3, 2), reason);
}

#[test]
fn validate_rejects_an_odd_or_zero_head_dim() {
    let reason = "head dimension must be even and non-zero";
    invalid(&heads(2, 1, 0), reason);
    invalid(&heads(4, 1, 1), reason);
}

#[test]
fn validate_rejects_an_embedding_other_than_heads_times_head_dim() {
    let reason = "embedding length must equal the head count times the head dimension";
    invalid(&heads(1, 1, 2), reason);
    invalid(&heads(1 << 40, 1, 1 << 40), reason);
}
