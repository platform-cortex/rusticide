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

#[test]
fn kv_dim_is_kv_heads_times_head_dim() {
    assert_eq!(config().kv_dim().unwrap(), 2);
}

#[test]
fn kv_dim_reports_an_overflow() {
    let mut config = config();
    config.head_count_kv = usize::MAX;
    match config.kv_dim() {
        Err(ModelError::InvalidConfig { reason }) => {
            assert_eq!(reason, "model dimensions are too large");
        }
        other => panic!("unexpected: {other:?}"),
    }
}
