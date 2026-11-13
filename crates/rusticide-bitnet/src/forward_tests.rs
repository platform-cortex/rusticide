// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One token through the tiny model against a scalar reference of the
//! `BitNet` block. At position 0 a token attends only to itself, so each
//! head's output is its kv head's value and `RoPE` is the identity.

use half::f16;
use rusticide_core::DType;
use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, ModelSource};

use crate::load;
use crate::testing::tiny;

/// Far from the 1e-5 default, so a norm reading any other epsilon fails.
const EPS: f32 = 0.5;
const WIDTH: usize = 4;
const LENGTH: usize = 8;

fn values(source: &InMemorySource, name: &str) -> Vec<f32> {
    let tensor = source.tensor(name).unwrap();
    if tensor.dtype() == DType::F16 {
        tensor
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect()
    } else {
        tensor.to_vec::<f32>().unwrap()
    }
}

fn rms_norm(row: &[f32], weight: &[f32]) -> Vec<f32> {
    let count = f32::from(u8::try_from(row.len()).unwrap());
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / count + EPS).sqrt();
    row.iter().zip(weight).map(|(v, w)| v / rms * w).collect()
}

fn matvec(matrix: &[f32], cols: usize, input: &[f32]) -> Vec<f32> {
    matrix
        .chunks_exact(cols)
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect()
}

fn add(hidden: &mut [f32], delta: &[f32]) {
    hidden.iter_mut().zip(delta).for_each(|(h, d)| *h += d);
}

fn reference(source: &InMemorySource, token: usize) -> Vec<f32> {
    let weight = |part: &str| values(source, &format!("blk.0.{part}.weight"));
    let embedding = values(source, "token_embd.weight");
    let mut hidden = embedding[token * WIDTH..][..WIDTH].to_vec();
    let normed = rms_norm(&hidden, &weight("attn_norm"));
    let value = matvec(&weight("attn_v"), WIDTH, &normed);
    let mixed = value.repeat(2);
    let mixed = rms_norm(&mixed, &weight("attn_sub_norm"));
    add(&mut hidden, &matvec(&weight("attn_output"), WIDTH, &mixed));
    let normed = rms_norm(&hidden, &weight("ffn_norm"));
    let gate = matvec(&weight("ffn_gate"), WIDTH, &normed);
    let up = matvec(&weight("ffn_up"), WIDTH, &normed);
    let gated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g.max(0.0).powi(2) * u)
        .collect();
    let gated = rms_norm(&gated, &weight("ffn_sub_norm"));
    add(&mut hidden, &matvec(&weight("ffn_down"), LENGTH, &gated));
    let output = rms_norm(&hidden, &values(source, "output_norm.weight"));
    matvec(&embedding, WIDTH, &output)
}

#[test]
fn a_token_runs_through_relu_squared_sub_norms_and_the_tied_output() {
    let source = tiny()
        .with_metadata(
            "bitnet-b1.58.attention.layer_norm_rms_epsilon",
            GGUFValue::Float32(EPS),
        )
        .build();
    let mut model = load(&source).unwrap();
    for token in 0..crate::testing::VOCAB {
        let logits = model.forward(u32::try_from(token).unwrap(), 0).unwrap();
        model.reset();
        let expected = reference(&source, token);
        assert_eq!(logits.len(), expected.len());
        for (got, want) in logits.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "{logits:?} vs {expected:?}");
        }
    }
}
