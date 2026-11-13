// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tiny `bitnet-b1.58` model: the tiny llama of `rusticide-llama` under
//! `bitnet-b1.58.*` keys, with both sub-norms, a signed F16 token embedding
//! and no `output.weight`, so the output is tied to the embedding.

use half::f16;
use rusticide_core::Tensor;
use rusticide_llama::testing::{TINY_VOCAB, TinySource};

pub(crate) const VOCAB: usize = TINY_VOCAB;
pub(crate) const ATTN_SUB_NORM: &str = "blk.0.attn_sub_norm.weight";
pub(crate) const FFN_SUB_NORM: &str = "blk.0.ffn_sub_norm.weight";

/// `0.5 + 0.25 * i`: a sub-norm weight that is not uniform.
pub(crate) fn ramp(len: usize) -> Tensor {
    let data: Vec<f32> = (0..len)
        .map(|index| 0.25_f32.mul_add(f32::from(u8::try_from(index).unwrap()), 0.5))
        .collect();
    Tensor::from_slice(&data, &[len]).unwrap()
}

/// `0.05 * ((i % 5) - 2)` as F16: a signed embedding, so the squared `ReLU`
/// sees negative gates.
fn embedding() -> Tensor {
    let data: Vec<f16> = (0..VOCAB * 4)
        .map(|index| {
            let step = f32::from(u8::try_from(index % 5).unwrap()) - 2.0;
            f16::from_f32(0.05 * step)
        })
        .collect();
    Tensor::from_slice(&data, &[VOCAB, 4]).unwrap()
}

/// The stock tiny `bitnet-b1.58`: embedding 4, 2 heads over 1 kv head,
/// feed-forward 8, context 16 and [`VOCAB`] tokens.
pub(crate) fn tiny() -> TinySource {
    TinySource::new()
        .with_architecture("bitnet-b1.58")
        .without_tensor("output.weight")
        .with_tensor("token_embd.weight", embedding())
        .with_tensor(ATTN_SUB_NORM, ramp(4))
        .with_tensor(FFN_SUB_NORM, ramp(8))
}
