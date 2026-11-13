// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tiny model widened to 128, the `I2_S` block, with every projection
//! held as ternary `I2_S` bytes. Its logits must track the same model with
//! those weights decoded to f32: the only difference is the 8-bit rounding
//! of the activations on the ternary path.

use half::f16;
use rusticide_core::Tensor;
use rusticide_io::GGUFValue;
use rusticide_model::InMemorySource;
use rusticide_quantize::{BlockFormat, block_format};

use crate::load;
use crate::testing::{ATTN_SUB_NORM, FFN_SUB_NORM, VOCAB, ramp, tiny};

const WIDTH: usize = 128;
const KV_DIM: usize = 64;
const I2_S: u32 = 36;

const PROJECTIONS: [(&str, [usize; 2]); 7] = [
    ("blk.0.attn_q.weight", [WIDTH, WIDTH]),
    ("blk.0.attn_k.weight", [KV_DIM, WIDTH]),
    ("blk.0.attn_v.weight", [KV_DIM, WIDTH]),
    ("blk.0.attn_output.weight", [WIDTH, WIDTH]),
    ("blk.0.ffn_gate.weight", [WIDTH, WIDTH]),
    ("blk.0.ffn_up.weight", [WIDTH, WIDTH]),
    ("blk.0.ffn_down.weight", [WIDTH, WIDTH]),
];

fn i2_s() -> &'static BlockFormat {
    block_format(I2_S).unwrap()
}

/// Codes `(7 * i + offset) % 3` packed as `I2_S`, then the trailer: the
/// scale `0.05 + 0.01 * offset` and 28 unused bytes.
fn ternary(rows: usize, offset: usize) -> Vec<u8> {
    let code = |index: usize| u8::try_from((7 * index + offset) % 3).unwrap();
    let mut data: Vec<u8> = (0..rows * WIDTH / 128)
        .flat_map(|block| {
            let first = block * 128;
            (0..32).map(move |k| {
                let at = first + k;
                code(at) << 6 | code(at + 32) << 4 | code(at + 64) << 2 | code(at + 96)
            })
        })
        .collect();
    let scale = 0.01_f32.mul_add(f32::from(u8::try_from(offset).unwrap()), 0.05);
    data.extend(scale.to_le_bytes());
    data.extend([0xA5; 28]);
    data
}

fn embedding() -> Tensor {
    let data: Vec<f16> = (0..VOCAB * WIDTH)
        .map(|index| f16::from_f32(0.1 * (f32::from(u8::try_from(index % 9).unwrap()) - 4.0)))
        .collect();
    Tensor::from_slice(&data, &[VOCAB, WIDTH]).unwrap()
}

/// The 128-wide tiny model with its projections as `ternary` or decoded.
fn wide(quantized: bool) -> InMemorySource {
    let count = |n: u32| GGUFValue::Uint32(n);
    let mut source = tiny()
        .with_metadata("bitnet-b1.58.embedding_length", count(128))
        .with_metadata("bitnet-b1.58.feed_forward_length", count(128))
        .with_tensor("token_embd.weight", embedding())
        .with_tensor("blk.0.attn_norm.weight", ramp(WIDTH))
        .with_tensor("blk.0.ffn_norm.weight", ramp(WIDTH))
        .with_tensor("output_norm.weight", ramp(WIDTH))
        .with_tensor(ATTN_SUB_NORM, ramp(WIDTH))
        .with_tensor(FFN_SUB_NORM, ramp(WIDTH))
        .build();
    for (offset, (name, shape)) in PROJECTIONS.into_iter().enumerate() {
        let bytes = ternary(shape[0], offset);
        source = if quantized {
            source.with_quantized(name, i2_s(), bytes, &shape).unwrap()
        } else {
            source.with_tensor(name, i2_s().dequantize(&bytes, &shape).unwrap())
        };
    }
    source
}

#[test]
fn i2_s_projections_load_ternary_and_track_their_f32_decoding() {
    let mut ternary = load(&wide(true)).unwrap();
    let mut dense = load(&wide(false)).unwrap();
    for token in 0..u32::try_from(VOCAB).unwrap() {
        let got = ternary.forward(token, 0).unwrap();
        let want = dense.forward(token, 0).unwrap();
        ternary.reset();
        dense.reset();
        assert!(got.iter().all(|logit| logit.is_finite()), "{got:?}");
        assert_ne!(got, want, "the ternary path rounds its activations");
        let span = want.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 0.02 * span, "{got:?} vs {want:?}");
        }
    }
}
