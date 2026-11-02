// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod dot;
mod elementwise;
pub(crate) mod int_dot;
mod norm;
mod q8;
mod rope;

use crate::adapter::{Adapter, F32Kernels};
use crate::q8_input::Q8Kind;
use crate::quant_kernel::QuantKernel;

#[expect(
    clippy::cast_precision_loss,
    reason = "counts beyond 2^24 lose low bits, an accepted f32 rounding"
)]
const fn lossy(count: usize) -> f32 {
    count as f32
}

static QUANTIZED: [QuantKernel; 5] = [
    QuantKernel {
        type_id: 2,
        activation: Q8Kind::Block,
        dot: int_dot::q4_0,
        dot_rows: None,
        group_rows: 1,
        prepare: None,
        read_row: None,
        dot_group: None,
    },
    QuantKernel {
        type_id: 6,
        activation: Q8Kind::Block,
        dot: int_dot::q5_0,
        dot_rows: Some(int_dot::q5_0_rows),
        group_rows: 1,
        prepare: None,
        read_row: None,
        dot_group: None,
    },
    QuantKernel {
        type_id: 12,
        activation: Q8Kind::SuperBlock,
        dot: int_dot::q4_k,
        dot_rows: Some(int_dot::q4_k_rows),
        group_rows: 1,
        prepare: None,
        read_row: None,
        dot_group: None,
    },
    QuantKernel {
        type_id: 14,
        activation: Q8Kind::SuperBlock,
        dot: int_dot::q6_k,
        dot_rows: Some(int_dot::q6_k_rows),
        group_rows: 1,
        prepare: None,
        read_row: None,
        dot_group: None,
    },
    QuantKernel {
        type_id: 8,
        activation: Q8Kind::Block,
        dot: int_dot::q8_0,
        dot_rows: None,
        group_rows: 1,
        prepare: None,
        read_row: None,
        dot_group: None,
    },
];

pub(crate) static ADAPTER: Adapter = Adapter {
    name: "portable",
    f32: F32Kernels {
        dot: dot::dot,
        dot_rows: dot::dot_rows,
        mix_rows: dot::mix_rows,
        add_assign: elementwise::add_assign,
        mul_assign: elementwise::mul_assign,
        silu_assign: elementwise::silu_assign,
        rms_norm: norm::rms_norm,
        softmax: norm::softmax,
        rope_adjacent: rope::rope_adjacent,
        rope_half_split: rope::rope_half_split,
    },
    quantize: q8::quantize,
    quantized: &QUANTIZED,
};
