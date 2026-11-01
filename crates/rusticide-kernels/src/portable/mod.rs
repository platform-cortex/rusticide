// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod dot;
mod elementwise;
mod norm;
mod rope;

use crate::adapter::{Adapter, F32Kernels};

pub(crate) static ADAPTER: Adapter = Adapter {
    name: "portable",
    f32: F32Kernels {
        dot: dot::dot,
        dot_rows: dot::dot_rows,
        add_assign: elementwise::add_assign,
        mul_assign: elementwise::mul_assign,
        silu_assign: elementwise::silu_assign,
        rms_norm: norm::rms_norm,
        softmax: norm::softmax,
        rope_adjacent: rope::rope_adjacent,
        rope_half_split: rope::rope_half_split,
    },
};
