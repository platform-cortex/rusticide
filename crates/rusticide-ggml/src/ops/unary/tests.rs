// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Op;
use crate::test_support::{assert_close, assert_rejects_non_f32, assert_values, f32_tensor, run};

#[test]
fn graph_scale() {
    let input = f32_tensor(&[1.0, 2.0, 3.0], &[3]);
    let out = run(Op::Scale(3.0), &[input]).unwrap();
    assert_values(&out, &[3], &[3.0, 6.0, 9.0]);
}

#[test]
fn graph_silu() {
    let input = f32_tensor(&[0.5, -1.0, 0.0], &[3]);
    let out = run(Op::Silu, &[input]).unwrap();
    let expected = [
        0.5_f32 / (1.0 + (-0.5_f32).exp()),
        -1.0_f32 / (1.0 + 1.0_f32.exp()),
        0.0,
    ];
    assert_close(&out, &[3], &expected);
}

#[test]
fn scale_rejects_non_f32() {
    assert_rejects_non_f32(Op::Scale(2.0), &[2]);
}

#[test]
fn silu_rejects_non_f32() {
    assert_rejects_non_f32(Op::Silu, &[2]);
}
