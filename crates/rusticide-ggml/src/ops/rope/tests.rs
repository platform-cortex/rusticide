// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Error};

use crate::Op;
use crate::test_support::{
    assert_close, assert_rejects_non_f32, assert_values, f32_tensor, run, run_declared,
    tensor_error,
};

#[test]
fn graph_rope() {
    let op = Op::RoPE {
        theta: 10000.0,
        dim_pairs: 2,
    };
    let out = run(op, &[f32_tensor(&[1.0; 12], &[3, 4])]).unwrap();
    let (cos1, sin1) = (1.0_f32.cos(), 1.0_f32.sin());
    let (cos2, sin2) = (2.0_f32.cos(), 2.0_f32.sin());
    let freq = 10000.0_f32.powf(-2.0 / 4.0);
    let (slow_cos1, slow_sin1) = (freq.cos(), freq.sin());
    let (slow_cos2, slow_sin2) = ((2.0 * freq).cos(), (2.0 * freq).sin());
    let expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        cos1 - sin1,
        cos1 + sin1,
        slow_cos1 - slow_sin1,
        slow_cos1 + slow_sin1,
        cos2 - sin2,
        cos2 + sin2,
        slow_cos2 - slow_sin2,
        slow_cos2 + slow_sin2,
    ];
    assert_close(&out, &[3, 4], &expected);
}

#[test]
fn rope_dim_pairs_beyond_dim_stop() {
    let op = Op::RoPE {
        theta: 10000.0,
        dim_pairs: 5,
    };
    let out = run(op, &[f32_tensor(&[1.0, 1.0, 7.0, 1.0, 1.0, 7.0], &[2, 3])]).unwrap();
    let (cos1, sin1) = (1.0_f32.cos(), 1.0_f32.sin());
    assert_close(
        &out,
        &[2, 3],
        &[1.0, 1.0, 7.0, cos1 - sin1, cos1 + sin1, 7.0],
    );
}

#[test]
fn rope_with_an_empty_row() {
    let op = Op::RoPE {
        theta: 10000.0,
        dim_pairs: 2,
    };
    let out = run(op, &[f32_tensor(&[], &[2, 0])]).unwrap();
    assert_values(&out, &[2, 0], &[]);
}

#[test]
fn rope_requires_2d() {
    let op = Op::RoPE {
        theta: 10000.0,
        dim_pairs: 1,
    };
    let fed = f32_tensor(&[1.0, 2.0], &[2]);
    let err = run_declared(op, &[(&[1, 2], DType::F32)], &[fed]).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 2,
            got: 1
        }
    ));
}

#[test]
fn rope_rejects_non_f32() {
    assert_rejects_non_f32(
        Op::RoPE {
            theta: 10000.0,
            dim_pairs: 1,
        },
        &[1, 2],
    );
}

#[test]
fn rope_rotates_only_dim_pairs_pairs() {
    let op = Op::RoPE {
        theta: 10000.0,
        dim_pairs: 1,
    };
    let out = run(op, &[f32_tensor(&[1.0; 8], &[2, 4])]).unwrap();
    let (cos1, sin1) = (1.0_f32.cos(), 1.0_f32.sin());
    let expected = [1.0, 1.0, 1.0, 1.0, cos1 - sin1, cos1 + sin1, 1.0, 1.0];
    assert_close(&out, &[2, 4], &expected);
}
