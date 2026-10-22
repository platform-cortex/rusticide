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
fn graph_softmax() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let out = run(Op::SoftMax, &[input]).unwrap();
    let sum = (-2.0_f32).exp() + (-1.0_f32).exp() + 1.0;
    let row = [(-2.0_f32).exp() / sum, (-1.0_f32).exp() / sum, 1.0 / sum];
    assert_close(
        &out,
        &[2, 3],
        &[row[0], row[1], row[2], row[0], row[1], row[2]],
    );
    let values = out.to_vec::<f32>().unwrap();
    for chunk in values.chunks(3) {
        assert!((chunk.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn softmax_stability() {
    let input = f32_tensor(&[100.0, 101.0, 102.0, 103.0], &[1, 4]);
    let out = run(Op::SoftMax, &[input]).unwrap();
    let values = out.to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_of_a_scalar() {
    let out = run(Op::SoftMax, &[f32_tensor(&[5.0], &[])]).unwrap();
    assert_values(&out, &[], &[0.0]);
}

#[test]
fn softmax_with_an_empty_last_dim() {
    let out = run(Op::SoftMax, &[f32_tensor(&[], &[2, 0])]).unwrap();
    assert_values(&out, &[2, 0], &[]);
}

#[test]
fn softmax_rejects_non_f32() {
    assert_rejects_non_f32(Op::SoftMax, &[2]);
}

#[test]
fn graph_rms_norm() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let out = run(Op::RMSNorm { eps: 1e-6 }, &[input]).unwrap();
    let rms0 = (7.5_f32 + 1e-6).sqrt();
    let rms1 = (43.5_f32 + 1e-6).sqrt();
    let expected = [
        1.0 / rms0,
        2.0 / rms0,
        3.0 / rms0,
        4.0 / rms0,
        5.0 / rms1,
        6.0 / rms1,
        7.0 / rms1,
        8.0 / rms1,
    ];
    assert_close(&out, &[2, 4], &expected);
}

#[test]
fn rms_norm_eps() {
    let input = f32_tensor(&[0.0, 0.0, 0.0, 3.0, 0.0, 0.0], &[2, 3]);
    let out = run(Op::RMSNorm { eps: 0.1 }, &[input]).unwrap();
    let rms = (3.0_f32 + 0.1).sqrt();
    assert_close(&out, &[2, 3], &[0.0, 0.0, 0.0, 3.0 / rms, 0.0, 0.0]);
}

#[test]
fn rms_norm_with_an_empty_last_dim() {
    let out = run(Op::RMSNorm { eps: 0.1 }, &[f32_tensor(&[], &[2, 0])]).unwrap();
    assert_values(&out, &[2, 0], &[]);
}

#[test]
fn rms_norm_requires_rank_1() {
    let scalar = f32_tensor(&[1.0], &[]);
    let err = run_declared(Op::RMSNorm { eps: 0.1 }, &[(&[2], DType::F32)], &[scalar]).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn rms_norm_rejects_non_f32() {
    assert_rejects_non_f32(Op::RMSNorm { eps: 0.1 }, &[2]);
}
