// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Error, Tensor};

use crate::Op;
use crate::test_support::{assert_values, f32_tensor, run, run_declared, tensor_error};

#[test]
fn graph_add() {
    let lhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let rhs = f32_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let out = run(Op::Add, &[lhs, rhs]).unwrap();
    assert_values(&out, &[2, 2], &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn graph_matmul() {
    let lhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let rhs = f32_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let out = run(Op::MatMul, &[lhs, rhs]).unwrap();
    assert_values(&out, &[2, 2], &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn graph_reshape() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out = run(Op::Reshape { shape: vec![3, 2] }, &[input]).unwrap();
    assert_values(&out, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn graph_repeat() {
    let input = f32_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let out = run(Op::Repeat { shape: vec![2, 3] }, &[input]).unwrap();
    assert_values(&out, &[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn graph_repeat_widens_rank_and_columns() {
    let input = f32_tensor(&[1.0, 2.0], &[2, 1]);
    let out = run(
        Op::Repeat {
            shape: vec![2, 2, 2],
        },
        &[input],
    )
    .unwrap();
    assert_values(&out, &[2, 2, 2], &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn repeat_rejects_non_f32() {
    let input = Tensor::from_slice(&[f16::from_f32(1.0), f16::from_f32(2.0)], &[2]).unwrap();
    let err = run(Op::Repeat { shape: vec![2, 2] }, &[input]).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::F16
        }
    ));
}

#[test]
fn repeat_runtime_shape_that_cannot_broadcast_is_an_error() {
    let fed = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = run_declared(
        Op::Repeat { shape: vec![2, 3] },
        &[(&[1, 3], DType::F32)],
        &[fed],
    )
    .unwrap_err();
    assert!(matches!(tensor_error(err), Error::NotBroadcastable { .. }));
}

#[test]
fn repeat_unsatisfiable_target_is_allocation_failed() {
    let input = f32_tensor(&[1.0], &[1]);
    let err = run(
        Op::Repeat {
            shape: vec![usize::MAX / 2, 2],
        },
        &[input],
    )
    .unwrap_err();
    assert!(matches!(tensor_error(err), Error::AllocationFailed { .. }));
}

#[test]
fn graph_copy_does_not_alias() {
    let input = f32_tensor(&[1.0, 2.0], &[2]);
    let out = run(Op::Copy, std::slice::from_ref(&input)).unwrap();
    assert_values(&out, &[2], &[1.0, 2.0]);
    assert!(!out.shares_storage_with(&input));
}

#[test]
fn graph_cont() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out = run(Op::Cont, &[input]).unwrap();
    assert_values(&out, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn graph_permute_noop() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out = run(Op::Permute, &[input]).unwrap();
    assert_values(&out, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn add_runtime_error_propagates() {
    let lhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let rhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let err = run_declared(
        Op::Add,
        &[(&[2, 3], DType::F32), (&[3], DType::F32)],
        &[lhs, rhs],
    )
    .unwrap_err();
    let Error::NotBroadcastable { lhs, rhs } = tensor_error(err) else {
        panic!("wrong variant");
    };
    assert_eq!((lhs, rhs), (vec![2, 3], vec![4, 2]));
}

#[test]
fn matmul_runtime_error_propagates() {
    let lhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let rhs = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = run_declared(
        Op::MatMul,
        &[(&[2, 3], DType::F32), (&[3, 2], DType::F32)],
        &[lhs, rhs],
    )
    .unwrap_err();
    assert!(matches!(tensor_error(err), Error::ShapeMismatch { .. }));
}

#[test]
fn reshape_runtime_error_propagates() {
    let fed = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = run_declared(
        Op::Reshape { shape: vec![3, 2] },
        &[(&[2, 3], DType::F32)],
        &[fed],
    )
    .unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::ElementCountMismatch {
            expected: 4,
            got: 6
        }
    ));
}
