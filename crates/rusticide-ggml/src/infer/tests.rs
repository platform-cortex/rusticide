// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Error};

use super::*;
use crate::test_support::tensor_error;

type Input<'a> = (&'a [usize], DType);

#[test]
fn add_broadcasts() {
    let inputs: [Input<'_>; 2] = [(&[2, 3], DType::F32), (&[3], DType::F32)];
    let (shape, dtype) = infer_shape_dtype(&Op::Add, &inputs).unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(dtype, DType::F32);
}

#[test]
fn add_dtype_mismatch() {
    let inputs: [Input<'_>; 2] = [(&[2], DType::F32), (&[2], DType::I64)];
    let err = infer_shape_dtype(&Op::Add, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::I64
        }
    ));
}

#[test]
fn add_not_broadcastable() {
    let inputs: [Input<'_>; 2] = [(&[2, 3], DType::F32), (&[4, 3], DType::F32)];
    let err = infer_shape_dtype(&Op::Add, &inputs).unwrap_err();
    let Error::NotBroadcastable { lhs, rhs } = tensor_error(err) else {
        panic!("wrong variant");
    };
    assert_eq!((lhs, rhs), (vec![2, 3], vec![4, 3]));
}

#[test]
fn matmul_output_shape() {
    let inputs: [Input<'_>; 2] = [(&[2, 3], DType::F32), (&[3, 4], DType::F32)];
    let (shape, dtype) = infer_shape_dtype(&Op::MatMul, &inputs).unwrap();
    assert_eq!((shape, dtype), (vec![2, 4], DType::F32));
}

#[test]
fn matmul_requires_2d_lhs() {
    let inputs: [Input<'_>; 2] = [(&[3], DType::F32), (&[3, 4], DType::F32)];
    let err = infer_shape_dtype(&Op::MatMul, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 2,
            got: 1
        }
    ));
}

#[test]
fn matmul_requires_2d_rhs() {
    let inputs: [Input<'_>; 2] = [(&[2, 3], DType::F32), (&[3, 4, 1], DType::F32)];
    let err = infer_shape_dtype(&Op::MatMul, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 2,
            got: 3
        }
    ));
}

#[test]
fn matmul_inner_dims() {
    let inputs: [Input<'_>; 2] = [(&[2, 3], DType::F32), (&[4, 5], DType::F32)];
    let err = infer_shape_dtype(&Op::MatMul, &inputs).unwrap_err();
    let Error::ShapeMismatch { expected, got } = tensor_error(err) else {
        panic!("wrong variant");
    };
    assert_eq!((expected, got), (vec![2, 3], vec![4, 5]));
}

#[test]
fn reshape_infers_target_shape_and_keeps_dtype() {
    let op = Op::Reshape { shape: vec![3, 2] };
    let (shape, dtype) = infer_shape_dtype(&op, &[(&[2, 3], DType::I64)]).unwrap();
    assert_eq!((shape, dtype), (vec![3, 2], DType::I64));
}

#[test]
fn reshape_element_count_mismatch() {
    let op = Op::Reshape { shape: vec![4, 2] };
    let err = infer_shape_dtype(&op, &[(&[2, 3], DType::F32)]).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::ElementCountMismatch {
            expected: 6,
            got: 8
        }
    ));
}

#[test]
fn repeat_infers_target_shape() {
    let op = Op::Repeat { shape: vec![2, 3] };
    let (shape, dtype) = infer_shape_dtype(&op, &[(&[1, 3], DType::F32)]).unwrap();
    assert_eq!((shape, dtype), (vec![2, 3], DType::F32));
}

#[test]
fn repeat_not_broadcastable() {
    let op = Op::Repeat { shape: vec![2, 4] };
    let err = infer_shape_dtype(&op, &[(&[2, 3], DType::F32)]).unwrap_err();
    let Error::NotBroadcastable { lhs, rhs } = tensor_error(err) else {
        panic!("wrong variant");
    };
    assert_eq!((lhs, rhs), (vec![2, 3], vec![2, 4]));
}

#[test]
fn repeat_target_must_not_shrink_the_input() {
    let op = Op::Repeat { shape: vec![1, 3] };
    let err = infer_shape_dtype(&op, &[(&[2, 3], DType::F32)]).unwrap_err();
    assert!(matches!(tensor_error(err), Error::NotBroadcastable { .. }));
}

#[test]
fn get_rows_requires_1d_indices() {
    let inputs: [Input<'_>; 2] = [(&[4, 2], DType::F32), (&[3, 1], DType::F32)];
    let err = infer_shape_dtype(&Op::GetRows, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 2
        }
    ));
}

#[test]
fn get_rows_rejects_scalar_indices() {
    let inputs: [Input<'_>; 2] = [(&[4, 2], DType::F32), (&[], DType::F32)];
    let err = infer_shape_dtype(&Op::GetRows, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn get_rows_requires_data_of_rank_one_or_more() {
    let inputs: [Input<'_>; 2] = [(&[], DType::F32), (&[3], DType::F32)];
    let err = infer_shape_dtype(&Op::GetRows, &inputs).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn get_rows_output_shape() {
    let inputs: [Input<'_>; 2] = [(&[4, 2, 5], DType::F32), (&[3], DType::F32)];
    let (shape, dtype) = infer_shape_dtype(&Op::GetRows, &inputs).unwrap();
    assert_eq!((shape, dtype), (vec![3, 2, 5], DType::F32));
}

#[test]
fn copy_cont_permute_keep_shape_and_dtype() {
    for op in [Op::Copy, Op::Cont, Op::Permute] {
        let (shape, dtype) = infer_shape_dtype(&op, &[(&[2, 3], DType::I64)]).unwrap();
        assert_eq!((shape, dtype), (vec![2, 3], DType::I64), "{op:?}");
    }
}
