// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Error, Tensor};

use crate::test_support::{assert_values, f32_tensor, run, run_declared, tensor_error};
use crate::{GraphError, Op};

fn invalid_index_bits(err: &GraphError) -> u32 {
    let GraphError::InvalidIndex { value } = err else {
        panic!("expected InvalidIndex, got {err:?}");
    };
    value.to_bits()
}

fn rows_data() -> Tensor {
    f32_tensor(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[4, 2])
}

#[test]
fn graph_get_rows() {
    let indices = f32_tensor(&[0.0, 2.0, 3.0], &[3]);
    let out = run(Op::GetRows, &[rows_data(), indices]).unwrap();
    assert_values(&out, &[3, 2], &[0.0, 1.0, 20.0, 21.0, 30.0, 31.0]);
}

#[test]
fn get_rows_1d_data() {
    let data = f32_tensor(&[5.0, 6.0, 7.0], &[3]);
    let indices = f32_tensor(&[2.0, 0.0], &[2]);
    let out = run(Op::GetRows, &[data, indices]).unwrap();
    assert_values(&out, &[2], &[7.0, 5.0]);
}

#[test]
fn get_rows_3d_data() {
    let values: Vec<f32> = (0..12_u8).map(f32::from).collect();
    let data = f32_tensor(&values, &[3, 2, 2]);
    let indices = f32_tensor(&[2.0, 1.0], &[2]);
    let out = run(Op::GetRows, &[data, indices]).unwrap();
    assert_values(
        &out,
        &[2, 2, 2],
        &[8.0, 9.0, 10.0, 11.0, 4.0, 5.0, 6.0, 7.0],
    );
}

#[test]
fn get_rows_empty_indices() {
    let out = run(Op::GetRows, &[rows_data(), f32_tensor(&[], &[0])]).unwrap();
    assert_values(&out, &[0, 2], &[]);
}

#[test]
fn get_rows_index_out_of_range() {
    let err = run(Op::GetRows, &[rows_data(), f32_tensor(&[4.0], &[1])]).unwrap_err();
    let Error::IndexOutOfBounds { index, shape } = tensor_error(err) else {
        panic!("wrong variant");
    };
    assert_eq!((index, shape), (vec![4], vec![4, 2]));
}

#[test]
fn get_rows_huge_index_is_out_of_range() {
    let err = run(Op::GetRows, &[rows_data(), f32_tensor(&[1.0e30], &[1])]).unwrap_err();
    assert!(matches!(tensor_error(err), Error::IndexOutOfBounds { .. }));
}

#[test]
fn get_rows_rejects_nan_index() {
    let err = run(Op::GetRows, &[rows_data(), f32_tensor(&[f32::NAN], &[1])]).unwrap_err();
    assert_eq!(invalid_index_bits(&err), f32::NAN.to_bits());
}

#[test]
fn get_rows_rejects_infinite_index() {
    let indices = f32_tensor(&[f32::INFINITY], &[1]);
    let err = run(Op::GetRows, &[rows_data(), indices]).unwrap_err();
    assert_eq!(invalid_index_bits(&err), f32::INFINITY.to_bits());
}

#[test]
fn get_rows_rejects_negative_index() {
    let err = run(Op::GetRows, &[rows_data(), f32_tensor(&[-1.0], &[1])]).unwrap_err();
    assert_eq!(invalid_index_bits(&err), (-1.0_f32).to_bits());
}

#[test]
fn get_rows_rejects_fractional_index() {
    let err = run(Op::GetRows, &[rows_data(), f32_tensor(&[1.5], &[1])]).unwrap_err();
    assert_eq!(invalid_index_bits(&err), 1.5_f32.to_bits());
}

#[test]
fn get_rows_rejects_f16_data() {
    let data = Tensor::from_slice(&[f16::from_f32(1.0), f16::from_f32(2.0)], &[2]).unwrap();
    let err = run(Op::GetRows, &[data, f32_tensor(&[0.0], &[1])]).unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::F16
        }
    ));
}

#[test]
fn get_rows_runtime_indices_must_be_1d() {
    let fed = f32_tensor(&[0.0, 1.0], &[2, 1]);
    let err = run_declared(
        Op::GetRows,
        &[(&[4, 2], DType::F32), (&[2], DType::F32)],
        &[rows_data(), fed],
    )
    .unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 2
        }
    ));
}

#[test]
fn get_rows_runtime_data_must_not_be_scalar() {
    let err = run_declared(
        Op::GetRows,
        &[(&[4, 2], DType::F32), (&[1], DType::F32)],
        &[f32_tensor(&[1.0], &[]), f32_tensor(&[0.0], &[1])],
    )
    .unwrap_err();
    assert!(matches!(
        tensor_error(err),
        Error::RankMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn get_rows_of_a_zero_element_tensor_with_huge_row_extents_is_empty() {
    let data = Tensor::zeros(
        &[1, 1 << 40, 1 << 40, 0],
        DType::F32,
        rusticide_core::Device::cpu(),
    )
    .unwrap();
    let indices = f32_tensor(&[0.0], &[1]);
    let out = run(Op::GetRows, &[data, indices]).unwrap();
    assert_eq!(out.shape().dims(), &[1, 1 << 40, 1 << 40, 0]);
}
