// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;

use crate::{DType, Error, Tensor};

fn matrix() -> Tensor {
    Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid data")
}

#[test]
fn from_slice_keeps_shape_dtype_and_values() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_slice(&data, &[2, 3]).expect("valid data");
    assert_eq!(t.shape().dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.to_vec::<f32>().expect("f32 tensor"), data);
}

#[test]
fn from_slice_of_f16_has_dtype_f16() {
    let data = [f16::from_f32(1.5), f16::from_f32(-2.0)];
    let t = Tensor::from_slice(&data, &[2]).expect("valid data");
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.to_vec::<f16>().expect("f16 tensor"), data);
}

#[test]
fn from_slice_rejects_a_length_that_disagrees_with_the_shape() {
    let err = Tensor::from_slice(&[0.0f32; 5], &[2, 3]).expect_err("5 != 6");
    assert!(matches!(
        err,
        Error::ElementCountMismatch {
            expected: 6,
            got: 5
        }
    ));
}

#[test]
fn from_slice_of_an_empty_shape_is_a_scalar() {
    let t = Tensor::from_slice(&[7.0f32], &[]).expect("scalar");
    assert_eq!(t.numel(), 1);
    assert_eq!(t.get::<f32>(&[]).expect("scalar index"), 7.0);
}

#[test]
fn get_reads_row_major() {
    assert_eq!(matrix().get::<f32>(&[1, 2]).expect("in bounds"), 6.0);
    assert_eq!(matrix().get::<f32>(&[0, 1]).expect("in bounds"), 2.0);
}

#[test]
fn get_with_the_wrong_rank_is_out_of_bounds() {
    let err = matrix().get::<f32>(&[1]).expect_err("rank 1 on rank 2");
    let Error::IndexOutOfBounds { index, shape } = err else {
        panic!("expected IndexOutOfBounds, got {err:?}");
    };
    assert_eq!(index, vec![1]);
    assert_eq!(shape, vec![2, 3]);
}

#[test]
fn get_past_an_extent_is_out_of_bounds() {
    let err = matrix().get::<f32>(&[0, 3]).expect_err("3 >= 3");
    let Error::IndexOutOfBounds { index, shape } = err else {
        panic!("expected IndexOutOfBounds, got {err:?}");
    };
    assert_eq!(index, vec![0, 3]);
    assert_eq!(shape, vec![2, 3]);
}

#[test]
fn get_with_the_wrong_element_type_is_a_dtype_mismatch() {
    let err = matrix().get::<f16>(&[0, 0]).expect_err("f16 on f32");
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            expected: DType::F16,
            got: DType::F32
        }
    ));
}

#[test]
fn set_writes_one_element_and_bumps_the_version() {
    let t = matrix();
    t.set(&[1, 0], 40.0f32).expect("in bounds");
    assert_eq!(t.version(), 1);
    assert_eq!(
        t.to_vec::<f32>().expect("f32 tensor"),
        [1.0, 2.0, 3.0, 40.0, 5.0, 6.0]
    );
}

#[test]
fn set_is_visible_through_an_aliasing_clone() {
    let a = matrix();
    let b = a.clone();
    a.set(&[0, 0], 9.0f32).expect("in bounds");
    assert_eq!(b.get::<f32>(&[0, 0]).expect("in bounds"), 9.0);
    assert_eq!(b.version(), 1);
}

#[test]
fn set_rejects_a_bad_index() {
    let t = matrix();
    let err = t.set(&[2, 0], 1.0f32).expect_err("2 >= 2");
    let Error::IndexOutOfBounds { index, shape } = err else {
        panic!("expected IndexOutOfBounds, got {err:?}");
    };
    assert_eq!(index, vec![2, 0]);
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(t.version(), 0, "a failed set is not a mutation");
}

#[test]
fn set_rejects_the_wrong_element_type() {
    let t = matrix();
    let err = t.set(&[0, 0], f16::ONE).expect_err("f16 on f32");
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            expected: DType::F16,
            got: DType::F32
        }
    ));
    assert_eq!(t.version(), 0);
}

#[test]
fn to_vec_rejects_the_wrong_element_type() {
    let err = matrix().to_vec::<f16>().expect_err("f16 on f32");
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            expected: DType::F16,
            got: DType::F32
        }
    ));
}

#[test]
fn ndim_counts_dimensions() {
    let cube = Tensor::from_slice(&[0.0f32; 24], &[2, 3, 4]).expect("valid data");
    assert_eq!(cube.ndim(), 3);
    let scalar = Tensor::from_slice(&[0.0f32], &[]).expect("scalar");
    assert_eq!(scalar.ndim(), 0);
}
