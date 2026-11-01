// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Device, Tensor};
use rusticide_io::Metadata;
use rusticide_quantize::block_format;

use super::{Matrix, load_vector, require_kernel};
use crate::error::ModelError;
use crate::source::InMemorySource;

fn source_with(name: &str, tensor: Tensor) -> InMemorySource {
    InMemorySource::new(Metadata::default()).with_tensor(name, tensor)
}

fn matrix_source() -> InMemorySource {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    source_with("w", Tensor::from_slice(&data, &[2, 3]).unwrap())
}

#[test]
fn matrix_loads_rows_and_multiplies() {
    let matrix = Matrix::load(&matrix_source(), "w", 2, 3).unwrap();
    assert_eq!((matrix.rows(), matrix.cols()), (2, 3));
    let mut row = [0.0; 3];
    assert!(matrix.copy_row(1, &mut row));
    assert_eq!(row, [4.0, 5.0, 6.0]);
    assert!(!matrix.copy_row(2, &mut row));
    let mut output = [0.0; 2];
    matrix.matvec(&[1.0, 0.0, 1.0], &mut output);
    assert_eq!(output, [4.0, 10.0]);
}

#[test]
fn matrix_with_wrong_shape_is_shape_mismatch() {
    match Matrix::load(&matrix_source(), "w", 3, 2) {
        Err(ModelError::ShapeMismatch {
            name,
            expected,
            got,
        }) => {
            assert_eq!(name, "w");
            assert_eq!(expected, vec![3, 2]);
            assert_eq!(got, vec![2, 3]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn matrix_of_an_absent_tensor_is_missing_tensor() {
    match Matrix::load(&matrix_source(), "nope", 2, 3) {
        Err(ModelError::MissingTensor { name }) => assert_eq!(name, "nope"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn matrix_widens_f16_weights() {
    let data = [f16::from_f32(1.5), f16::from_f32(-2.0)];
    let source = source_with("w", Tensor::from_slice(&data, &[1, 2]).unwrap());
    let matrix = Matrix::load(&source, "w", 1, 2).unwrap();
    let mut row = [0.0; 2];
    assert!(matrix.copy_row(0, &mut row));
    assert_eq!(row, [1.5, -2.0]);
}

#[test]
fn matrix_rejects_other_dtypes() {
    let source = source_with(
        "w",
        Tensor::zeros(&[1, 2], DType::I64, Device::cpu()).unwrap(),
    );
    assert!(matches!(
        Matrix::load(&source, "w", 1, 2),
        Err(ModelError::Tensor(
            rusticide_core::Error::DTypeMismatch { .. }
        ))
    ));
}

#[test]
fn matrix_with_overflowing_shape_is_shape_mismatch() {
    match Matrix::load(&matrix_source(), "w", usize::MAX, 2) {
        Err(ModelError::ShapeMismatch { expected, .. }) => {
            assert_eq!(expected, vec![usize::MAX, 2]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn load_vector_reads_and_checks_length() {
    let source = source_with("v", Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap());
    assert_eq!(load_vector(&source, "v", 2).unwrap(), vec![1.0, 2.0]);
    match load_vector(&source, "v", 3) {
        Err(ModelError::ShapeMismatch {
            name,
            expected,
            got,
        }) => {
            assert_eq!(name, "v");
            assert_eq!(expected, vec![3]);
            assert_eq!(got, vec![2]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

/// Two `Q8_0` rows of 32: row 0 is 0.5 x [0, 1, ..., 31], row 1 is all -1.
fn q8_0_source(rows: usize, cols: usize) -> InMemorySource {
    let half = f16::from_f32(0.5).to_le_bytes();
    let mut blocks = half.to_vec();
    blocks.extend(0..32_u8);
    blocks.extend(f16::from_f32(1.0).to_le_bytes());
    blocks.extend([0xFF; 32]);
    InMemorySource::new(Metadata::default())
        .with_quantized("q", block_format(8).unwrap(), blocks, &[rows, cols])
        .unwrap()
}

#[test]
fn a_quantized_matrix_stays_quantized_and_multiplies() {
    let matrix = Matrix::load(&q8_0_source(2, 32), "q", 2, 32).unwrap();
    assert!(matrix.is_quantized());
    let input = [127.0_f32; 32];
    let mut output = [0.0; 2];
    matrix.matvec(&input, &mut output);
    assert_eq!(output, [248.0 * 127.0, -32.0 * 127.0]);
    let mut row = [0.0; 32];
    assert!(matrix.copy_row(1, &mut row));
    assert_eq!(row, [-1.0; 32]);
    assert!(!matrix.copy_row(2, &mut row));
}

#[test]
fn a_quantized_matrix_whose_rows_split_blocks_is_widened() {
    let matrix = Matrix::load(&q8_0_source(4, 16), "q", 4, 16).unwrap();
    assert!(!matrix.is_quantized());
    let mut row = [0.0; 16];
    assert!(matrix.copy_row(3, &mut row));
    assert_eq!(row, [-1.0; 16]);
}

#[test]
fn a_matrix_without_columns_is_invalid() {
    let source = source_with("w", Tensor::from_slice::<f32>(&[], &[2, 0]).unwrap());
    assert!(matches!(
        Matrix::load(&source, "w", 2, 0),
        Err(ModelError::InvalidConfig {
            reason: "a weight matrix has no columns"
        })
    ));
}

#[test]
fn a_format_without_a_kernel_fails_to_load() {
    let error = require_kernel(block_format(12).unwrap(), None).unwrap_err();
    assert!(matches!(
        error,
        ModelError::MissingKernel { format: "Q4_K" }
    ));
    assert_eq!(error.to_string(), "no kernel for format Q4_K");
}
