// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::DType;

use super::{GgmlType, TensorInfo, parse_tensor_info};
use crate::error::{GgufError, Result};
use crate::gguf::cursor::Cursor;
use crate::gguf::test_support::tensor_info_bytes;

fn parse(bytes: &[u8]) -> Result<TensorInfo> {
    parse_tensor_info(&mut Cursor::new(bytes))
}

#[test]
fn gguf_dims_are_reversed() {
    let info = parse(&tensor_info_bytes("w", &[4, 3, 2], 0, 64)).unwrap();
    assert_eq!(info.name, "w");
    assert_eq!(info.shape, vec![2, 3, 4]);
    assert_eq!(info.offset, 64);
    assert_eq!(info.dtype, DType::F32);
    assert_eq!(info.ggml_type, GgmlType::F32);
}

#[test]
fn zero_dims_become_a_scalar_of_one() {
    assert_eq!(
        parse(&tensor_info_bytes("s", &[], 0, 0)).unwrap().shape,
        [1]
    );
}

#[test]
fn each_ggml_type_maps_to_its_dtype() {
    let cases = [
        (0, GgmlType::F32, DType::F32),
        (1, GgmlType::F16, DType::F16),
        (2, GgmlType::Q4_0, DType::F32),
        (12, GgmlType::Q4K, DType::F32),
        (14, GgmlType::Q6K, DType::F32),
    ];
    for (raw, ggml, dtype) in cases {
        assert_eq!(GgmlType::from_raw(raw), Some(ggml));
        assert_eq!(ggml.raw(), raw);
        assert_eq!(ggml.materialized_dtype(), dtype);
        assert_eq!(
            parse(&tensor_info_bytes("t", &[1], raw, 0)).unwrap().dtype,
            dtype
        );
    }
}

#[test]
fn unsupported_tensor_type() {
    assert!(matches!(
        parse(&tensor_info_bytes("t", &[1], 3, 0)),
        Err(GgufError::UnsupportedTensorType(3))
    ));
}

#[test]
fn more_than_four_dimensions_are_rejected() {
    assert!(matches!(
        parse(&tensor_info_bytes("t", &[1; 5], 0, 0)),
        Err(GgufError::TooManyDimensions { got: 5, max: 4 })
    ));
}

#[test]
fn four_dimensions_are_accepted() {
    assert_eq!(
        parse(&tensor_info_bytes("t", &[1, 2, 3, 4], 0, 0))
            .unwrap()
            .shape,
        [4, 3, 2, 1]
    );
}

#[test]
fn a_huge_declared_dimension_count_is_rejected_without_allocating() {
    let mut bytes = tensor_info_bytes("t", &[], 0, 0);
    bytes[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        parse(&bytes),
        Err(GgufError::TooManyDimensions {
            got: u32::MAX,
            max: 4
        })
    ));
}

#[test]
fn declared_dims_missing_from_the_file_are_eof() {
    let mut bytes = tensor_info_bytes("t", &[], 0, 0);
    bytes[9..13].copy_from_slice(&4_u32.to_le_bytes());
    bytes.truncate(13 + 8);
    assert!(matches!(parse(&bytes), Err(GgufError::UnexpectedEof)));
}

#[test]
fn element_count_overflow_is_shape_overflow() {
    let dim = 1_u64 << 33;
    assert!(matches!(
        parse(&tensor_info_bytes("big", &[dim, dim], 0, 0)),
        Err(GgufError::ShapeOverflow { name }) if name == "big"
    ));
}

#[test]
fn a_dimension_beyond_usize_range_still_fits_when_another_is_zero() {
    let info = parse(&tensor_info_bytes("z", &[u64::MAX, 0], 0, 0)).unwrap();
    assert_eq!(info.shape, [0, usize::MAX]);
}

#[test]
fn truncated_type_tag_is_eof() {
    let bytes = tensor_info_bytes("t", &[1], 0, 0);
    assert!(matches!(
        parse(&bytes[..bytes.len() - 10]),
        Err(GgufError::UnexpectedEof)
    ));
}

#[test]
fn truncated_offset_is_eof() {
    let bytes = tensor_info_bytes("t", &[1], 0, 0);
    assert!(matches!(
        parse(&bytes[..bytes.len() - 4]),
        Err(GgufError::UnexpectedEof)
    ));
}

#[test]
fn byte_len_matches_the_on_disk_layouts() {
    assert_eq!(GgmlType::F32.byte_len(0), Some(0));
    assert_eq!(GgmlType::F32.byte_len(3), Some(12));
    assert_eq!(GgmlType::F16.byte_len(3), Some(6));
    assert_eq!(GgmlType::Q4_0.byte_len(32), Some(18));
    assert_eq!(GgmlType::Q4_0.byte_len(33), Some(36));
    assert_eq!(GgmlType::Q4K.byte_len(256), Some(144));
    assert_eq!(GgmlType::Q4K.byte_len(1), Some(144));
    assert_eq!(GgmlType::Q6K.byte_len(256), Some(210));
    assert_eq!(GgmlType::Q6K.byte_len(257), Some(420));
}

#[test]
fn byte_len_overflow_returns_none() {
    assert_eq!(GgmlType::F32.byte_len(usize::MAX), None);
    assert_eq!(GgmlType::F16.byte_len(usize::MAX), None);
}

#[test]
fn quantized_byte_len_cannot_overflow_because_blocks_compress() {
    for ggml_type in [GgmlType::Q4_0, GgmlType::Q4K, GgmlType::Q6K] {
        assert!(ggml_type.byte_len(usize::MAX).is_some(), "{ggml_type:?}");
    }
}
