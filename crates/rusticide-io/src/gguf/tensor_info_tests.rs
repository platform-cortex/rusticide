// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{TensorInfo, parse_tensor_info};
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
    assert_eq!(info.type_id, 0);
}

#[test]
fn zero_dims_become_a_scalar_of_one() {
    assert_eq!(
        parse(&tensor_info_bytes("s", &[], 0, 0)).unwrap().shape,
        [1]
    );
}

#[test]
fn parse_records_an_unknown_type_id() {
    let info = parse(&tensor_info_bytes("t", &[1], 99, 0)).unwrap();
    assert_eq!(info.type_id, 99);
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
