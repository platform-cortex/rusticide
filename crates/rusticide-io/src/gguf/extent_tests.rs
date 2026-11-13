// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::error::GgufError;
use crate::gguf::file::GgufFile;
use crate::gguf::test_support::{GgufBuilder, header_bytes, tensor_info_bytes};

const F32: u32 = 0;

/// Two tensors at explicit offsets over a 128-byte data section.
fn two_tensors(first: (&[u64], u32, u64), second: (&[u64], u32, u64)) -> Vec<u8> {
    let mut bytes = header_bytes(3, 2, 0);
    bytes.extend(tensor_info_bytes("a", first.0, first.1, first.2));
    bytes.extend(tensor_info_bytes("b", second.0, second.1, second.2));
    bytes.resize(bytes.len().next_multiple_of(32) + 128, 0);
    bytes
}

fn overlap(bytes: Vec<u8>) -> Option<(String, String)> {
    match GgufFile::from_bytes(bytes) {
        Err(GgufError::OverlappingTensors { first, second }) => Some((first, second)),
        other => {
            assert!(other.is_ok(), "{other:?}");
            None
        }
    }
}

#[test]
fn tensors_sharing_their_bytes_are_rejected() {
    let bytes = two_tensors((&[4], F32, 0), (&[4], F32, 0));
    assert_eq!(overlap(bytes), Some(("a".into(), "b".into())));
}

#[test]
fn a_tensor_starting_inside_another_is_rejected() {
    let bytes = two_tensors((&[16], F32, 0), (&[4], F32, 32));
    assert_eq!(overlap(bytes), Some(("a".into(), "b".into())));
}

#[test]
fn the_report_names_tensors_in_file_order_not_table_order() {
    let bytes = two_tensors((&[4], F32, 32), (&[16], F32, 0));
    assert_eq!(overlap(bytes), Some(("b".into(), "a".into())));
}

#[test]
fn adjacent_tensors_are_accepted() {
    let bytes = two_tensors((&[8], F32, 0), (&[8], F32, 32));
    assert_eq!(overlap(bytes), None);
}

#[test]
fn an_empty_tensor_overlaps_nothing() {
    let bytes = two_tensors((&[0], F32, 0), (&[4], F32, 0));
    assert_eq!(overlap(bytes), None);
}

#[test]
fn a_tensor_of_unknown_type_is_left_to_fail_on_load() {
    let bytes = two_tensors((&[4], F32, 0), (&[4], 999, 0));
    assert_eq!(overlap(bytes), None);
}

#[test]
fn a_tensor_whose_end_overflows_is_left_to_fail_on_load() {
    let bytes = two_tensors((&[4], F32, 0), (&[4], F32, u64::MAX - 31));
    assert_eq!(overlap(bytes), None);
}

#[test]
fn builder_files_with_many_tensors_have_no_overlap() {
    let bytes = GgufBuilder::new()
        .tensor("x", &[3], F32, &[0; 12])
        .tensor("y", &[5], F32, &[0; 20])
        .tensor("z", &[1], F32, &[0; 4])
        .build();
    assert_eq!(overlap(bytes), None);
}

#[test]
fn byte_len_matches_each_storage_kind() {
    let info = |shape: Vec<usize>, type_id| TensorInfo {
        name: "t".into(),
        shape,
        type_id,
        offset: 0,
    };
    assert_eq!(byte_len(&info(vec![3], 0)).unwrap().1, 12);
    assert_eq!(byte_len(&info(vec![3], 1)).unwrap().1, 6);
    assert_eq!(byte_len(&info(vec![32], 8)).unwrap().1, 34);
    assert_eq!(byte_len(&info(vec![256], 36)).unwrap().1, 96);
    assert!(matches!(
        byte_len(&info(vec![usize::MAX, 2], 0)),
        Err(GgufError::CastOverflow(_))
    ));
}

#[test]
fn an_empty_tensor_inside_another_is_accepted() {
    let bytes = two_tensors((&[16], F32, 0), (&[0], F32, 32));
    assert_eq!(overlap(bytes), None);
}

#[test]
fn the_i2_s_trailer_belongs_to_its_tensor() {
    let bytes = two_tensors((&[128], 36, 0), (&[4], F32, 32));
    assert_eq!(overlap(bytes), Some(("a".into(), "b".into())));
    let bytes = two_tensors((&[128], 36, 0), (&[4], F32, 64));
    assert_eq!(overlap(bytes), None);
}
