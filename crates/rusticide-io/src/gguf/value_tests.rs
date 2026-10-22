// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{GGUFValue, GGUFValueType, MAX_ARRAY_DEPTH, parse_typed_value, parse_value};
use crate::error::GgufError;
use crate::gguf::cursor::Cursor;

fn parse(ty: GGUFValueType, bytes: &[u8]) -> GGUFValue {
    parse_value(&mut Cursor::new(bytes), ty).unwrap()
}

fn parse_error(ty: GGUFValueType, bytes: &[u8]) -> GgufError {
    parse_value(&mut Cursor::new(bytes), ty).unwrap_err()
}

fn array_header(item_tag: u32, len: u64) -> Vec<u8> {
    let mut bytes = item_tag.to_le_bytes().to_vec();
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes
}

fn string_bytes(text: &str) -> Vec<u8> {
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

#[test]
fn each_scalar_type_parses_little_endian() {
    use GGUFValueType as T;
    assert_eq!(parse(T::Uint8, &[200]), GGUFValue::Uint8(200));
    assert_eq!(parse(T::Int8, &[0xff]), GGUFValue::Int8(-1));
    assert_eq!(parse(T::Uint16, &[0x01, 0x02]), GGUFValue::Uint16(0x0201));
    assert_eq!(parse(T::Int16, &[0xfe, 0xff]), GGUFValue::Int16(-2));
    assert_eq!(
        parse(T::Uint32, &[1, 2, 3, 4]),
        GGUFValue::Uint32(0x0403_0201)
    );
    assert_eq!(
        parse(T::Int32, &(-5_i32).to_le_bytes()),
        GGUFValue::Int32(-5)
    );
    assert_eq!(
        parse(T::Float32, &1.5_f32.to_le_bytes()),
        GGUFValue::Float32(1.5)
    );
    assert_eq!(parse(T::Bool, &[1]), GGUFValue::Bool(true));
    assert_eq!(parse(T::Bool, &[0]), GGUFValue::Bool(false));
    assert_eq!(parse(T::Bool, &[7]), GGUFValue::Bool(true));
    assert_eq!(
        parse(T::Int64, &(-9_i64).to_le_bytes()),
        GGUFValue::Int64(-9)
    );
    assert_eq!(
        parse(T::Float64, &2.5_f64.to_le_bytes()),
        GGUFValue::Float64(2.5)
    );
}

#[test]
fn string_value_parses() {
    assert_eq!(
        parse(GGUFValueType::String, &string_bytes("hi")),
        GGUFValue::String("hi".to_owned())
    );
}

#[test]
fn uint64_max_round_trips() {
    assert_eq!(
        parse(GGUFValueType::Uint64, &u64::MAX.to_le_bytes()),
        GGUFValue::Uint64(u64::MAX)
    );
}

#[test]
fn typed_value_reads_the_tag_first() {
    let mut bytes = 4_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&7_u32.to_le_bytes());
    let mut cursor = Cursor::new(&bytes);
    assert_eq!(
        parse_typed_value(&mut cursor).unwrap(),
        GGUFValue::Uint32(7)
    );
}

#[test]
fn invalid_value_type_tag() {
    let bytes = 13_u32.to_le_bytes();
    assert!(matches!(
        parse_typed_value(&mut Cursor::new(&bytes)),
        Err(GgufError::InvalidValueType(13))
    ));
}

#[test]
fn array_of_strings() {
    let mut bytes = array_header(8, 2);
    bytes.extend(string_bytes("a"));
    bytes.extend(string_bytes("bc"));
    assert_eq!(
        parse(GGUFValueType::Array, &bytes),
        GGUFValue::Array(vec![
            GGUFValue::String("a".to_owned()),
            GGUFValue::String("bc".to_owned())
        ])
    );
}

#[test]
fn array_of_arrays() {
    let mut bytes = array_header(9, 2);
    bytes.extend(array_header(0, 1));
    bytes.push(5);
    bytes.extend(array_header(0, 0));
    assert_eq!(
        parse(GGUFValueType::Array, &bytes),
        GGUFValue::Array(vec![
            GGUFValue::Array(vec![GGUFValue::Uint8(5)]),
            GGUFValue::Array(vec![])
        ])
    );
}

#[test]
fn empty_array() {
    assert_eq!(
        parse(GGUFValueType::Array, &array_header(4, 0)),
        GGUFValue::Array(vec![])
    );
}

#[test]
fn invalid_array_item_type() {
    assert!(matches!(
        parse_error(GGUFValueType::Array, &array_header(99, 1)),
        GgufError::InvalidValueType(99)
    ));
}

#[test]
fn array_claiming_more_items_than_bytes_is_eof() {
    let mut bytes = array_header(0, u64::MAX);
    bytes.extend_from_slice(&[1, 2, 3]);
    assert!(matches!(
        parse_error(GGUFValueType::Array, &bytes),
        GgufError::UnexpectedEof
    ));
}

#[test]
fn truncated_scalar_is_eof() {
    for (ty, width) in [
        (GGUFValueType::Uint8, 1),
        (GGUFValueType::Uint16, 2),
        (GGUFValueType::Uint32, 4),
        (GGUFValueType::Uint64, 8),
    ] {
        let bytes = vec![0; width - 1];
        assert!(matches!(parse_error(ty, &bytes), GgufError::UnexpectedEof));
    }
}

fn nested_array_bytes(depth: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 1..depth {
        bytes.extend(array_header(9, 1));
    }
    bytes.extend(array_header(0, 0));
    bytes
}

#[test]
fn arrays_nested_to_the_limit_parse() {
    let value = parse(GGUFValueType::Array, &nested_array_bytes(MAX_ARRAY_DEPTH));
    let expected = (1..MAX_ARRAY_DEPTH).fold(GGUFValue::Array(vec![]), |inner, _| {
        GGUFValue::Array(vec![inner])
    });
    assert_eq!(value, expected);
}

#[test]
fn arrays_nested_past_the_limit_are_rejected() {
    let bytes = nested_array_bytes(MAX_ARRAY_DEPTH + 1);
    assert!(matches!(
        parse_error(GGUFValueType::Array, &bytes),
        GgufError::NestingTooDeep { limit: 8 }
    ));
}

#[test]
fn deeply_nested_input_does_not_overflow_the_stack() {
    let bytes: Vec<u8> = (0..100_000).flat_map(|_| array_header(9, 1)).collect();
    assert!(matches!(
        parse_error(GGUFValueType::Array, &bytes),
        GgufError::NestingTooDeep { limit: 8 }
    ));
}

#[test]
fn every_wire_tag_maps_to_its_type() {
    let expected = [
        (0, GGUFValueType::Uint8),
        (1, GGUFValueType::Int8),
        (2, GGUFValueType::Uint16),
        (3, GGUFValueType::Int16),
        (4, GGUFValueType::Uint32),
        (5, GGUFValueType::Int32),
        (6, GGUFValueType::Float32),
        (7, GGUFValueType::Bool),
        (8, GGUFValueType::String),
        (9, GGUFValueType::Array),
        (10, GGUFValueType::Uint64),
        (11, GGUFValueType::Int64),
        (12, GGUFValueType::Float64),
    ];
    for (tag, ty) in expected {
        assert_eq!(GGUFValueType::from_u32(tag), Some(ty));
    }
    assert_eq!(GGUFValueType::from_u32(13), None);
    assert_eq!(GGUFValueType::from_u32(u32::MAX), None);
}
