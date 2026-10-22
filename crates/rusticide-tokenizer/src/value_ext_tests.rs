// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::error::TokenizerError;
use crate::value_ext::GGUFValueExt;

#[test]
fn as_u32_lossy_uint64_in_range() {
    assert_eq!(GGUFValue::Uint64(7).as_u32_lossy().unwrap(), 7);
}

#[test]
fn as_u32_lossy_int64_in_range() {
    assert_eq!(GGUFValue::Int64(7).as_u32_lossy().unwrap(), 7);
}

#[test]
fn as_u32_lossy_int32_negative_is_invalid_id() {
    let err = GGUFValue::Int32(-1).as_u32_lossy().unwrap_err();
    assert!(matches!(err, TokenizerError::InvalidId(-1)));
}

#[test]
fn as_u32_lossy_uint64_out_of_range_is_err() {
    let too_big = u64::from(u32::MAX) + 1;
    let err = GGUFValue::Uint64(too_big).as_u32_lossy().unwrap_err();
    assert!(matches!(
        err,
        TokenizerError::MissingMetadata("id exceeds u32")
    ));
}

#[test]
fn as_u32_lossy_non_integer_variant_is_err() {
    let err = GGUFValue::String("x".to_string())
        .as_u32_lossy()
        .unwrap_err();
    assert!(matches!(err, TokenizerError::MissingMetadata(_)));
}

#[test]
fn value_ext_as_string_accepts_a_string_only() {
    assert_eq!(GGUFValue::String("x".to_string()).as_string().unwrap(), "x");
    assert!(matches!(
        GGUFValue::Bool(true).as_string(),
        Err(TokenizerError::MissingMetadata(
            "expected string value type"
        ))
    ));
}

#[test]
fn value_ext_as_array_accepts_an_array_only() {
    let array = GGUFValue::Array(vec![GGUFValue::Bool(true)]);
    assert_eq!(array.as_array().unwrap().len(), 1);
    assert!(matches!(
        GGUFValue::Bool(true).as_array(),
        Err(TokenizerError::MissingMetadata("expected array value type"))
    ));
}

#[test]
fn value_ext_as_f32_accepts_a_float32_only() {
    assert!((GGUFValue::Float32(1.5).as_f32().unwrap() - 1.5).abs() < f32::EPSILON);
    assert!(matches!(
        GGUFValue::Float64(1.5).as_f32(),
        Err(TokenizerError::MissingMetadata(
            "expected float32 value type"
        ))
    ));
}

#[test]
fn value_ext_as_bool_accepts_a_bool_only() {
    assert!(GGUFValue::Bool(true).as_bool().unwrap());
    assert!(matches!(
        GGUFValue::Uint8(1).as_bool(),
        Err(TokenizerError::MissingMetadata("expected bool value type"))
    ));
}

#[test]
fn value_ext_as_u32_lossy_accepts_uint32_and_non_negative_int32() {
    assert_eq!(GGUFValue::Uint32(9).as_u32_lossy().unwrap(), 9);
    assert_eq!(GGUFValue::Int32(9).as_u32_lossy().unwrap(), 9);
}

#[test]
fn value_ext_as_u32_lossy_negative_int64_is_out_of_range() {
    assert!(matches!(
        GGUFValue::Int64(-1).as_u32_lossy(),
        Err(TokenizerError::MissingMetadata("id exceeds u32"))
    ));
}
