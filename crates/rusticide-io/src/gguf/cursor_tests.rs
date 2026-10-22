// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Cursor;
use crate::error::GgufError;

#[test]
fn take_advances_the_position() {
    let mut cursor = Cursor::new(&[1, 2, 3, 4, 5]);
    assert_eq!(cursor.take(2).unwrap(), &[1, 2]);
    assert_eq!(cursor.position(), 2);
    assert_eq!(cursor.remaining(), 3);
    assert_eq!(cursor.take(3).unwrap(), &[3, 4, 5]);
    assert_eq!(cursor.remaining(), 0);
}

#[test]
fn take_past_the_end_is_eof() {
    let mut cursor = Cursor::new(&[1, 2]);
    assert!(matches!(cursor.take(3), Err(GgufError::UnexpectedEof)));
    assert_eq!(cursor.position(), 0);
}

#[test]
fn take_with_an_overflowing_length_is_eof() {
    let mut cursor = Cursor::new(&[1, 2]);
    cursor.take(1).unwrap();
    assert!(matches!(
        cursor.take(usize::MAX),
        Err(GgufError::UnexpectedEof)
    ));
}

#[test]
fn u32_and_u64_are_little_endian() {
    let mut bytes = vec![0x01, 0x02, 0x03, 0x04];
    bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut cursor = Cursor::new(&bytes);
    assert_eq!(cursor.u32().unwrap(), 0x0403_0201);
    assert_eq!(cursor.u64().unwrap(), 0x0807_0605_0403_0201);
}

#[test]
fn len_saturates_instead_of_failing() {
    let mut bytes = u64::MAX.to_le_bytes().to_vec();
    bytes.push(0);
    let mut cursor = Cursor::new(&bytes);
    assert_eq!(cursor.len().unwrap(), usize::MAX);
    assert!(matches!(
        cursor.take(usize::MAX),
        Err(GgufError::UnexpectedEof)
    ));
}

#[test]
fn string_reads_length_prefixed_utf8() {
    let mut bytes = 3_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"abc");
    assert_eq!(Cursor::new(&bytes).string().unwrap(), "abc");
}

#[test]
fn string_replaces_invalid_utf8() {
    let mut bytes = 2_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0xff, b'a']);
    assert_eq!(Cursor::new(&bytes).string().unwrap(), "\u{fffd}a");
}

#[test]
fn string_longer_than_the_input_is_eof() {
    let mut bytes = 9_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"abc");
    assert!(matches!(
        Cursor::new(&bytes).string(),
        Err(GgufError::UnexpectedEof)
    ));
}
