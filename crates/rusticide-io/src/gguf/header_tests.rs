// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Header;
use crate::error::GgufError;
use crate::gguf::cursor::Cursor;
use crate::gguf::test_support::header_bytes;

fn parse(bytes: &[u8]) -> crate::error::Result<Header> {
    Header::parse(&mut Cursor::new(bytes))
}

#[test]
fn valid_header_parses() {
    let header = parse(&header_bytes(3, 7, 9)).unwrap();
    assert_eq!(header.tensor_count, 7);
    assert_eq!(header.kv_count, 9);
}

#[test]
fn invalid_magic() {
    let mut bytes = header_bytes(3, 0, 0);
    bytes[..4].copy_from_slice(b"GGUX");
    assert!(matches!(parse(&bytes), Err(GgufError::InvalidMagic)));
}

#[test]
fn unsupported_version() {
    assert!(matches!(
        parse(&header_bytes(2, 0, 0)),
        Err(GgufError::UnsupportedVersion(2))
    ));
}

#[test]
fn a_newer_version_is_unsupported_too() {
    assert!(matches!(
        parse(&header_bytes(4, 0, 0)),
        Err(GgufError::UnsupportedVersion(4))
    ));
}

#[test]
fn truncated_header() {
    let bytes = header_bytes(3, 0, 0);
    assert!(matches!(parse(&bytes[..23]), Err(GgufError::UnexpectedEof)));
}

#[test]
fn a_short_file_with_bad_magic_is_eof_first() {
    assert!(matches!(parse(b"XXXX"), Err(GgufError::UnexpectedEof)));
}
