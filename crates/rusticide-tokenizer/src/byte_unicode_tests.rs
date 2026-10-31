// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{byte_to_char, char_to_byte};

#[test]
fn printable_bytes_map_to_themselves() {
    assert_eq!(byte_to_char(b'A'), 'A');
    assert_eq!(byte_to_char(0xE9), '\u{e9}');
}

#[test]
fn space_and_newline_are_shifted() {
    assert_eq!(byte_to_char(b' '), '\u{120}');
    assert_eq!(byte_to_char(b'\n'), '\u{10a}');
    assert_eq!(byte_to_char(0x00), '\u{100}');
    assert_eq!(byte_to_char(0xAD), '\u{143}');
}

#[test]
fn every_byte_round_trips() {
    for byte in 0..=255u8 {
        assert_eq!(char_to_byte(byte_to_char(byte)), Some(byte));
    }
}

#[test]
fn unmapped_char_has_no_byte() {
    assert_eq!(char_to_byte('\u{2581}'), None);
    assert_eq!(char_to_byte('\u{144}'), None);
    assert_eq!(char_to_byte(' '), None);
}
