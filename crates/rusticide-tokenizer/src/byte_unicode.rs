// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPT-2's `bytes_to_unicode`: every byte gets one printable character.

use std::collections::HashMap;

const SHIFTED: [char; 68] = [
    '\u{100}', '\u{101}', '\u{102}', '\u{103}', '\u{104}', '\u{105}', '\u{106}', '\u{107}',
    '\u{108}', '\u{109}', '\u{10a}', '\u{10b}', '\u{10c}', '\u{10d}', '\u{10e}', '\u{10f}',
    '\u{110}', '\u{111}', '\u{112}', '\u{113}', '\u{114}', '\u{115}', '\u{116}', '\u{117}',
    '\u{118}', '\u{119}', '\u{11a}', '\u{11b}', '\u{11c}', '\u{11d}', '\u{11e}', '\u{11f}',
    '\u{120}', '\u{121}', '\u{122}', '\u{123}', '\u{124}', '\u{125}', '\u{126}', '\u{127}',
    '\u{128}', '\u{129}', '\u{12a}', '\u{12b}', '\u{12c}', '\u{12d}', '\u{12e}', '\u{12f}',
    '\u{130}', '\u{131}', '\u{132}', '\u{133}', '\u{134}', '\u{135}', '\u{136}', '\u{137}',
    '\u{138}', '\u{139}', '\u{13a}', '\u{13b}', '\u{13c}', '\u{13d}', '\u{13e}', '\u{13f}',
    '\u{140}', '\u{141}', '\u{142}', '\u{143}',
];
const SHIFT_BASE: u32 = 0x100;

const fn is_printable(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// Position of a non-printable byte among the 68 non-printable bytes.
const fn shift_index(byte: u8) -> usize {
    match byte {
        0x00..=0x20 => byte as usize,
        0x7F..=0xA0 => byte as usize - 0x7F + 33,
        _ => 67,
    }
}

pub(crate) const fn byte_to_char(byte: u8) -> char {
    if is_printable(byte) {
        byte as char
    } else {
        SHIFTED[shift_index(byte)]
    }
}

pub(crate) fn char_to_byte(c: char) -> Option<u8> {
    let code = u32::from(c);
    if let Ok(byte) = u8::try_from(code) {
        return is_printable(byte).then_some(byte);
    }
    let index = usize::try_from(code.checked_sub(SHIFT_BASE)?).ok()?;
    let byte = match index {
        0..=32 => index,
        33..=66 => index - 33 + 0x7F,
        67 => 0xAD,
        _ => return None,
    };
    u8::try_from(byte).ok()
}

/// The id of the one-character token spelling each byte, when the vocabulary
/// has it (the lowest id wins if a spelling repeats).
pub(crate) fn byte_token_table(vocab: &[Vec<u8>]) -> [Option<u32>; 256] {
    let mut ids: HashMap<&[u8], u32> = HashMap::new();
    for (index, piece) in vocab.iter().enumerate() {
        if let Ok(id) = u32::try_from(index) {
            ids.entry(piece).or_insert(id);
        }
    }
    let mut table = [None; 256];
    for byte in 0..=255u8 {
        let spelling = byte_to_char(byte)
            .encode_utf8(&mut [0; 4])
            .as_bytes()
            .to_vec();
        table[usize::from(byte)] = ids.get(spelling.as_slice()).copied();
    }
    table
}

#[cfg(test)]
#[path = "byte_unicode_tests.rs"]
mod tests;
