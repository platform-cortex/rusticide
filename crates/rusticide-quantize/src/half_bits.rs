// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Little-endian f16 block scale reader.

use half::f16;

/// Reads the two bytes at `offset` as a little-endian f16.
///
/// Callers must guarantee `offset + 2 <= data.len()`; every block reader
/// checks the total length first.
pub(crate) fn read_f16_le(data: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_at_an_offset() {
        let bytes = f16::from_f32(-2.5).to_bits().to_le_bytes();
        assert!((read_f16_le(&[0xff, bytes[0], bytes[1]], 1) - -2.5).abs() < 1e-6);
    }
}
