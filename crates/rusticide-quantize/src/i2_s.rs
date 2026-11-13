// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `I2_S`: `BitNet` b1.58 ternary weights, 2 bits each, 128 per 32-byte
//! block, then one f32 scale for the whole tensor in a 32-byte trailer.

pub(crate) const I2_S_BLOCK_SIZE: usize = 128;
pub(crate) const I2_S_BLOCK_BYTES: usize = 32;
pub(crate) const I2_S_TRAILER_BYTES: usize = 32;

/// Byte `k` holds weights `k`, `k + 32`, `k + 64` and `k + 96`, high bits
/// first. Code `c` decodes to `c - 1`; ternary weights never use code 3, and
/// a hostile one decodes to `+2` rather than failing.
pub(crate) fn dequantize_block(block: &[u8], out: &mut [f32]) {
    for (lane, shift) in out.chunks_mut(I2_S_BLOCK_BYTES).zip([6, 4, 2, 0]) {
        for (value, &byte) in lane.iter_mut().zip(block) {
            *value = f32::from((byte >> shift) & 0b11) - 1.0;
        }
    }
}

/// The `I2_S` tensor scale: the little-endian f32 that opens the trailer.
/// `None` when the trailer is shorter than an f32.
#[must_use]
pub fn i2_s_scale(trailer: &[u8]) -> Option<f32> {
    trailer
        .first_chunk()
        .map(|bytes| f32::from_le_bytes(*bytes))
}

/// Multiplies dequantized `I2_S` codes by the tensor's weight scale.
pub fn i2_s_apply_scale(values: &mut [f32], scale: f32) {
    for value in values {
        *value *= scale;
    }
}

pub(crate) fn apply_trailer(trailer: &[u8], values: &mut [f32]) {
    if let Some(scale) = i2_s_scale(trailer) {
        i2_s_apply_scale(values, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantizeError, block_format};

    const FORMAT_ID: u32 = 36;

    fn code(index: usize) -> u8 {
        u8::try_from(index % 3).unwrap()
    }

    /// Byte `k` packs the codes of weights `k`, `k+32`, `k+64`, `k+96`.
    fn packed(code: impl Fn(usize) -> u8) -> Vec<u8> {
        (0..32)
            .map(|k| code(k) << 6 | code(k + 32) << 4 | code(k + 64) << 2 | code(k + 96))
            .collect()
    }

    fn trailer(scale: f32) -> Vec<u8> {
        let mut trailer = scale.to_le_bytes().to_vec();
        trailer.extend([0xA5; 28]);
        trailer
    }

    fn dequantize(data: &[u8], shape: &[usize]) -> Result<Vec<f32>, QuantizeError> {
        let tensor = block_format(FORMAT_ID).unwrap().dequantize(data, shape)?;
        Ok(tensor.to_vec::<f32>().unwrap())
    }

    #[test]
    fn the_table_holds_i2_s() {
        let format = block_format(FORMAT_ID).unwrap();
        assert_eq!(
            (
                format.name(),
                format.block_size(),
                format.block_bytes(),
                format.trailer_bytes()
            ),
            ("I2_S", 128, 32, 32)
        );
    }

    #[test]
    fn byte_k_holds_weights_k_to_k_plus_96_high_bits_first() {
        let mut out = [f32::NAN; 128];
        dequantize_block(&packed(code), &mut out);
        let expected: Vec<f32> = (0..128).map(|i| f32::from(code(i)) - 1.0).collect();
        assert_eq!(out.to_vec(), expected);
    }

    #[test]
    fn code_three_decodes_to_plus_two() {
        let mut block = [0_u8; 32];
        block[0] = 0b00_01_10_11;
        let mut out = [f32::NAN; 128];
        dequantize_block(&block, &mut out);
        assert_eq!([out[0], out[32], out[64], out[96]], [-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn byte_len_adds_one_trailer_per_tensor() {
        let format = block_format(FORMAT_ID).unwrap();
        assert_eq!(format.byte_len(6_553_600), Some(1_638_432));
        assert_eq!(format.byte_len(0), Some(32));
    }

    #[test]
    fn a_row_has_no_trailer() {
        assert_eq!(block_format(FORMAT_ID).unwrap().row_bytes(2560), Some(640));
    }

    #[test]
    fn a_tensor_one_byte_short_is_too_short() {
        let err = dequantize(&[0; 63], &[128]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: "I2_S",
                got: 63,
                needed: 64
            }
        ));
    }

    #[test]
    fn a_partial_block_is_unaligned() {
        let err = dequantize(&[0; 64], &[100]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: "I2_S",
                numel: 100,
                block: 128
            }
        ));
    }

    #[test]
    fn dequantize_applies_the_trailer_scale() {
        let mut data = packed(code);
        data.extend(packed(|_| 2));
        data.extend(trailer(1.5));
        let values = dequantize(&data, &[2, 128]).unwrap();
        let expected: Vec<f32> = (0..128)
            .map(|i| (f32::from(code(i)) - 1.0) * 1.5)
            .chain([1.5; 128])
            .collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn dequantize_row_leaves_the_codes_unscaled() {
        let mut out = [f32::NAN; 128];
        block_format(FORMAT_ID)
            .unwrap()
            .dequantize_row(&packed(|_| 0), &mut out);
        assert_eq!(out, [-1.0; 128]);
    }

    #[test]
    fn the_scale_is_the_leading_f32_of_the_trailer() {
        assert_eq!(i2_s_scale(&trailer(1.218_854_8)), Some(1.218_854_8));
        assert_eq!(i2_s_scale(&[0; 3]), None);
    }

    #[test]
    fn a_trailer_without_a_scale_leaves_the_values() {
        let mut values = [1.0, -1.0];
        apply_trailer(&[0; 3], &mut values);
        assert_eq!(values, [1.0, -1.0]);
    }
}
