// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q5_0`: 32 elements per block, an f16 scale, 32 high bits, then 16 packed nibbles.

use crate::half_bits::read_f16_le;

pub(crate) const Q5_0_BLOCK_SIZE: usize = 32;
pub(crate) const Q5_0_BLOCK_BYTES: usize = 22;

const HALF_BLOCK: usize = Q5_0_BLOCK_SIZE / 2;

fn fifth_bit(high_bits: u32, bit: usize) -> u8 {
    u8::from((high_bits >> bit) & 1 != 0) << 4
}

pub(crate) fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let scale = read_f16_le(block, 0);
    let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for (index, &packed) in block[6..].iter().enumerate() {
        let low = (packed & 0x0F) | fifth_bit(high_bits, index);
        let high = (packed >> 4) | fifth_bit(high_bits, index + HALF_BLOCK);
        out[index] = scale * (f32::from(low) - 16.0);
        out[index + HALF_BLOCK] = scale * (f32::from(high) - 16.0);
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{QuantizeError, block_format};

    fn block(high_bits: u32, packed: &[u8; 16]) -> Vec<u8> {
        let mut bytes = f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        bytes.extend_from_slice(&high_bits.to_le_bytes());
        bytes.extend_from_slice(packed);
        bytes
    }

    fn dequantize(data: &[u8], shape: &[usize]) -> Result<Vec<f32>, QuantizeError> {
        let tensor = block_format(6).unwrap().dequantize(data, shape)?;
        Ok(tensor.to_vec::<f32>().unwrap())
    }

    #[test]
    fn q5_0_without_high_bits_is_nibble_minus_16() {
        let mut packed = [0u8; 16];
        packed[0] = 0x21;
        let values = dequantize(&block(0, &packed), &[32]).unwrap();
        assert_eq!(values[0], -15.0);
        assert_eq!(values[16], -14.0);
    }

    #[test]
    fn q5_0_high_bits_add_sixteen() {
        let values = dequantize(&block(1 | (1 << 16), &[0u8; 16]), &[32]).unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[16], 0.0);
        assert_eq!(values[1], -16.0);
    }

    #[test]
    fn q5_0_uses_every_high_bit() {
        let values = dequantize(&block(u32::MAX, &[0xFF; 16]), &[32]).unwrap();
        assert!(values.iter().all(|&v| (v - 15.0).abs() < 1e-6));
    }

    #[test]
    fn q5_0_rejects_a_partial_block() {
        let err = dequantize(&[0u8; 22], &[31]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: "Q5_0",
                numel: 31,
                block: 32
            }
        ));
    }
}
