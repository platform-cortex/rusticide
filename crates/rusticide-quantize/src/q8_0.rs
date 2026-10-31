// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q8_0`: 32 elements per block, an f16 scale then 32 signed bytes.

use crate::half_bits::read_f16_le;

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub(crate) fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let scale = read_f16_le(block, 0);
    for (value, &quant) in out.iter_mut().zip(&block[2..]) {
        *value = scale * f32::from(i8::from_le_bytes([quant]));
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{QuantizeError, block_format};

    fn dequantize(data: &[u8], shape: &[usize]) -> Result<Vec<f32>, QuantizeError> {
        let tensor = block_format(8).unwrap().dequantize(data, shape)?;
        Ok(tensor.to_vec::<f32>().unwrap())
    }

    #[test]
    fn q8_0_scales_signed_bytes() {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        for (slot, value) in block[2..7].iter_mut().zip([-128i8, -1, 0, 1, 127]) {
            *slot = value.to_le_bytes()[0];
        }
        let values = dequantize(&block, &[32]).unwrap();
        assert_eq!(&values[..5], &[-64.0, -0.5, 0.0, 0.5, 63.5]);
        assert!(values[5..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_rejects_short_data() {
        let err = dequantize(&[0u8; 33], &[32]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: "Q8_0",
                got: 33,
                needed: 34
            }
        ));
    }
}
