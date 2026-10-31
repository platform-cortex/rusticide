// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q4_0`: 32 elements per block, an f16 scale then 16 packed nibbles.

use half::f16;
use rusticide_core::Tensor;

use crate::error::QuantizeError;
use crate::half_bits::read_f16_le;

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

const ZERO_NIBBLE: u8 = 8;

#[must_use]
pub const fn bytes_for_q4_0(num_elements: usize) -> usize {
    num_elements.div_ceil(Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_BYTES
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0.0..=15.0 first, so it fits a nibble"
)]
fn quantize_nibble(value: f32, scale: f32) -> u8 {
    (value / scale + 8.0).round().clamp(0.0, 15.0) as u8
}

/// ggml layout: element `j` is the low nibble of byte `j % 16`, and element
/// `j + 16` is the high nibble of the same byte.
const HALF_BLOCK: usize = Q4_0_BLOCK_SIZE / 2;

const fn set_nibble(byte: &mut u8, offset: usize, nibble: u8) {
    if offset < HALF_BLOCK {
        *byte = (*byte & 0xf0) | nibble;
    } else {
        *byte = (*byte & 0x0f) | (nibble << 4);
    }
}

const fn get_nibble(byte: u8, offset: usize) -> u8 {
    if offset < HALF_BLOCK {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

fn quantize_block(values: &[f32], block: &mut [u8]) {
    let max_abs = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    if max_abs == 0.0 {
        return;
    }
    let scale = max_abs / 7.0;
    block[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
    for offset in 0..Q4_0_BLOCK_SIZE {
        let nibble = values
            .get(offset)
            .map_or(ZERO_NIBBLE, |&v| quantize_nibble(v, scale));
        set_nibble(&mut block[2 + offset % HALF_BLOCK], offset, nibble);
    }
}

/// `Q4_0` quantizer entry points.
pub struct Quantizer;

impl Quantizer {
    /// Quantize an F32 tensor into `Q4_0` blocks.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Tensor`] if the tensor is not F32.
    pub fn quantize_q4_0(tensor: &Tensor) -> Result<Vec<u8>, QuantizeError> {
        let values = tensor.to_vec::<f32>()?;
        let mut result = vec![0u8; bytes_for_q4_0(values.len())];
        for (chunk, block) in values
            .chunks(Q4_0_BLOCK_SIZE)
            .zip(result.chunks_mut(Q4_0_BLOCK_BYTES))
        {
            quantize_block(chunk, block);
        }
        Ok(result)
    }
}

pub(crate) fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let scale = read_f16_le(block, 0);
    for (pos, value) in out.iter_mut().enumerate() {
        let nibble = get_nibble(block[2 + pos % HALF_BLOCK], pos);
        *value = (f32::from(nibble) - f32::from(ZERO_NIBBLE)) * scale;
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use rusticide_core::{DType, Error};

    use super::*;
    use crate::block_format;

    fn dequantize_q4_0(data: &[u8], shape: &[usize]) -> Result<Tensor, QuantizeError> {
        block_format(2).unwrap().dequantize(data, shape)
    }

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() <= epsilon,
            "expected {a} to be close to {b} (epsilon {epsilon})"
        );
    }

    #[test]
    fn test_q4_0_roundtrip_simple() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0, -2.0, 3.0, -3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        let deq = dequantize_q4_0(&q, &[32]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 0.5);
        }
    }

    #[test]
    fn test_q4_0_roundtrip_2d() {
        let data: Vec<f32> = (0..64u8).map(f32::from).collect();
        let tensor = Tensor::from_slice(&data, &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        let deq = dequantize_q4_0(&q, &[8, 8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i / 8, i % 8]).unwrap();
            assert_close(v, expected, 6.0);
        }
    }

    #[test]
    fn test_q4_0_block_boundary() {
        let data: Vec<f32> = (0..33u8).map(f32::from).collect();
        let tensor = Tensor::from_slice(&data, &[33]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        let deq = dequantize_q4_0(&q, &[64]).unwrap();

        assert_eq!(deq.shape().dims(), &[64]);
        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 5.0);
        }
    }

    #[test]
    fn test_q4_0_all_zeros() {
        let tensor = Tensor::from_slice(&[0.0f32; 64], &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        let deq = dequantize_q4_0(&q, &[8, 8]).unwrap();

        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = deq.get(&[i, j]).unwrap();
                assert_close(v, 0.0, 1e-6);
            }
        }
    }

    #[test]
    fn test_q4_0_byte_size() {
        assert_eq!(bytes_for_q4_0(1), 18);
        assert_eq!(bytes_for_q4_0(32), 18);
        assert_eq!(bytes_for_q4_0(33), 36);
        assert_eq!(bytes_for_q4_0(64), 36);
        assert_eq!(bytes_for_q4_0(65), 54);
    }

    #[test]
    fn test_q4_0_negative_values() {
        let data: Vec<f32> = vec![-8.0, -5.0, -1.0, 0.0, 1.0, 5.0, 8.0, -3.0];
        let tensor = Tensor::from_slice(&data, &[8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        let deq = dequantize_q4_0(&q, &[32]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 1.0);
        }
    }

    #[test]
    fn test_q4_0_short_data_returns_error() {
        let err = dequantize_q4_0(&[0u8; 5], &[64]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: "Q4_0",
                got: 5,
                needed: 36
            }
        ));
    }

    #[test]
    fn quantize_q4_0_rejects_a_non_f32_tensor() {
        let tensor = Tensor::from_slice(&[f16::from_f32(1.0); 4], &[4]).unwrap();
        let err = Quantizer::quantize_q4_0(&tensor).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::Tensor(Error::DTypeMismatch {
                expected: DType::F32,
                got: DType::F16
            })
        ));
    }

    #[test]
    fn partial_block_is_padded_with_the_zero_nibble() {
        let tensor = Tensor::from_slice(&[7.0f32, -7.0, 7.0], &[3]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor).unwrap();
        assert_eq!(&q[..2], &[0x00, 0x3c]);
        assert_eq!(&q[2..5], &[0x8f, 0x81, 0x8f]);
        assert!(q[5..].iter().all(|&b| b == 0x88));
    }

    #[test]
    fn dequantize_q4_0_matches_the_ggml_layout() {
        let mut block = vec![0x88u8; Q4_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2] = 0x1f;
        let deq = dequantize_q4_0(&block, &[32]).unwrap();
        let at = |i: usize| -> f32 { deq.get(&[i]).unwrap() };
        assert!((at(0) - 7.0).abs() < 1e-6);
        assert!((at(1) - 0.0).abs() < 1e-6);
        assert!((at(16) - -7.0).abs() < 1e-6);
        assert!((at(17) - 0.0).abs() < 1e-6);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn dequantize_q4_0_overflowing_shape_is_unaligned() {
        let err = dequantize_q4_0(&[], &[1 << 32, 1 << 32]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: "Q4_0",
                numel: usize::MAX,
                block: 32
            }
        ));
    }
}
