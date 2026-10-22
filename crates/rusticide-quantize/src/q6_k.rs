// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q6_K`: 256 elements per block, 128 low-nibble bytes, 64 high-bit bytes,
//! 16 signed scales and an f16 `d`.

use rusticide_core::{Shape, Tensor};

use crate::error::{QuantizationType, QuantizeError, ensure_aligned, ensure_len};
use crate::half_bits::read_f16_le;

pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

#[must_use]
pub const fn bytes_for_q6_k(num_elements: usize) -> usize {
    num_elements.div_ceil(Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES
}

fn six_bit(low: u8, high_bits: u8) -> f32 {
    f32::from(i16::from(low | ((high_bits & 3) << 4)) - 32)
}

fn dequantize_half(low: &[u8], high: &[u8], scales: &[u8], d: f32, out: &mut [f32]) {
    for l in 0..32 {
        let scale_group = l / 16;
        let scale = |k: usize| f32::from(scales[scale_group + k].cast_signed());
        let q1 = six_bit(low[l] & 0x0F, high[l]);
        let q2 = six_bit(low[l + 32] & 0x0F, high[l] >> 2);
        let q3 = six_bit(low[l] >> 4, high[l] >> 4);
        let q4 = six_bit(low[l + 32] >> 4, high[l] >> 6);
        out[l] = d * scale(0) * q1;
        out[l + 32] = d * scale(2) * q2;
        out[l + 64] = d * scale(4) * q3;
        out[l + 96] = d * scale(6) * q4;
    }
}

fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 208);
    for (half_index, half_out) in out.chunks_mut(128).enumerate() {
        dequantize_half(
            &block[half_index * 64..half_index * 64 + 64],
            &block[128 + half_index * 32..128 + half_index * 32 + 32],
            &block[192 + half_index * 8..192 + half_index * 8 + 8],
            d,
            half_out,
        );
    }
}

/// Dequantizes `Q6_K` data to F32. Row length must be a multiple of 256
/// (llama.cpp guarantees this for K-quant tensors).
///
/// # Errors
///
/// Returns [`QuantizeError::UnalignedShape`] if the element count is not a
/// multiple of 256 (checked first, so an overflowing shape lands here), or
/// [`QuantizeError::DataTooShort`] if `data` is too short for the `shape`.
pub fn dequantize_q6_k(data: &[u8], shape: &[usize]) -> Result<Tensor, QuantizeError> {
    let num_elements = Shape::new(shape).numel();
    ensure_aligned(QuantizationType::Q6K, num_elements, Q6_K_BLOCK_SIZE)?;
    ensure_len(QuantizationType::Q6K, data, bytes_for_q6_k(num_elements))?;

    let mut values = vec![0.0f32; num_elements];
    for (block, out) in data
        .chunks(Q6_K_BLOCK_BYTES)
        .zip(values.chunks_mut(Q6_K_BLOCK_SIZE))
    {
        dequantize_block(block, out);
    }
    Ok(Tensor::from_slice(&values, shape)?)
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn q6_k_all_zero_quants_dequantize_to_minus_32() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        block[192..208].fill(1);
        block[208..210].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());

        let out = dequantize_q6_k(&block, &[256]).unwrap();
        for i in 0..256 {
            let v: f32 = out.get(&[i]).unwrap();
            assert!((v - (-32.0)).abs() < 1e-4, "elem {i} = {v}, want -32.0");
        }
    }

    #[test]
    fn dequantize_q6_k_short_data() {
        let err = dequantize_q6_k(&[0u8; 209], &[256]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: QuantizationType::Q6K,
                got: 209,
                needed: 210
            }
        ));
    }

    #[test]
    fn bytes_for_q6_k_rounds_up_to_whole_blocks() {
        assert_eq!(bytes_for_q6_k(0), 0);
        assert_eq!(bytes_for_q6_k(1), 210);
        assert_eq!(bytes_for_q6_k(256), 210);
        assert_eq!(bytes_for_q6_k(257), 420);
    }

    #[test]
    fn q6_k_high_bits_scales_and_second_half_are_decoded() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        // element 0: low nibble 5, high bits 2 -> q = 5 | 32 - 32 = 5.
        block[0] = 0x05;
        block[128] = 0b10;
        // element 32 (q2): low nibble 0, high bits 1 -> 16 - 32 = -16.
        block[128] |= 0b01 << 2;
        // second half, element 128: low nibble 3 -> 3 - 32 = -29, scale 2.
        block[64] = 0x03;
        block[128 + 32] = 0b10;
        block[192 + 8] = 2;
        block[192] = 3;
        block[192 + 2] = 4;
        block[208..210].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());

        let out = dequantize_q6_k(&block, &[256]).unwrap();
        let at = |i: usize| -> f32 { out.get(&[i]).unwrap() };
        assert!((at(0) - 15.0).abs() < 1e-4, "got {}", at(0));
        assert!((at(32) - -64.0).abs() < 1e-4, "got {}", at(32));
        assert!((at(128) - 6.0).abs() < 1e-4, "got {}", at(128));
    }

    #[test]
    fn dequantize_q6_k_rejects_unaligned_shapes() {
        let err = dequantize_q6_k(&[0u8; 210], &[1]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: QuantizationType::Q6K,
                numel: 1,
                block: 256
            }
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn dequantize_q6_k_overflowing_shape_is_unaligned() {
        let err = dequantize_q6_k(&[], &[1 << 32, 1 << 32]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: QuantizationType::Q6K,
                numel: usize::MAX,
                block: 256
            }
        ));
    }

    #[test]
    fn q6_k_every_quadrant_scale_and_scale_group_is_decoded() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        let scales: Vec<i8> = (0..16).map(|i| if i == 5 { -6 } else { i + 1 }).collect();
        for (slot, scale) in block[192..208].iter_mut().zip(&scales) {
            *slot = scale.cast_unsigned();
        }
        block[208..210].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for half_index in 0..2 {
            for l in [0usize, 16] {
                block[half_index * 64 + l] = 0x21;
                block[half_index * 64 + l + 32] = 0x43;
                block[128 + half_index * 32 + l] = 0b1110_0100;
            }
        }

        let out = dequantize_q6_k(&block, &[256]).unwrap();
        for half_index in 0..2usize {
            for l in [0usize, 16] {
                let group = half_index * 8 + l / 16;
                let at = |quadrant: usize| -> f32 {
                    out.get(&[half_index * 128 + l + 32 * quadrant]).unwrap()
                };
                let want = |offset: usize, q: f32| f32::from(scales[group + offset]) * q;
                assert!((at(0) - want(0, -31.0)).abs() < 1e-4);
                assert!((at(1) - want(2, -13.0)).abs() < 1e-4);
                assert!((at(2) - want(4, 2.0)).abs() < 1e-4);
                assert!((at(3) - want(6, 20.0)).abs() < 1e-4);
            }
        }
    }
}
