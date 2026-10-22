// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Q4_K`: 256 elements per block, f16 `d` and `dmin`, 12 packed 6-bit
//! scale/min bytes and 128 bytes of nibbles.

use rusticide_core::{Shape, Tensor};

use crate::error::{QuantizationType, QuantizeError, ensure_aligned, ensure_len};
use crate::half_bits::read_f16_le;

pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;

const SUB_BLOCK_SIZE: usize = 32;

#[must_use]
pub const fn bytes_for_q4_k(num_elements: usize) -> usize {
    num_elements.div_ceil(Q4_K_BLOCK_SIZE) * Q4_K_BLOCK_BYTES
}

/// ggml `get_scale_min_k4`: unpacks the 6-bit scale and min for sub-block `j`
/// from the 12-byte packed `scales` array.
const fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the literal `d*q - m` form matches ggml's reference expression bit for bit; \
              a fused multiply-add would round differently"
)]
fn dequantize_sub_block(out: &mut [f32], quants: impl Iterator<Item = u8>, scale: f32, min: f32) {
    for (value, quant) in out.iter_mut().zip(quants) {
        *value = scale * f32::from(quant) - min;
    }
}

fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let dmin = read_f16_le(block, 2);
    let scales = &block[4..16];
    let quants = &block[16..Q4_K_BLOCK_BYTES];

    // Eight 32-element sub-blocks, processed in pairs sharing 32 quant bytes.
    for (pair, (pair_out, pair_quants)) in out
        .chunks_mut(2 * SUB_BLOCK_SIZE)
        .zip(quants.chunks(SUB_BLOCK_SIZE))
        .enumerate()
    {
        let (low_out, high_out) = pair_out.split_at_mut(SUB_BLOCK_SIZE);
        let (low_scale, low_min) = q4_k_scale_min(pair * 2, scales);
        let (high_scale, high_min) = q4_k_scale_min(pair * 2 + 1, scales);
        dequantize_sub_block(
            low_out,
            pair_quants.iter().map(|q| q & 0x0F),
            d * f32::from(low_scale),
            dmin * f32::from(low_min),
        );
        dequantize_sub_block(
            high_out,
            pair_quants.iter().map(|q| q >> 4),
            d * f32::from(high_scale),
            dmin * f32::from(high_min),
        );
    }
}

/// Dequantizes `Q4_K` data to F32. Row length must be a multiple of 256.
///
/// # Errors
///
/// Returns [`QuantizeError::UnalignedShape`] if the element count is not a
/// multiple of 256 (checked first, so an overflowing shape lands here), or
/// [`QuantizeError::DataTooShort`] if `data` is too short for the `shape`.
pub fn dequantize_q4_k(data: &[u8], shape: &[usize]) -> Result<Tensor, QuantizeError> {
    let num_elements = Shape::new(shape).numel();
    ensure_aligned(QuantizationType::Q4K, num_elements, Q4_K_BLOCK_SIZE)?;
    ensure_len(QuantizationType::Q4K, data, bytes_for_q4_k(num_elements))?;

    let mut values = vec![0.0f32; num_elements];
    for (block, out) in data
        .chunks(Q4_K_BLOCK_BYTES)
        .zip(values.chunks_mut(Q4_K_BLOCK_SIZE))
    {
        dequantize_block(block, out);
    }
    Ok(Tensor::from_slice(&values, shape)?)
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    fn set_f16(block: &mut [u8], offset: usize, value: f32) {
        block[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }

    #[test]
    fn q4_k_zero_scale_yields_negative_min() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        set_f16(&mut block, 2, 1.0);
        block[4 + 4] = 1;

        let out = dequantize_q4_k(&block, &[256]).unwrap();
        for i in 0..32 {
            let v: f32 = out.get(&[i]).unwrap();
            assert!((v - (-1.0)).abs() < 1e-4, "elem {i} = {v}, want -1.0");
        }
    }

    #[test]
    fn q4_k_full_block_uses_every_sub_block_pair() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        set_f16(&mut block, 0, 1.0);
        set_f16(&mut block, 2, 1.0);
        // scales 1..=8; mins 8..=1 so a scale/min swap changes the output.
        block[4..8].copy_from_slice(&[1, 2, 3, 4]);
        block[8..12].copy_from_slice(&[8, 7, 6, 5]);
        block[12..16].copy_from_slice(&[0x45, 0x36, 0x27, 0x18]);
        block[16..].fill(0x32);

        let out = dequantize_q4_k(&block, &[256]).unwrap();
        for sub_block in 0..8u8 {
            let scale = f32::from(sub_block) + 1.0;
            let quant = if sub_block % 2 == 0 { 2.0 } else { 3.0 };
            let min = 8.0 - f32::from(sub_block);
            let want = scale.mul_add(quant, -min);
            for l in [0usize, 31] {
                let v: f32 = out.get(&[usize::from(sub_block) * 32 + l]).unwrap();
                assert!(
                    (v - want).abs() < 1e-4,
                    "sub-block {sub_block} = {v}, want {want}"
                );
            }
        }
    }

    #[test]
    fn q4_k_scale_min_upper_sub_blocks() {
        let mut scales = [0u8; 12];
        scales[0] = 0xC0;
        scales[4] = 0x40;
        scales[8] = 0xA5;
        assert_eq!(q4_k_scale_min(4, &scales), (53, 26));
        assert_eq!(q4_k_scale_min(0, &scales), (0, 0));
    }

    #[test]
    fn dequantize_q4_k_short_data() {
        let err = dequantize_q4_k(&[0u8; 143], &[256]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: QuantizationType::Q4K,
                got: 143,
                needed: 144
            }
        ));
    }

    #[test]
    fn bytes_for_q4_k_rounds_up_to_whole_blocks() {
        assert_eq!(bytes_for_q4_k(0), 0);
        assert_eq!(bytes_for_q4_k(1), 144);
        assert_eq!(bytes_for_q4_k(256), 144);
        assert_eq!(bytes_for_q4_k(257), 288);
    }

    #[test]
    fn dequantize_q4_k_rejects_unaligned_shapes() {
        let err = dequantize_q4_k(&[0u8; 144], &[1]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: QuantizationType::Q4K,
                numel: 1,
                block: 256
            }
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn dequantize_q4_k_overflowing_shape_is_unaligned() {
        let err = dequantize_q4_k(&[], &[1 << 32, 1 << 32]).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::UnalignedShape {
                format: QuantizationType::Q4K,
                numel: usize::MAX,
                block: 256
            }
        ));
    }
}
