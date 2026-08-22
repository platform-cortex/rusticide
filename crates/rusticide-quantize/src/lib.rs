#![allow(

// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
)]

use half::f16;
use xllm_tensor::{Tensor, TensorError};

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;

pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4_0,
    Q4K,
    Q6K,
}

#[derive(Debug)]
pub struct QuantizationError;

impl std::fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "quantization error")
    }
}

impl std::error::Error for QuantizationError {}

#[must_use]
pub const fn bytes_for_q4_0(num_elements: usize) -> usize {
    let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
    num_blocks * Q4_0_BLOCK_BYTES
}

#[must_use]
pub const fn bytes_for_q4_k(num_elements: usize) -> usize {
    num_elements.div_ceil(Q4_K_BLOCK_SIZE) * Q4_K_BLOCK_BYTES
}

#[must_use]
pub const fn bytes_for_q6_k(num_elements: usize) -> usize {
    num_elements.div_ceil(Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES
}

fn flat_index_to_indices(idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = idx;
    for d in (0..shape.len()).rev() {
        indices[d] = remaining % shape[d];
        remaining /= shape[d];
    }
    indices
}

fn tensor_val(tensor: &Tensor, flat_idx: usize) -> f32 {
    let indices = flat_index_to_indices(flat_idx, tensor.shape());
    tensor.get::<f32>(&indices).unwrap_or(0.0)
}

pub struct Quantizer;

impl Quantizer {
    #[must_use]
    pub fn quantize_q4_0(tensor: &Tensor) -> Vec<u8> {
        let shape = tensor.shape().to_vec();
        let num_elements: usize = shape.iter().product();
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        let mut result = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];

        for block_idx in 0..num_blocks {
            let start = block_idx * Q4_0_BLOCK_SIZE;
            let end = (start + Q4_0_BLOCK_SIZE).min(num_elements);

            let mut max_abs = 0.0f32;
            for i in start..end {
                let val = tensor_val(tensor, i);
                max_abs = max_abs.max(val.abs());
            }

            if max_abs == 0.0 {
                continue;
            }

            let d = max_abs / 7.0f32;
            let d_bits = f16::from_f32(d).to_bits();
            let block_offset = block_idx * Q4_0_BLOCK_BYTES;
            result[block_offset..block_offset + 2].copy_from_slice(&d_bits.to_le_bytes());

            for i in start..end {
                let val = tensor_val(tensor, i);
                let q = ((val / d + 8.0).round() as i32).clamp(0, 15) as u8;
                let offset = i - start;
                let byte_idx = block_offset + 2 + offset / 2;
                if offset & 1 == 0 {
                    result[byte_idx] = (result[byte_idx] & 0xf0) | q;
                } else {
                    result[byte_idx] = (result[byte_idx] & 0x0f) | (q << 4);
                }
            }

            for i in end..start + Q4_0_BLOCK_SIZE {
                let offset = i - start;
                let byte_idx = block_offset + 2 + offset / 2;
                if offset & 1 == 0 {
                    result[byte_idx] = (result[byte_idx] & 0xf0) | 8;
                } else {
                    result[byte_idx] = (result[byte_idx] & 0x0f) | (8 << 4);
                }
            }
        }

        result
    }

    /// Dequantize Q4_0 block-compressed data back into a float32 tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidOperation`] if `data` is too short for
    /// the given `shape`.
    pub fn dequantize_q4_0(data: &[u8], shape: &[usize]) -> Result<Tensor, TensorError> {
        let num_elements: usize = shape.iter().product();
        let expected_size = bytes_for_q4_0(num_elements);
        if data.len() < expected_size {
            return Err(TensorError::InvalidOperation(format!(
                "Q4_0 data too short: {} bytes, need {}",
                data.len(),
                expected_size
            )));
        }

        let mut values = vec![0.0f32; num_elements];
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let start = block_idx * Q4_0_BLOCK_SIZE;
            let end = (start + Q4_0_BLOCK_SIZE).min(num_elements);
            let block_offset = block_idx * Q4_0_BLOCK_BYTES;

            let mut d_bits = [0u8; 2];
            d_bits.copy_from_slice(&data[block_offset..block_offset + 2]);
            let d = f16::from_bits(u16::from_le_bytes(d_bits)).to_f32();

            for (pos, val_ref) in values[start..end].iter_mut().enumerate() {
                let byte_idx = block_offset + 2 + pos / 2;
                let byte_val = data[byte_idx];
                let nibble = if pos & 1 == 0 {
                    byte_val & 0x0f
                } else {
                    (byte_val >> 4) & 0x0f
                };
                *val_ref = (f32::from(nibble) - 8.0) * d;
            }
        }

        Tensor::from_slice(&values, shape)
    }
}

/// ggml get_scale_min_k4: unpacks the 6-bit scale and min for sub-block `j`
/// from the 12-byte packed `scales` array.
fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantizes Q4_K data to F32. Row length must be a multiple of 256.
///
/// # Errors
///
/// Returns [`TensorError::InvalidOperation`] if `data` is too short for
/// the given `shape`.
// Keep the literal `d*q - m` form to match ggml's reference expression
// exactly for bit-fidelity parity; clippy would fuse it into an FMA.
#[allow(clippy::suboptimal_flops)]
pub fn dequantize_q4_k(data: &[u8], shape: &[usize]) -> Result<Tensor, TensorError> {
    let num_elements: usize = shape.iter().product();
    let expected = bytes_for_q4_k(num_elements);
    if data.len() < expected {
        return Err(TensorError::InvalidOperation(format!(
            "Q4_K data too short: {} bytes, need {expected}",
            data.len()
        )));
    }

    let mut values = vec![0.0f32; num_elements];
    let num_blocks = num_elements.div_ceil(Q4_K_BLOCK_SIZE);

    for blk in 0..num_blocks {
        let base = blk * Q4_K_BLOCK_BYTES;
        let mut d_bits = [0u8; 2];
        d_bits.copy_from_slice(&data[base..base + 2]);
        let d = f16::from_bits(u16::from_le_bytes(d_bits)).to_f32();
        let mut dmin_bits = [0u8; 2];
        dmin_bits.copy_from_slice(&data[base + 2..base + 4]);
        let dmin = f16::from_bits(u16::from_le_bytes(dmin_bits)).to_f32();
        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];

        let out = &mut values[blk * Q4_K_BLOCK_SIZE..blk * Q4_K_BLOCK_SIZE + 256];
        // Eight 32-element sub-blocks, processed in pairs of 64 sharing 32 qs bytes.
        for pair in 0..4 {
            let (sc0, m0) = q4_k_scale_min(pair * 2, scales);
            let (sc1, m1) = q4_k_scale_min(pair * 2 + 1, scales);
            let d1 = d * f32::from(sc0);
            let mm1 = dmin * f32::from(m0);
            let d2 = d * f32::from(sc1);
            let mm2 = dmin * f32::from(m1);
            let q = &qs[pair * 32..pair * 32 + 32];
            let base_out = pair * 64;
            for l in 0..32 {
                out[base_out + l] = d1 * f32::from(q[l] & 0x0F) - mm1;
            }
            for l in 0..32 {
                out[base_out + 32 + l] = d2 * f32::from(q[l] >> 4) - mm2;
            }
        }
    }

    Tensor::from_slice(&values, shape)
}

/// Dequantizes Q6_K data to F32. Row length must be a multiple of 256
/// (llama.cpp guarantees this for K-quant tensors).
///
/// # Errors
///
/// Returns [`TensorError::InvalidOperation`] if `data` is too short for
/// the given `shape`.
pub fn dequantize_q6_k(data: &[u8], shape: &[usize]) -> Result<Tensor, TensorError> {
    let num_elements: usize = shape.iter().product();
    let expected = bytes_for_q6_k(num_elements);
    if data.len() < expected {
        return Err(TensorError::InvalidOperation(format!(
            "Q6_K data too short: {} bytes, need {expected}",
            data.len()
        )));
    }

    let mut values = vec![0.0f32; num_elements];
    let num_blocks = num_elements.div_ceil(Q6_K_BLOCK_SIZE);

    for blk in 0..num_blocks {
        let base = blk * Q6_K_BLOCK_BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        // scales are signed 8-bit.
        let sc = &data[base + 192..base + 208];
        let mut d_bits = [0u8; 2];
        d_bits.copy_from_slice(&data[base + 208..base + 210]);
        let d = f16::from_bits(u16::from_le_bytes(d_bits)).to_f32();

        let out = &mut values[blk * Q6_K_BLOCK_SIZE..blk * Q6_K_BLOCK_SIZE + 256];
        // Two halves of 128 elements each.
        for half_idx in 0..2 {
            let ql_h = &ql[half_idx * 64..half_idx * 64 + 64];
            let qh_slice = &qh[half_idx * 32..half_idx * 32 + 32];
            let sc_h = &sc[half_idx * 8..half_idx * 8 + 8];
            let y = &mut out[half_idx * 128..half_idx * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let qh_l = qh_slice[l];
                let q1 = i32::from((ql_h[l] & 0x0F) | (((qh_l) & 3) << 4)) - 32;
                let q2 = i32::from((ql_h[l + 32] & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql_h[l] >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql_h[l + 32] >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;
                let s = |k: usize| f32::from(sc_h[k].cast_signed());
                y[l] = d * s(is) * q1 as f32;
                y[l + 32] = d * s(is + 2) * q2 as f32;
                y[l + 64] = d * s(is + 4) * q3 as f32;
                y[l + 96] = d * s(is + 6) * q4 as f32;
            }
        }
    }

    Tensor::from_slice(&values, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 0.5);
        }
    }

    #[test]
    fn test_q4_0_roundtrip_2d() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8, 8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let row = i / 8;
            let col = i % 8;
            let v: f32 = deq.get(&[row, col]).unwrap();
            assert_close(v, expected, 6.0);
        }
    }

    #[test]
    fn test_q4_0_block_boundary() {
        let data: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, &[33]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[33]).unwrap();

        assert_eq!(deq.shape(), &[33]);
        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 5.0);
        }
    }

    #[test]
    fn test_q4_0_all_zeros() {
        let data = vec![0.0f32; 64];
        let tensor = Tensor::from_slice(&data, &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8, 8]).unwrap();

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
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 1.0);
        }
    }

    #[test]
    fn test_q4_0_short_data_returns_error() {
        let result = Quantizer::dequantize_q4_0(&[0u8; 5], &[64]);
        assert!(result.is_err());
    }

    #[test]
    fn quantizer_exists() {
        let _ = Quantizer;
    }

    #[test]
    fn q6_k_all_zero_quants_dequantize_to_minus_32() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        // scales[16] occupy bytes [192..208], set each to 1 (i8).
        for b in &mut block[192..208] {
            *b = 1;
        }
        // d (f16) in the last 2 bytes [208..210].
        let d = f16::from_f32(1.0).to_bits().to_le_bytes();
        block[208] = d[0];
        block[209] = d[1];

        let out = dequantize_q6_k(&block, &[256]).unwrap();
        for i in 0..256 {
            let v: f32 = out.get(&[i]).unwrap();
            assert!((v - (-32.0)).abs() < 1e-4, "elem {i} = {v}, want -32.0");
        }
    }

    #[test]
    fn q4_k_zero_scale_yields_negative_min() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        // d (f16) at [0..2] = 0.0 (already zero).
        // dmin (f16) at [2..4] = 1.0.
        let dmin = f16::from_f32(1.0).to_bits().to_le_bytes();
        block[2] = dmin[0];
        block[3] = dmin[1];
        // scales[12] at [4..16]. get_scale_min_k4(0) reads min from scales[0+4]=scales[4].
        // Set scales[4] low 6 bits = 1 so m = 1 for the first sub-block.
        block[4 + 4] = 1;

        let out = dequantize_q4_k(&block, &[256]).unwrap();
        // First 32 elements belong to sub-block 0 -> value = d1*q - m1 = 0 - 1*1 = -1.
        for i in 0..32 {
            let v: f32 = out.get(&[i]).unwrap();
            assert!((v - (-1.0)).abs() < 1e-4, "elem {i} = {v}, want -1.0");
        }
    }
}
