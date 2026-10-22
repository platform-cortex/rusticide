// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

use crate::error::{Result, SamplingError};

pub(crate) fn extract_logits(tensor: &Tensor) -> Result<Vec<f32>> {
    let dims = tensor.shape().dims();
    match dims {
        [_] | [1, _] => Ok(tensor.to_vec::<f32>()?),
        _ => Err(SamplingError::InvalidShape(dims.to_vec())),
    }
}

pub(crate) fn token_id(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| SamplingError::VocabOverflow)
}

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::*;
    use crate::params::SamplingParams;
    use crate::sampler::Sampler;

    #[test]
    fn test_invalid_shape_returns_error() {
        let logits = Tensor::zeros(&[2, 3, 4], DType::F32, Device::cpu()).unwrap();
        let mut sampler = Sampler::new(42);
        let params = SamplingParams::default();
        let result = sampler.sample(&logits, &params);
        assert!(matches!(result, Err(SamplingError::InvalidShape(_))));
    }

    #[test]
    fn test_2d_logits_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::from_slice(&data, &[1, 5]).unwrap();
        let sampler = Sampler::new(42);
        let token = sampler.sample_greedy(&logits).unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn f16_logits_are_a_tensor_error() {
        let logits = Tensor::zeros(&[3], DType::F16, Device::cpu()).unwrap();
        let result = extract_logits(&logits);
        assert!(matches!(
            result,
            Err(SamplingError::Tensor(
                rusticide_core::Error::DTypeMismatch {
                    expected: DType::F32,
                    got: DType::F16,
                }
            ))
        ));
    }

    #[test]
    fn invalid_shape_2d_with_leading_dim_not_one() {
        let logits = Tensor::zeros(&[2, 3], DType::F32, Device::cpu()).unwrap();
        let result = extract_logits(&logits);
        assert!(matches!(result, Err(SamplingError::InvalidShape(shape)) if shape == [2, 3]));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn token_id_overflow_is_vocab_overflow() {
        let overflowing = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        assert!(matches!(
            token_id(overflowing),
            Err(SamplingError::VocabOverflow)
        ));
    }

    #[test]
    fn token_id_keeps_the_index() {
        assert!(matches!(token_id(7), Ok(7)));
    }
}
