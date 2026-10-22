// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Device, Tensor};

use super::*;

#[test]
fn test_greedy_returns_argmax() {
    let data = vec![-2.0, 3.0, 5.0, -1.0, 0.0];
    let logits = Tensor::from_slice(&data, &[5]).unwrap();
    let sampler = Sampler::new(42);
    let token = sampler.sample_greedy(&logits).unwrap();
    assert_eq!(token, 2);
}

#[test]
fn test_temperature_zero_falls_back_to_greedy() {
    let data: Vec<f32> = (0..100u8).map(f32::from).collect();
    let logits = Tensor::from_slice(&data, &[100]).unwrap();
    let mut sampler = Sampler::new(42);
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
    };
    let token = sampler.sample(&logits, &params).unwrap();
    assert_eq!(token, 99);
}

#[test]
fn test_all_neg_inf_logits_error() {
    let data = vec![f32::NEG_INFINITY; 10];
    let logits = Tensor::from_slice(&data, &[10]).unwrap();
    let mut sampler = Sampler::new(42);
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
    };
    let result = sampler.sample(&logits, &params);
    assert!(matches!(result, Err(SamplingError::NoValidTokens)));
}

#[test]
fn test_different_seeds_produce_different_sequences() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let logits = Tensor::from_slice(&data, &[10]).unwrap();

    let mut sampler_a = Sampler::new(42);
    let mut sampler_b = Sampler::new(999);

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };

    let mut results_a = Vec::new();
    let mut results_b = Vec::new();
    for _ in 0..20 {
        results_a.push(sampler_a.sample(&logits, &params).unwrap());
        results_b.push(sampler_b.sample(&logits, &params).unwrap());
    }

    assert_ne!(results_a, results_b);
}

#[test]
fn test_same_seed_reproduces_results() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let logits = Tensor::from_slice(&data, &[10]).unwrap();

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 123,
    };

    let mut sampler_1 = Sampler::new(123);
    let mut sampler_2 = Sampler::new(123);

    let mut results_1 = Vec::new();
    let mut results_2 = Vec::new();
    for _ in 0..20 {
        results_1.push(sampler_1.sample(&logits, &params).unwrap());
        results_2.push(sampler_2.sample(&logits, &params).unwrap());
    }

    assert_eq!(results_1, results_2);
}

#[test]
fn test_sample_with_temp_greedy_fallback() {
    let data = vec![-5.0, 100.0, -5.0];
    let logits = Tensor::from_slice(&data, &[3]).unwrap();
    let mut sampler = Sampler::new(42);
    let token = sampler.sample_with_temp(&logits, 0.0).unwrap();
    assert_eq!(token, 1);
}

#[test]
fn test_penalize_modifies_logits() {
    let mut logits = vec![1.0, 1.0, 1.0, 1.0];
    let sampler = Sampler::new(42);
    sampler.penalize(&mut logits, &[0, 2], 2.0);
    assert!((logits[0] - 0.5).abs() < 1e-5);
    assert!((logits[1] - 1.0).abs() < 1e-5);
    assert!((logits[2] - 0.5).abs() < 1e-5);
    assert!((logits[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_top_k_equals_one_is_greedy() {
    let data = vec![0.0, 100.0, 0.0];
    let logits = Tensor::from_slice(&data, &[3]).unwrap();
    let mut sampler = Sampler::new(42);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let token = sampler.sample(&logits, &params).unwrap();
    assert_eq!(token, 1);
}

#[test]
fn sample_multinomial_falls_back_to_the_last_index() {
    let mut sampler = Sampler::new(42);
    assert_eq!(sampler.sample_multinomial(&[0.0, 0.0]), 1);
}

#[test]
fn sample_with_temp_on_an_empty_vocabulary_has_no_valid_tokens() {
    let logits = Tensor::zeros(&[0], DType::F32, Device::cpu()).unwrap();
    let mut sampler = Sampler::new(42);
    let result = sampler.sample_with_temp(&logits, 1.0);
    assert!(matches!(result, Err(SamplingError::NoValidTokens)));
}

#[test]
fn top_k_one_is_greedy_even_when_the_top_logits_tie() {
    let logits = Tensor::from_slice(&[5.0f32, 5.0, 0.0], &[3]).unwrap();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let greedy = Sampler::new(7).sample_greedy(&logits).unwrap();
    let mut sampler = Sampler::new(7);
    for _ in 0..32 {
        assert_eq!(sampler.sample(&logits, &params).unwrap(), greedy);
    }
}

#[test]
fn sample_greedy_on_an_empty_vocabulary_has_no_valid_tokens() {
    let logits = Tensor::zeros(&[0], DType::F32, Device::cpu()).unwrap();
    let result = Sampler::new(42).sample_greedy(&logits);
    assert!(matches!(result, Err(SamplingError::NoValidTokens)));
}

#[test]
fn sample_on_an_empty_vocabulary_has_no_valid_tokens() {
    let logits = Tensor::zeros(&[0], DType::F32, Device::cpu()).unwrap();
    let result = Sampler::new(42).sample(&logits, &SamplingParams::default());
    assert!(matches!(result, Err(SamplingError::NoValidTokens)));
}

#[test]
fn low_temperature_concentrates_on_the_largest_logit() {
    let logits = Tensor::from_slice(&[0.0f32, 0.5, 1.0], &[3]).unwrap();
    let params = SamplingParams {
        temperature: 0.05,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let mut sampler = Sampler::new(3);
    for _ in 0..32 {
        assert_eq!(sampler.sample(&logits, &params).unwrap(), 2);
    }
}

#[test]
fn temperature_zero_is_greedy_on_every_draw() {
    let data: Vec<f32> = (0..100u8).map(f32::from).collect();
    let logits = Tensor::from_slice(&data, &[100]).unwrap();
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let mut sampler = Sampler::new(5);
    for _ in 0..32 {
        assert_eq!(sampler.sample(&logits, &params).unwrap(), 99);
    }
}

#[test]
fn sample_draws_only_from_the_top_k_tokens() {
    let logits = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4]).unwrap();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let mut sampler = Sampler::new(11);
    for _ in 0..64 {
        assert!(sampler.sample(&logits, &params).unwrap() >= 2);
    }
}

#[test]
fn sample_draws_only_from_the_top_p_nucleus() {
    let logits = Tensor::from_slice(&[1.0f32, 0.9, 0.0, 0.0], &[4]).unwrap();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.3,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let mut sampler = Sampler::new(13);
    for _ in 0..64 {
        assert_eq!(sampler.sample(&logits, &params).unwrap(), 0);
    }
}

#[test]
fn sample_with_temp_draws_from_the_scaled_distribution() {
    let logits = Tensor::from_slice(&[0.0f32, 0.5, 1.0], &[3]).unwrap();
    let mut sampler = Sampler::new(17);
    for _ in 0..32 {
        assert_eq!(sampler.sample_with_temp(&logits, 0.05).unwrap(), 2);
    }
}
