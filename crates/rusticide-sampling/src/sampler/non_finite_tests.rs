// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

use super::*;

const NON_FINITE: [[f32; 3]; 3] = [
    [f32::NAN; 3],
    [f32::NEG_INFINITY; 3],
    [f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
];

#[test]
fn sample_greedy_rejects_logits_without_a_finite_value() {
    for data in NON_FINITE {
        let logits = Tensor::from_slice(&data, &[3]).unwrap();
        let result = Sampler::new(1).sample_greedy(&logits);
        assert!(
            matches!(result, Err(SamplingError::NoValidTokens)),
            "{data:?} gave {result:?}"
        );
    }
}

#[test]
fn sample_with_zero_temperature_rejects_logits_without_a_finite_value() {
    for data in NON_FINITE {
        let logits = Tensor::from_slice(&data, &[3]).unwrap();
        let result = Sampler::new(1).sample_with_temp(&logits, 0.0);
        assert!(
            matches!(result, Err(SamplingError::NoValidTokens)),
            "{data:?} gave {result:?}"
        );
    }
}

#[test]
fn one_finite_logit_among_non_finite_ones_is_still_greedy_choosable() {
    let data = [f32::NEG_INFINITY, 0.5, f32::NEG_INFINITY];
    let logits = Tensor::from_slice(&data, &[3]).unwrap();
    assert_eq!(Sampler::new(1).sample_greedy(&logits).unwrap(), 1);
}
