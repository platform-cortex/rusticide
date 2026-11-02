// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::candidates::Candidates;
use crate::error::SamplingError;
use crate::filters::{apply_temperature, softmax};
use crate::params::SamplingParams;
use crate::sampler::Sampler;

const VOCAB: usize = 1000;
const TEMPERATURES: [f32; 3] = [0.0, 0.7, 1.3];
const TOP_KS: [u32; 5] = [0, 1, 40, 1000, 5000];
const TOP_PS: [f32; 3] = [0.5, 0.95, 1.0];

fn params(temperature: f32, top_k: u32, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty: 1.0,
        seed: 0,
    }
}

fn reference_top_k(logits: &mut [f32], k: u32) {
    let k = k as usize;
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    let threshold = logits[indices[k - 1]];
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn reference_top_p(logits: &mut [f32], p: f32) {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let Some(&best) = indices.first() else {
        return;
    };
    let max_logit = logits[best];
    let exp_vals: Vec<f32> = indices
        .iter()
        .map(|&idx| (logits[idx] - max_logit).exp())
        .collect();
    let exp_sum = exp_vals.iter().fold(0.0, |sum, &v| sum + v);
    let mut cumsum = 0.0;
    let mut cutoff = indices.len();
    for (i, &ev) in exp_vals.iter().enumerate() {
        cumsum += ev / exp_sum;
        if cumsum > p {
            cutoff = i + 1;
            break;
        }
    }
    for &idx in &indices[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn reference_argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

fn reference_multinomial(rng: &mut fastrand::Rng, probs: &[f32]) -> usize {
    let r: f32 = rng.f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

fn reference_sample(
    rng: &mut fastrand::Rng,
    logits: &[f32],
    params: &SamplingParams,
) -> Result<u32, String> {
    let mut logits: Vec<f32> = logits
        .iter()
        .map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l })
        .collect();
    if params.temperature > 0.0 {
        apply_temperature(&mut logits, params.temperature);
    }
    reference_top_k(&mut logits, params.top_k);
    if params.top_p < 1.0 {
        reference_top_p(&mut logits, params.top_p);
    }
    let probs = softmax(&logits);
    let no_valid = || SamplingError::NoValidTokens.to_string();
    if probs.iter().all(|&p| p.is_nan() || p <= 0.0) {
        return Err(no_valid());
    }
    if params.temperature == 0.0 || params.top_k == 1 {
        return match reference_argmax(&logits) {
            Some(index) if logits.iter().any(|l| l.is_finite()) => {
                Ok(u32::try_from(index).unwrap())
            }
            _ => Err(no_valid()),
        };
    }
    Ok(u32::try_from(reference_multinomial(rng, &probs)).unwrap())
}

fn reference_greedy(logits: &[f32]) -> Result<u32, String> {
    let cleaned: Vec<f32> = logits
        .iter()
        .map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l })
        .collect();
    match reference_argmax(&cleaned) {
        Some(index) if cleaned.iter().any(|l| l.is_finite()) => Ok(u32::try_from(index).unwrap()),
        _ => Err(SamplingError::NoValidTokens.to_string()),
    }
}

fn random_logits(seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed ^ 0x5EED);
    let mut logits: Vec<f32> = Vec::with_capacity(VOCAB);
    for i in 0..VOCAB {
        let roll = rng.u32(0..100);
        let logit = match roll {
            0..5 => f32::NAN,
            5..10 => f32::NEG_INFINITY,
            10..20 if i > 0 => logits[rng.usize(0..i)],
            _ => f32::from(u16::try_from(rng.u32(0..2000)).unwrap()) / 100.0 - 10.0,
        };
        logits.push(logit);
    }
    if seed.is_multiple_of(50) {
        logits[rng.usize(0..VOCAB)] = f32::INFINITY;
    }
    logits
}

#[test]
fn survivors_reproduce_the_full_vocabulary_sampler() {
    for seed in 0..200_u64 {
        let logits = random_logits(seed);
        assert_eq!(
            Sampler::new(seed)
                .greedy_logits(&logits)
                .map_err(|e| e.to_string()),
            reference_greedy(&logits),
            "greedy, seed {seed}"
        );
        for temperature in TEMPERATURES {
            for top_k in TOP_KS {
                for top_p in TOP_PS {
                    let params = params(temperature, top_k, top_p);
                    let mut sampler = Sampler::new(seed);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    for draw in 0..2 {
                        let got = sampler
                            .sample_logits(&logits, &params)
                            .map_err(|e| e.to_string());
                        let want = reference_sample(&mut rng, &logits, &params);
                        assert_eq!(got, want, "seed {seed} draw {draw} {params:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn all_non_finite_logits_are_no_valid_tokens() {
    let inputs: [&[f32]; 5] = [
        &[],
        &[f32::NAN; 4],
        &[f32::NEG_INFINITY; 4],
        &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
        &[f32::INFINITY, 1.0, 2.0],
    ];
    for logits in inputs {
        for temperature in TEMPERATURES {
            for top_k in TOP_KS {
                for top_p in TOP_PS {
                    let result =
                        Sampler::new(1).sample_logits(logits, &params(temperature, top_k, top_p));
                    assert!(
                        matches!(result, Err(SamplingError::NoValidTokens)),
                        "{logits:?} t={temperature} k={top_k} p={top_p} gave {result:?}"
                    );
                }
            }
        }
    }
    for logits in &inputs[..4] {
        let result = Sampler::new(1).greedy_logits(logits);
        assert!(
            matches!(result, Err(SamplingError::NoValidTokens)),
            "{logits:?}"
        );
    }
}

#[test]
fn greedy_logits_reads_nan_as_negative_infinity_and_keeps_the_last_maximum() {
    let sampler = Sampler::new(1);
    assert_eq!(
        sampler.greedy_logits(&[f32::NAN, 0.5, f32::NAN]).unwrap(),
        1
    );
    assert_eq!(sampler.greedy_logits(&[2.0, 1.0, 2.0, 0.0]).unwrap(), 2);
}

#[test]
fn top_k_beyond_the_vocabulary_keeps_everything() {
    let logits = [3.0, f32::NAN, -1.0, f32::NEG_INFINITY, 0.5, 3.0];
    let survivors = Candidates::top_k(&logits, 1.0, 5000);
    assert_eq!(survivors.values(), [3.0, -1.0, 0.5, 3.0]);
    assert_eq!(survivors.token(3).unwrap(), 5);
    let mut beyond = Sampler::new(9);
    let mut disabled = Sampler::new(9);
    for _ in 0..32 {
        assert_eq!(
            beyond
                .sample_logits(&logits, &params(1.0, 5000, 1.0))
                .unwrap(),
            disabled
                .sample_logits(&logits, &params(1.0, 0, 1.0))
                .unwrap()
        );
    }
}

#[test]
fn top_k_keeps_every_logit_tied_with_the_kth() {
    let logits = [1.0, 4.0, 2.0, 4.0, 2.0, 0.0];
    let survivors = Candidates::top_k(&logits, 0.0, 3);
    assert_eq!(survivors.values(), [4.0, 2.0, 4.0, 2.0]);
}

#[test]
fn top_p_splits_ties_by_the_lowest_index() {
    let mut survivors = Candidates::top_k(&[5.0, 0.0, 5.0], 0.0, 0);
    survivors.top_p(0.4);
    assert_eq!(survivors.values(), [5.0]);
    assert_eq!(survivors.token(0).unwrap(), 0);
}

#[test]
fn top_p_on_no_survivors_is_a_no_op() {
    let mut survivors = Candidates::top_k(&[f32::NAN, f32::NEG_INFINITY], 1.0, 0);
    survivors.top_p(0.5);
    assert_eq!(survivors.values(), [0.0f32; 0]);
}
