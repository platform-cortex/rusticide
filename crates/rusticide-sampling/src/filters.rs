// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

pub(crate) fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}

pub(crate) fn apply_top_k(logits: &mut [f32], k: u32) {
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

pub(crate) fn apply_top_p(logits: &mut [f32], p: f32) {
    let n = logits.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    let Some(&best) = indices.first() else {
        return;
    };
    let max_logit = logits[best];
    let mut exp_vals = Vec::with_capacity(n);
    let mut exp_sum = 0.0;
    for &idx in &indices {
        let v = (logits[idx] - max_logit).exp();
        exp_vals.push(v);
        exp_sum += v;
    }

    let mut cumsum = 0.0;
    let mut cutoff = n;
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

pub(crate) fn apply_repetition_penalty(logits: &mut [f32], tokens: &[u32], penalty: f32) {
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

pub(crate) fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_filtering() {
        let mut logits = vec![-5.0, -3.0, 10.0, 8.0, 4.0, -1.0, 2.0];
        apply_top_k(&mut logits, 3);
        let non_inf = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(non_inf, 3);
    }

    #[test]
    fn test_top_k_with_k_greater_than_len() {
        let data = vec![-5.0, -3.0, 10.0, 8.0];
        let mut logits = data.clone();
        apply_top_k(&mut logits, 100);
        assert_eq!(logits, data);
    }

    #[test]
    fn test_top_p_nucleus_filtering() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut logits = data;
        apply_top_p(&mut logits, 0.5);
        let non_inf = logits.iter().filter(|&&v| v.is_finite()).count();
        assert!(non_inf < 5);
        assert!(non_inf >= 1);
    }

    #[test]
    fn test_top_p_disabled_at_one() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut logits = data.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, data);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![-2.0, 3.0, 5.0, -1.0, 0.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_all_inf() {
        let logits = vec![f32::NEG_INFINITY; 5];
        let probs = softmax(&logits);
        assert!(probs.iter().all(|&p| p.is_nan()));
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert_eq!(probs, Vec::<f32>::new());
    }

    #[test]
    fn test_repetition_penalty_penalizes_specified_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[2], 2.0);
        assert!((logits[2] - 1.5).abs() < 1e-5);
        assert!((logits[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-1.0, 2.0, -3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - (-2.0)).abs() < 1e-5);
        assert!((logits[2] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[0.0]), Some(0));
        assert_eq!(
            argmax(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]),
            Some(1)
        );
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn repetition_penalty_ignores_tokens_outside_the_vocabulary() {
        let mut logits = vec![1.0, 2.0];
        apply_repetition_penalty(&mut logits, &[5, u32::MAX], 2.0);
        assert_eq!(logits, [1.0, 2.0]);
    }

    #[test]
    fn softmax_skips_normalisation_when_the_sum_is_not_finite() {
        let probs = softmax(&[f32::NAN, 1.0]);
        assert!(probs[0].is_nan());
        assert!((probs[1] - 1.0).abs() < 1e-6);
    }
}
