// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use crate::error::{Result, SamplingError};
use crate::logits::token_id;
use crate::params::SamplingParams;

pub(crate) struct Candidates {
    entries: Vec<(usize, f32)>,
}

impl Candidates {
    pub(crate) fn filter(logits: &[f32], params: &SamplingParams) -> Self {
        let mut candidates = Self::top_k(logits, params.temperature, params.top_k);
        if params.top_p < 1.0 {
            candidates.top_p(params.top_p);
        }
        candidates
    }

    pub(crate) fn top_k(logits: &[f32], temperature: f32, k: u32) -> Self {
        let scale = if temperature > 0.0 {
            1.0 / temperature
        } else {
            1.0
        };
        let scaled = |&logit: &f32| {
            let logit = if logit.is_nan() {
                f32::NEG_INFINITY
            } else {
                logit
            };
            logit * scale
        };
        let k = k as usize;
        let threshold = if k == 0 || k >= logits.len() {
            f32::NEG_INFINITY
        } else {
            kth_largest(logits.iter().map(scaled).collect(), k)
        };
        let entries = logits
            .iter()
            .map(scaled)
            .enumerate()
            .filter(|&(_, value)| {
                value != f32::NEG_INFINITY && value.partial_cmp(&threshold) != Some(Ordering::Less)
            })
            .collect();
        Self { entries }
    }

    pub(crate) fn top_p(&mut self, p: f32) {
        self.entries
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let Some(&(_, max)) = self.entries.first() else {
            return;
        };
        let exp_vals: Vec<f32> = self
            .entries
            .iter()
            .map(|&(_, value)| (value - max).exp())
            .collect();
        let exp_sum = exp_vals.iter().fold(0.0, |sum, &value| sum + value);
        let mut cumsum = 0.0;
        let mut cutoff = exp_vals.len();
        for (i, &value) in exp_vals.iter().enumerate() {
            cumsum += value / exp_sum;
            if cumsum > p {
                cutoff = i + 1;
                break;
            }
        }
        self.entries.truncate(cutoff);
        self.entries.sort_unstable_by_key(|&(index, _)| index);
    }

    pub(crate) fn values(&self) -> Vec<f32> {
        self.entries.iter().map(|&(_, value)| value).collect()
    }

    pub(crate) fn token(&self, position: usize) -> Result<u32> {
        let &(index, _) = self
            .entries
            .get(position)
            .ok_or(SamplingError::NoValidTokens)?;
        token_id(index)
    }
}

fn kth_largest(mut values: Vec<f32>, k: usize) -> f32 {
    *values
        .select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a))
        .1
}
