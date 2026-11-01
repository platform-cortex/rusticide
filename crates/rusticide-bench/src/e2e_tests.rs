// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use rusticide_model::{Model, ModelError};

use super::{DECODE_TOKENS, e2e_metrics};
use crate::{BenchError, Plan};

const PLAN: Plan = Plan {
    runs: 2,
    budget: Duration::from_millis(1),
};

struct FakeModel {
    resets: usize,
    forwards: usize,
    fail: bool,
}

impl FakeModel {
    const fn new(fail: bool) -> Self {
        Self {
            resets: 0,
            forwards: 0,
            fail,
        }
    }
}

impl Model for FakeModel {
    fn vocab_size(&self) -> usize {
        4
    }

    fn context_length(&self) -> usize {
        4096
    }

    fn forward(&mut self, _token: u32, _position: usize) -> Result<Vec<f32>, ModelError> {
        if self.fail {
            return Err(ModelError::InvalidConfig { reason: "boom" });
        }
        self.forwards += 1;
        Ok(vec![0.0, 1.0, 0.5, 0.0])
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

#[test]
fn e2e_metrics_time_decode_and_prefill_on_a_model() {
    let mut model = FakeModel::new(false);
    let metrics = e2e_metrics("fake", &mut model, &[1, 2], &[1, 2, 3], &PLAN).unwrap();
    let keys: Vec<&str> = metrics.keys().map(String::as_str).collect();
    assert_eq!(keys, ["e2e/fake/decode", "e2e/fake/prefill"]);
    assert!(
        metrics
            .values()
            .all(|s| s.median.is_finite() && s.median > 0.0)
    );
    assert_eq!(model.resets, 2 * (PLAN.runs + 1));
    assert_eq!(model.forwards, (PLAN.runs + 1) * (2 + DECODE_TOKENS + 3));
}

#[test]
fn e2e_metrics_propagate_model_errors() {
    let mut model = FakeModel::new(true);
    let result = e2e_metrics("fake", &mut model, &[1], &[1], &PLAN);
    assert!(matches!(result, Err(BenchError::Model(_))));
}

#[test]
fn the_fake_model_reports_its_shape() {
    let model = FakeModel::new(false);
    assert_eq!((model.vocab_size(), model.context_length()), (4, 4096));
}
