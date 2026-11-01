// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Model;
use crate::error::{ModelError, Result};

/// Records every forward and fails at position `fail_at`.
struct Recorder {
    forwards: Vec<(u32, usize)>,
    fail_at: usize,
}

impl Model for Recorder {
    fn vocab_size(&self) -> usize {
        2
    }

    fn context_length(&self) -> usize {
        8
    }

    fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        self.forwards.push((token, position));
        if position == self.fail_at {
            return Err(ModelError::ContextFull { limit: position });
        }
        Ok(vec![f32::from(u8::try_from(token).unwrap()), 0.5])
    }

    fn reset(&mut self) {
        self.forwards.clear();
    }
}

fn recorder(fail_at: usize) -> Recorder {
    Recorder {
        forwards: Vec::new(),
        fail_at,
    }
}

#[test]
fn forward_batch_runs_each_token_in_order_and_returns_the_last_logits() {
    let mut model = recorder(usize::MAX);
    assert_eq!(model.forward_batch(&[4, 6, 3], 2).unwrap(), [3.0, 0.5]);
    assert_eq!(model.forwards, [(4, 2), (6, 3), (3, 4)]);
}

#[test]
fn forward_batch_stops_at_the_first_error() {
    let mut model = recorder(1);
    match model.forward_batch(&[4, 6, 3], 0) {
        Err(ModelError::ContextFull { limit }) => assert_eq!(limit, 1),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(model.forwards, [(4, 0), (6, 1)]);
}

#[test]
fn forward_batch_of_no_tokens_is_invalid_config() {
    let mut model = recorder(usize::MAX);
    match model.forward_batch(&[], 0) {
        Err(ModelError::InvalidConfig { reason }) => assert_eq!(reason, "no tokens to forward"),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(model.forwards, []);
    model.reset();
    assert_eq!((model.vocab_size(), model.context_length()), (2, 8));
}
