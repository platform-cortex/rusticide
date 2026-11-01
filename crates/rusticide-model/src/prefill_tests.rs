// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PREFILL_CHUNK, prefill};
use crate::error::{ModelError, Result};
use crate::model::Model;

/// Records every batch and answers with its start position as the logits.
#[derive(Default)]
struct Batches {
    batches: Vec<(Vec<u32>, usize)>,
    resets: usize,
}

impl Model for Batches {
    fn vocab_size(&self) -> usize {
        1
    }

    fn context_length(&self) -> usize {
        1024
    }

    fn forward(&mut self, _: u32, _: usize) -> Result<Vec<f32>> {
        Err(ModelError::InvalidConfig {
            reason: "prefill must forward in batches",
        })
    }

    fn forward_batch(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        self.batches.push((tokens.to_vec(), start));
        Ok(vec![f32::from(u16::try_from(start).unwrap())])
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

#[test]
fn prefill_forwards_the_prompt_in_chunks() {
    assert_eq!(PREFILL_CHUNK, 64);
    let prompt: Vec<u32> = (0..150).collect();
    let mut model = Batches::default();
    assert_eq!(prefill(&mut model, &prompt).unwrap(), [128.0]);
    let expected: Vec<(Vec<u32>, usize)> = [(0, 64), (64, 128), (128, 150)]
        .into_iter()
        .map(|(start, end)| ((start..end).collect(), usize::try_from(start).unwrap()))
        .collect();
    assert_eq!(model.batches, expected);
    assert_eq!(model.resets, 1);
    assert_eq!((model.vocab_size(), model.context_length()), (1, 1024));
    match model.forward(0, 0) {
        Err(ModelError::InvalidConfig { reason }) => {
            assert_eq!(reason, "prefill must forward in batches");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn prefill_of_no_tokens_is_invalid_config() {
    let mut model = Batches::default();
    match prefill(&mut model, &[]) {
        Err(ModelError::InvalidConfig { reason }) => {
            assert_eq!(reason, "prompt encodes to no tokens");
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(model.batches, []);
}
