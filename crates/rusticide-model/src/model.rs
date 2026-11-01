// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{ModelError, Result};

/// A loaded language model that turns tokens into logits.
pub trait Model {
    fn vocab_size(&self) -> usize;
    fn context_length(&self) -> usize;
    /// Logits for the token after `token` at `position`; positions must
    /// arrive in order from 0 after `reset`.
    ///
    /// # Errors
    ///
    /// Implementation-defined; typically a full context or a bad token.
    fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>>;
    /// Runs `tokens` at positions `start..` and returns the logits after the
    /// last one. The default forwards one token at a time.
    ///
    /// # Errors
    ///
    /// `InvalidConfig` for no tokens, else the first error of a token.
    fn forward_batch(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let mut last = Err(ModelError::InvalidConfig {
            reason: "no tokens to forward",
        });
        for (offset, &token) in tokens.iter().enumerate() {
            last = Ok(self.forward(token, start.saturating_add(offset))?);
        }
        last
    }
    fn reset(&mut self);
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
