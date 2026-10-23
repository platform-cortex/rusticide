// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::Result;

/// A loaded language model that turns one token at a time into logits.
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
    fn reset(&mut self);
}
