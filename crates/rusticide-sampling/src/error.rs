// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
    #[error("tensor error: {0}")]
    Tensor(#[from] rusticide_core::Error),
    #[error("logits tensor has wrong shape: expected 1D or 2D, got {0:?}")]
    InvalidShape(Vec<usize>),
    #[error("all logits are -inf or NaN after filtering")]
    NoValidTokens,
    #[error("vocabulary size exceeds u32 range")]
    VocabOverflow,
}

pub type Result<T> = std::result::Result<T, SamplingError>;
