// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// The tokenization algorithm a vocabulary was trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    LLaMA,
    GPT2,
}
