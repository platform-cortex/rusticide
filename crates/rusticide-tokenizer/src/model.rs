// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// The tokenization algorithm a vocabulary was trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    LLaMA,
    GPT2,
    /// tiktoken: byte-level BPE on raw bytes, ranks as ids.
    Tiktoken,
}
