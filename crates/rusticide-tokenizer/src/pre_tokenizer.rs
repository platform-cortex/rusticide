// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use fancy_regex::Regex;

use crate::error::{Result, TokenizerError};

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT2` (`tokenizer.ggml.pre` = `gpt-2`).
const GPT2: &[&str] = &[r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` (`tokenizer.ggml.pre` = `qwen2`).
const QWEN2: &[&str] = &[
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEFAULT`, used when `tokenizer.ggml.pre`
/// is absent.
const DEFAULT: &[&str] = &[
    r"[\p{P}\$\+<=>\^~\|]+",
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    r"\p{N}+",
    r"[0-9][0-9][0-9]",
];

/// Splits text into the pieces that byte-level BPE merges independently.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreTokenizer {
    patterns: &'static [&'static str],
}

impl PreTokenizer {
    pub(crate) const GPT2: Self = Self { patterns: GPT2 };

    /// `None` is a vocabulary without `tokenizer.ggml.pre`.
    pub(crate) fn from_name(name: Option<&str>) -> Result<Self> {
        let patterns = match name {
            None => DEFAULT,
            Some("gpt2" | "gpt-2") => GPT2,
            Some("qwen2") => QWEN2,
            Some(other) => return Err(TokenizerError::UnsupportedPreTokenizer(other.to_string())),
        };
        Ok(Self { patterns })
    }

    /// Applies the patterns in order, as llama.cpp's `unicode_regex_split`
    /// does: each pattern splits every current piece into its matches and
    /// the gaps between them, and the gaps are kept.
    pub(crate) fn split(self, text: &str) -> Result<Vec<&str>> {
        let mut pieces = vec![text];
        for pattern in self.patterns {
            let regex =
                Regex::new(pattern).map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
            let mut next = Vec::new();
            for piece in pieces {
                split_piece(&regex, piece, &mut next)?;
            }
            next.retain(|piece| !piece.is_empty());
            pieces = next;
        }
        Ok(pieces)
    }
}

fn split_piece<'t>(regex: &Regex, piece: &'t str, out: &mut Vec<&'t str>) -> Result<()> {
    let mut last = 0;
    for found in regex.find_iter(piece) {
        let found = found.map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
        out.push(&piece[last..found.start()]);
        out.push(found.as_str());
        last = found.end();
    }
    out.push(&piece[last..]);
    Ok(())
}

#[cfg(test)]
#[path = "pre_tokenizer_tests.rs"]
mod tests;
