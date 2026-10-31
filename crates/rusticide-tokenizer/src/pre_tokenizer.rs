// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use regex_automata::Input;
use regex_automata::meta::Regex;

use crate::error::{Result, TokenizerError};

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT2` (`tokenizer.ggml.pre` = `gpt-2`).
pub(crate) const GPT2: &[&str] =
    &[r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` (`tokenizer.ggml.pre` = `qwen2`).
pub(crate) const QWEN2: &[&str] = &[
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEFAULT`, used when `tokenizer.ggml.pre`
/// is absent.
pub(crate) const DEFAULT: &[&str] = &[
    r"[\p{P}\$\+<=>\^~\|]+",
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    r"\p{N}+",
    r"[0-9][0-9][0-9]",
];

/// A pattern list and its regexes, compiled on first use and then shared:
/// compiling the Unicode classes costs far more than one split.
#[derive(Debug)]
pub(crate) struct PatternSet {
    patterns: &'static [&'static str],
    compiled: OnceLock<std::result::Result<Vec<Splitter>, String>>,
}

impl PatternSet {
    pub(crate) const fn new(patterns: &'static [&'static str]) -> Self {
        Self {
            patterns,
            compiled: OnceLock::new(),
        }
    }

    fn regexes(&self) -> Result<&[Splitter]> {
        self.compiled
            .get_or_init(|| self.patterns.iter().map(|p| Splitter::new(p)).collect())
            .as_deref()
            .map_err(|error| TokenizerError::EncodeError(error.clone()))
    }
}

static GPT2_SET: PatternSet = PatternSet::new(GPT2);
static QWEN2_SET: PatternSet = PatternSet::new(QWEN2);
static DEFAULT_SET: PatternSet = PatternSet::new(DEFAULT);

/// Splits text into the pieces that byte-level BPE merges independently.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreTokenizer {
    pub(crate) set: &'static PatternSet,
}

impl PreTokenizer {
    pub(crate) const GPT2: Self = Self { set: &GPT2_SET };

    /// `None` is a vocabulary without `tokenizer.ggml.pre`.
    pub(crate) fn from_name(name: Option<&str>) -> Result<Self> {
        let set = match name {
            None => &DEFAULT_SET,
            Some("gpt2" | "gpt-2") => &GPT2_SET,
            Some("qwen2") => &QWEN2_SET,
            Some(other) => return Err(TokenizerError::UnsupportedPreTokenizer(other.to_string())),
        };
        Ok(Self { set })
    }

    pub(crate) fn regexes(self) -> Result<&'static [Splitter]> {
        self.set.regexes()
    }

    /// Applies the patterns in order, as llama.cpp's `unicode_regex_split`
    /// does: each pattern splits every current piece into its matches and
    /// the gaps between them, and the gaps are kept.
    pub(crate) fn split(self, text: &str) -> Result<Vec<&str>> {
        let mut pieces = vec![text];
        for regex in self.regexes()? {
            let mut next = Vec::new();
            for piece in pieces {
                regex.split_piece(piece, &mut next);
            }
            next.retain(|piece| !piece.is_empty());
            pieces = next;
        }
        Ok(pieces)
    }
}

/// The only look-ahead in llama.cpp's patterns. `regex-automata` has no
/// look-around, so it is matched as `\s+` and its effect is re-applied by
/// [`Splitter::end_of`]: far faster than a backtracking engine.
const SPACE_RUN_LOOKAHEAD: &str = r"\s+(?!\S)";
/// qwen2's earlier whitespace-only alternative; its matches end in a line
/// break and never come from the look-ahead alternative.
const LINE_BREAK_RUN: &str = r"\s*[\r\n]+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookahead {
    Absent,
    Always,
    UnlessLineBreak,
}

/// One compiled pattern plus how to replay its space-run look-ahead.
#[derive(Debug)]
pub(crate) struct Splitter {
    regex: Regex,
    lookahead: Lookahead,
}

impl Splitter {
    fn new(pattern: &str) -> std::result::Result<Self, String> {
        let lookahead = if !pattern.contains(SPACE_RUN_LOOKAHEAD) {
            Lookahead::Absent
        } else if pattern.contains(LINE_BREAK_RUN) {
            Lookahead::UnlessLineBreak
        } else {
            Lookahead::Always
        };
        let regex =
            Regex::new(&pattern.replace(SPACE_RUN_LOOKAHEAD, r"\s+")).map_err(|e| e.to_string())?;
        Ok(Self { regex, lookahead })
    }

    fn split_piece<'t>(&self, piece: &'t str, out: &mut Vec<&'t str>) {
        let mut last = 0;
        let mut input = Input::new(piece);
        while let Some(found) = self.regex.search(&input) {
            let end = self.end_of(piece, found.start(), found.end());
            out.push(&piece[last..found.start()]);
            out.push(&piece[found.start()..end]);
            last = end;
            input.set_start(end);
        }
        out.push(&piece[last..]);
    }

    /// `\s+(?!\S)`: a space run followed by a non-space character gives
    /// its last character back, unless that would leave the match empty.
    fn end_of(&self, piece: &str, start: usize, end: usize) -> usize {
        let matched = &piece[start..end];
        let before_text = piece[end..]
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace());
        let from_lookahead = match self.lookahead {
            Lookahead::Absent => false,
            Lookahead::Always => true,
            Lookahead::UnlessLineBreak => !matched.ends_with(['\r', '\n']),
        };
        let give_back = matched.chars().next_back().map_or(0, char::len_utf8);
        let keeps_a_character = give_back < matched.len();
        if from_lookahead
            && before_text
            && keeps_a_character
            && matched.chars().all(char::is_whitespace)
        {
            end - give_back
        } else {
            end
        }
    }
}

#[cfg(test)]
#[path = "pre_tokenizer_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "pre_tokenizer_equivalence_tests.rs"]
mod equivalence_tests;
