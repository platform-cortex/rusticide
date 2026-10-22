// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use super::value::GGUFValue;
use crate::error::{GgufError, Result};

const DEFAULT_ALIGNMENT: u64 = 32;
const ALIGNMENT_KEY: &str = "general.alignment";

/// GGUF key-value metadata with typed accessors for common model settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata(HashMap<String, GGUFValue>);

impl FromIterator<(String, GGUFValue)> for Metadata {
    fn from_iter<I: IntoIterator<Item = (String, GGUFValue)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Metadata {
    /// Looks up a raw metadata value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GGUFValue> {
        self.0.get(key)
    }

    /// Number of metadata keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no metadata keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The model architecture (`general.architecture`).
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        match self.get("general.architecture")? {
            GGUFValue::String(name) => Some(name),
            _ => None,
        }
    }

    /// Maximum context length.
    #[must_use]
    pub fn context_length(&self) -> Option<u64> {
        self.arch_unsigned("context_length")
    }

    /// Embedding (hidden) size.
    #[must_use]
    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_unsigned("embedding_length")
    }

    /// Number of transformer blocks.
    #[must_use]
    pub fn block_count(&self) -> Option<u64> {
        self.arch_unsigned("block_count")
    }

    /// Number of attention heads.
    #[must_use]
    pub fn head_count(&self) -> Option<u64> {
        self.arch_unsigned("attention.head_count")
    }

    /// Number of key-value attention heads.
    #[must_use]
    pub fn head_count_kv(&self) -> Option<u64> {
        self.arch_unsigned("attention.head_count_kv")
    }

    /// Feed-forward intermediate size.
    #[must_use]
    pub fn feed_forward_length(&self) -> Option<u64> {
        self.arch_unsigned("feed_forward_length")
    }

    /// `RoPE` base frequency.
    #[must_use]
    pub fn rope_freq_base(&self) -> Option<f32> {
        self.arch_float("rope.freq_base")
    }

    /// RMS norm epsilon.
    #[must_use]
    pub fn rms_norm_eps(&self) -> Option<f32> {
        self.arch_float("attention.layer_norm_rms_epsilon")
    }

    /// The tensor-data alignment: 32 when `general.alignment` is absent.
    ///
    /// # Errors
    ///
    /// `InvalidAlignment` when the key is present but is not a non-zero
    /// power-of-two `Uint32`.
    pub(crate) fn alignment(&self) -> Result<u64> {
        match self.get(ALIGNMENT_KEY) {
            None => Ok(DEFAULT_ALIGNMENT),
            Some(GGUFValue::Uint32(value)) if value.is_power_of_two() => Ok(u64::from(*value)),
            Some(other) => Err(GgufError::InvalidAlignment(other.clone())),
        }
    }

    fn arch_value(&self, suffix: &str) -> Option<&GGUFValue> {
        self.get(&format!("{}.{suffix}", self.architecture()?))
    }

    fn arch_unsigned(&self, suffix: &str) -> Option<u64> {
        match *self.arch_value(suffix)? {
            GGUFValue::Uint64(n) => Some(n),
            GGUFValue::Uint32(n) => Some(u64::from(n)),
            GGUFValue::Int32(n) => u64::try_from(n).ok(),
            GGUFValue::Int64(n) => u64::try_from(n).ok(),
            _ => None,
        }
    }

    #[expect(clippy::cast_possible_truncation, reason = "narrowing is the contract")]
    fn arch_float(&self, suffix: &str) -> Option<f32> {
        match *self.arch_value(suffix)? {
            GGUFValue::Float32(n) => Some(n),
            GGUFValue::Float64(n) => Some(n as f32),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
