// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{ModelError, Result};

/// Per-layer key/value history, appended one position at a time.
#[derive(Debug)]
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    rows: Vec<usize>,
    width: usize,
    capacity: usize,
}

impl KvCache {
    /// Storage grows on [`Self::store`]; `capacity` is only the position limit.
    #[must_use]
    pub fn new(layers: usize, width: usize, capacity: usize) -> Self {
        Self {
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
            rows: vec![0; layers],
            width,
            capacity,
        }
    }

    /// # Errors
    ///
    /// `InvalidConfig` for an unknown layer or a wrong key/value width;
    /// `ContextFull` at or past the capacity; `CachePosition` unless
    /// `position` is the next free position of the layer.
    pub fn store(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let expected = *self.rows.get(layer).ok_or(ModelError::InvalidConfig {
            reason: "layer out of range",
        })?;
        if key.len() != self.width || value.len() != self.width {
            return Err(ModelError::InvalidConfig {
                reason: "key or value width mismatch",
            });
        }
        if position >= self.capacity {
            return Err(ModelError::ContextFull {
                limit: self.capacity,
            });
        }
        if position != expected {
            return Err(ModelError::CachePosition {
                expected,
                got: position,
            });
        }
        self.keys[layer].extend_from_slice(key);
        self.values[layer].extend_from_slice(value);
        self.rows[layer] += 1;
        Ok(())
    }

    /// All stored key rows of `layer`; empty for an unknown layer.
    #[must_use]
    pub fn keys(&self, layer: usize) -> &[f32] {
        self.keys.get(layer).map_or(&[], Vec::as_slice)
    }

    /// All stored value rows of `layer`; empty for an unknown layer.
    #[must_use]
    pub fn values(&self, layer: usize) -> &[f32] {
        self.values.get(layer).map_or(&[], Vec::as_slice)
    }

    pub fn clear(&mut self) {
        self.keys.iter_mut().for_each(Vec::clear);
        self.values.iter_mut().for_each(Vec::clear);
        self.rows.fill(0);
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
