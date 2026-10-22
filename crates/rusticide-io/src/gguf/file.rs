// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;

use super::cursor::Cursor;
use super::header::Header;
use super::metadata::Metadata;
use super::tensor_info::{TensorInfo, parse_tensor_info};
use super::value::parse_typed_value;
use crate::error::{GgufError, Result};

/// A parsed GGUF v3 file: metadata, tensor infos and the raw bytes.
///
/// Memory: metadata arrays are expanded into `GGUFValue`s, costing roughly
/// 24-32 bytes per element in memory regardless of the element's size on
/// disk. A hostile file can therefore amplify its size; a memory ceiling is
/// deferred to SP2.
#[derive(Debug)]
pub struct GgufFile {
    metadata: Metadata,
    tensors: HashMap<String, TensorInfo>,
    tensor_names: Vec<String>,
    #[cfg_attr(not(test), expect(dead_code, reason = "read by tensor loading"))]
    data_start: usize,
    #[cfg_attr(not(test), expect(dead_code, reason = "read by tensor loading"))]
    data: Vec<u8>,
}

impl GgufFile {
    /// Parses a GGUF file held in memory.
    ///
    /// # Errors
    ///
    /// Any malformed, truncated or unsupported input yields a `GgufError`.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut cursor = Cursor::new(&data);
        let header = Header::parse(&mut cursor)?;
        let mut entries = Vec::new();
        for _ in 0..header.kv_count {
            entries.push((cursor.string()?, parse_typed_value(&mut cursor)?));
        }
        let metadata: Metadata = entries.into_iter().collect();
        let mut tensors = HashMap::new();
        for _ in 0..header.tensor_count {
            let info = parse_tensor_info(&mut cursor)?;
            match tensors.entry(info.name.clone()) {
                Entry::Occupied(_) => return Err(GgufError::DuplicateTensor { name: info.name }),
                Entry::Vacant(slot) => slot.insert(info),
            };
        }
        let data_start = align_up(cursor.position(), metadata.alignment()?)?;
        let mut tensor_names: Vec<String> = tensors.keys().cloned().collect();
        tensor_names.sort();
        Ok(Self {
            metadata,
            tensors,
            tensor_names,
            data_start,
            data,
        })
    }

    /// Reads and parses a GGUF file from disk.
    ///
    /// # Errors
    ///
    /// `Io` when the file cannot be read, otherwise as [`Self::from_bytes`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// The key-value metadata.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Looks up a tensor info by name.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Tensor names in sorted order.
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Rounds `position` up to a multiple of the power-of-two `alignment`.
fn align_up(position: usize, alignment: u64) -> Result<usize> {
    let mask = usize::try_from(alignment).unwrap_or(usize::MAX) - 1;
    let padded = position
        .checked_add(mask)
        .ok_or(GgufError::CastOverflow("tensor data start"))?;
    Ok(padded & !mask)
}

#[cfg(test)]
#[path = "file_tests.rs"]
mod tests;
