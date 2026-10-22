// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cursor::Cursor;
use crate::error::{GgufError, Result};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const HEADER_BYTES: usize = 24;

/// The fixed-size GGUF header counts, both untrusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) tensor_count: u64,
    pub(crate) kv_count: u64,
}

impl Header {
    pub(crate) fn parse(cursor: &mut Cursor<'_>) -> Result<Self> {
        if cursor.remaining() < HEADER_BYTES {
            return Err(GgufError::UnexpectedEof);
        }
        if cursor.array::<4>()? != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }
        let version = cursor.u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        Ok(Self {
            tensor_count: cursor.u64()?,
            kv_count: cursor.u64()?,
        })
    }
}

#[cfg(test)]
#[path = "header_tests.rs"]
mod tests;
