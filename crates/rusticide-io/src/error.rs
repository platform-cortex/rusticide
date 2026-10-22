// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failures while reading a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The input ended before a read could complete.
    #[error("unexpected end of file")]
    UnexpectedEof,
    /// A value type tag outside the GGUF specification.
    #[error("invalid value type: {0}")]
    InvalidValueType(u32),
    /// Arrays were nested deeper than the parser allows.
    #[error("GGUF arrays nested deeper than {limit}")]
    NestingTooDeep {
        /// The maximum permitted nesting depth.
        limit: usize,
    },
}

/// Result alias for GGUF reading.
pub type Result<T> = std::result::Result<T, GgufError>;
