// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("token {0} out of range (vocab size: {1})")]
    TokenOutOfRange(u32, usize),
    #[error("unknown model type: {0}")]
    UnknownModel(String),
    #[error("missing required tokenizer metadata: {0}")]
    MissingMetadata(&'static str),
    #[error("cannot encode: {0}")]
    EncodeError(String),
    #[error("cannot decode: {0}")]
    DecodeError(String),
    #[error("invalid token ID (negative): {0}")]
    InvalidId(i32),
    #[error("non-finite score at vocabulary index {index}")]
    NonFiniteScore { index: usize },
    #[error("token at vocabulary index {index} is {len} bytes, over the {max}-byte limit")]
    TokenTooLong {
        index: usize,
        len: usize,
        max: usize,
    },
    #[error("gpt2 vocabulary has no token for byte 0x{byte:02X}")]
    MissingByteToken { byte: u8 },
    #[error("unsupported pre-tokenizer: {0}")]
    UnsupportedPreTokenizer(String),
    #[error("vocabulary size exceeds u32 range")]
    VocabOverflow,
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
