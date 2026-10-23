// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GgufError;
use rusticide_sampling::SamplingError;
use rusticide_tokenizer::TokenizerError;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("gguf error: {0}")]
    Gguf(#[from] GgufError),
    #[error("tensor error: {0}")]
    Tensor(#[from] rusticide_core::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("sampling error: {0}")]
    Sampling(#[from] SamplingError),
    #[error("missing tensor {name}")]
    MissingTensor { name: String },
    #[error("missing metadata key {key}")]
    MissingMetadata { key: String },
    #[error("invalid config: {reason}")]
    InvalidConfig { reason: &'static str },
    #[error("tensor {name} has shape {got:?}, expected {expected:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("unknown architecture {name}; supported: {supported}")]
    UnknownArchitecture { name: String, supported: String },
    #[error("context is full: limit is {limit} tokens")]
    ContextFull { limit: usize },
    #[error("cache position {got} is out of order, expected {expected}")]
    CachePosition { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use super::ModelError;

    #[test]
    fn unknown_architecture_lists_the_supported_ones() {
        let error = ModelError::UnknownArchitecture {
            name: "gpt".to_owned(),
            supported: "llama".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "unknown architecture gpt; supported: llama"
        );
    }
}
