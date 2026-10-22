// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::gguf::GGUFValue;

/// Failures while reading a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The file does not start with the GGUF magic bytes.
    #[error("invalid GGUF magic")]
    InvalidMagic,
    /// The GGUF version is not the supported version 3.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    /// The file could not be read from disk.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A tensor uses a GGML type this reader does not support.
    #[error("unsupported tensor type: {0}")]
    UnsupportedTensorType(u32),
    /// A computed offset does not fit the platform word size.
    #[error("{0} overflows")]
    CastOverflow(&'static str),
    /// `general.alignment` is not a non-zero power-of-two `u32`.
    #[error("invalid general.alignment: {0:?}")]
    InvalidAlignment(GGUFValue),
    /// Two tensor infos share one name.
    #[error("duplicate tensor name: {name}")]
    DuplicateTensor {
        /// The repeated tensor name.
        name: String,
    },
    /// A tensor declares more dimensions than GGML supports.
    #[error("tensor declares {got} dimensions, at most {max} are supported")]
    TooManyDimensions {
        /// The declared dimension count.
        got: u32,
        /// The maximum supported dimension count.
        max: u32,
    },
    /// A tensor's element count does not fit in `usize`.
    #[error("tensor {name} element count overflows")]
    ShapeOverflow {
        /// The offending tensor name.
        name: String,
    },
    /// A tensor offset is not a multiple of the file's alignment.
    #[error("tensor {name} offset {offset} is not a multiple of alignment {alignment}")]
    MisalignedTensor {
        /// The offending tensor name.
        name: String,
        /// The declared offset.
        offset: u64,
        /// The file's tensor-data alignment.
        alignment: u64,
    },
    /// The input ended before a read could complete.
    #[error("unexpected end of file")]
    UnexpectedEof,
    /// No tensor matches the requested name or index.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    /// Building the tensor failed.
    #[error(transparent)]
    Tensor(#[from] rusticide_core::Error),
    /// Dequantizing the tensor data failed.
    #[error(transparent)]
    Quantize(#[from] rusticide_quantize::QuantizeError),
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
