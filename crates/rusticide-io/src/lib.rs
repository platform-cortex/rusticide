// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF model file reader.

mod error;
mod gguf;

pub use error::{GgufError, Result};
pub use gguf::{GGUFValue, GgmlType, GgufFile, Metadata, TensorInfo};
