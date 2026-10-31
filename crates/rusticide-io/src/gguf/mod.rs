// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF format layer.

mod cursor;
mod file;
mod header;
mod load;
mod metadata;
mod tensor_info;
mod value;

#[cfg(test)]
mod test_support;

pub use file::GgufFile;
pub use metadata::Metadata;
pub use tensor_info::TensorInfo;
pub use value::GGUFValue;
