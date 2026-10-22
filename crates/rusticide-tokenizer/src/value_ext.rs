// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::error::{Result, TokenizerError};

pub(crate) trait GGUFValueExt {
    fn as_string(&self) -> Result<String>;
    fn as_array(&self) -> Result<&Vec<GGUFValue>>;
    fn as_f32(&self) -> Result<f32>;
    fn as_bool(&self) -> Result<bool>;
    fn as_u32_lossy(&self) -> Result<u32>;
}

impl GGUFValueExt for GGUFValue {
    fn as_string(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            _ => Err(TokenizerError::MissingMetadata(
                "expected string value type",
            )),
        }
    }

    fn as_array(&self) -> Result<&Vec<GGUFValue>> {
        match self {
            Self::Array(arr) => Ok(arr),
            _ => Err(TokenizerError::MissingMetadata("expected array value type")),
        }
    }

    fn as_f32(&self) -> Result<f32> {
        match self {
            Self::Float32(f) => Ok(*f),
            _ => Err(TokenizerError::MissingMetadata(
                "expected float32 value type",
            )),
        }
    }

    fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(b) => Ok(*b),
            _ => Err(TokenizerError::MissingMetadata("expected bool value type")),
        }
    }

    fn as_u32_lossy(&self) -> Result<u32> {
        match self {
            Self::Uint32(u) => Ok(*u),
            Self::Int32(i) => u32::try_from(*i).map_err(|_| TokenizerError::InvalidId(*i)),
            Self::Uint64(u) => {
                u32::try_from(*u).map_err(|_| TokenizerError::MissingMetadata("id exceeds u32"))
            }
            Self::Int64(i) => {
                u32::try_from(*i).map_err(|_| TokenizerError::MissingMetadata("id exceeds u32"))
            }
            _ => Err(TokenizerError::MissingMetadata(
                "expected integer id value type",
            )),
        }
    }
}

#[cfg(test)]
#[path = "value_ext_tests.rs"]
mod tests;
