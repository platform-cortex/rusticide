// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF format layer.

// Consumers (header, metadata, tensor loading) land in later tasks.
#![cfg_attr(not(test), allow(dead_code))]

mod cursor;
mod value;

pub use value::GGUFValue;
