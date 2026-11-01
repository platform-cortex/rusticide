// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::Registry;

use crate::model::{load, load_qwen2};

/// The one place model families are registered.
#[must_use]
pub fn registry() -> Registry {
    Registry::new()
        .with("llama", load)
        .with("qwen2", load_qwen2)
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
