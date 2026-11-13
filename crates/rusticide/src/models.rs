// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model families the facade can load.

use rusticide_model::Registry;

/// Every architecture the facade can load, keyed by `general.architecture`.
#[must_use]
pub fn registry() -> Registry {
    rusticide_llama::registry().with(rusticide_bitnet::ARCHITECTURE, rusticide_bitnet::load)
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
