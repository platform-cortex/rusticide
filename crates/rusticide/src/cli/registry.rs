// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::Registry;

/// The one place model families are registered.
pub(crate) fn registry() -> Registry {
    Registry::new().with("llama", rusticide_llama::load)
}

#[cfg(test)]
mod tests {
    use super::registry;

    #[test]
    fn registers_the_llama_family() {
        assert_eq!(registry().architectures(), vec!["llama"]);
    }
}
