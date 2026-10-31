// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::Registry;

/// The one place model families are registered.
pub(crate) fn registry() -> Registry {
    Registry::new()
        .with("llama", rusticide_llama::load)
        .with("qwen2", rusticide_llama::load_qwen2)
}

#[cfg(test)]
mod tests {
    use super::registry;

    #[test]
    fn registers_the_llama_and_qwen2_families() {
        assert_eq!(registry().architectures(), vec!["llama", "qwen2"]);
    }
}
