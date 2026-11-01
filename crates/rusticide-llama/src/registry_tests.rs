// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::registry;

#[test]
fn registry_lists_llama_and_qwen2() {
    assert_eq!(registry().architectures(), vec!["llama", "qwen2"]);
}
