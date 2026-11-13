// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::registry;

#[test]
fn registry_lists_llama_qwen2_and_bitnet() {
    assert_eq!(
        registry().architectures(),
        vec!["llama", "qwen2", "bitnet-b1.58"]
    );
}
