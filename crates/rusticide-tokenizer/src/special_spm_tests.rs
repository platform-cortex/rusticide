// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;

use crate::test_support::{build, float_array, tokenizer_meta, with};
use crate::tokenizer::Tokenizer;

const HI: u32 = 7;

/// `<unk>`, `<s>`, `</s>` typed as llama.cpp's SPM files type them, then the
/// pieces that merge "\u{2581}hi" (id 7).
fn spm(add_space_prefix: bool) -> Tokenizer {
    let tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "\u{2581}",
        "h",
        "i",
        "\u{2581}h",
        "\u{2581}hi",
    ];
    let types = [2, 3, 3, 1, 1, 1, 1, 1].map(GGUFValue::Int32).to_vec();
    let entries = with(
        tokenizer_meta("llama", &tokens),
        "tokenizer.ggml.scores",
        float_array(&[0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -0.8, -0.5]),
    );
    let entries = with(
        entries,
        "tokenizer.ggml.token_type",
        GGUFValue::Array(types),
    );
    let entries = with(
        entries,
        "tokenizer.ggml.add_space_prefix",
        GGUFValue::Bool(add_space_prefix),
    );
    build(&entries)
}

#[test]
fn control_tokens_in_an_spm_prompt_follow_the_bos() {
    assert_eq!(spm(true).encode("<s>hi</s>").unwrap(), [1, 1, HI, 2]);
}

#[test]
fn every_fragment_after_a_special_token_gets_the_space_prefix() {
    assert_eq!(spm(true).encode("hi</s>hi").unwrap(), [1, HI, 2, HI]);
}

#[test]
fn without_a_space_prefix_fragments_are_encoded_bare() {
    assert_eq!(spm(false).encode("hi</s>hi").unwrap(), [1, 4, 5, 2, 4, 5]);
}

#[test]
fn a_prompt_of_only_special_tokens_has_no_text_pieces() {
    assert_eq!(spm(true).encode("</s><unk>").unwrap(), [1, 2, 0]);
}
