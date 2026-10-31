// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Token texts llama.cpp treats as end of generation
/// (`llama-vocab.cpp` at 8642541b2, lines 3257-3281).
const END_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eot|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "[e~[",
    "<|ifm|im_end|>",
];

/// Sorted, de-duplicated ids of the tokens that end generation.
pub(crate) fn end_of_generation_ids(
    vocab: &[Vec<u8>],
    eos_id: u32,
    end_of_turn_id: Option<u32>,
) -> Vec<u32> {
    let mut ids: Vec<u32> = vocab
        .iter()
        .enumerate()
        .filter(|(_, token)| {
            END_TEXTS
                .iter()
                .any(|text| text.as_bytes() == token.as_slice())
        })
        .filter_map(|(index, _)| u32::try_from(index).ok())
        .collect();
    ids.push(eos_id);
    ids.extend(end_of_turn_id);
    ids.sort_unstable();
    ids.dedup();
    ids
}
