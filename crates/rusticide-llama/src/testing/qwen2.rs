// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tiny qwen2 model: `qwen2.*` keys, attention biases, a GPT-2 tokenizer.

use rusticide_io::GGUFValue;
use rusticide_model::InMemorySource;

use super::{TinySource, text, tiny_tensor};

const BYTE_TOKENS: usize = 256;
const BOS: u32 = 256;
const EOS: u32 = 257;
const VOCAB: usize = BYTE_TOKENS + 2;

const HYPERPARAMETERS: [(&str, u32); 6] = [
    ("context_length", 16),
    ("embedding_length", 4),
    ("block_count", 1),
    ("feed_forward_length", 8),
    ("attention.head_count", 2),
    ("attention.head_count_kv", 1),
];

const BIASES: [(&str, usize); 3] = [
    ("blk.0.attn_q.bias", 4),
    ("blk.0.attn_k.bias", 2),
    ("blk.0.attn_v.bias", 2),
];

const DROPPED_TOKENIZER_KEYS: [&str; 2] = ["tokenizer.ggml.scores", "tokenizer.ggml.token_type"];

/// GPT-2's byte-to-unicode table: printable bytes map to themselves, the
/// rest to consecutive code points from U+0100.
fn byte_tokens() -> Vec<String> {
    let mut spare = 0x100_u32;
    (0..=u8::MAX)
        .map(|byte| {
            let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
            let code = if printable {
                u32::from(byte)
            } else {
                spare += 1;
                spare - 1
            };
            char::from_u32(code).map(String::from).unwrap_or_default()
        })
        .collect()
}

fn tokens() -> GGUFValue {
    let specials = ["<|endoftext|>", "<|im_end|>"].map(String::from);
    let pieces = byte_tokens().into_iter().chain(specials);
    GGUFValue::Array(pieces.map(|piece| text(&piece)).collect())
}

impl TinySource {
    /// The tiny model as qwen2: one block, `qwen2.*` hyperparameters, the three
    /// attention biases, and a GPT-2 tokenizer over the 256 byte tokens plus
    /// `<|endoftext|>` (BOS) and `<|im_end|>` (EOS), with no merges.
    #[must_use]
    pub fn qwen2() -> Self {
        let mut source = Self::new().with_metadata("general.architecture", text("qwen2"));
        for (suffix, value) in HYPERPARAMETERS {
            source = source
                .without_metadata(&format!("llama.{suffix}"))
                .with_metadata(&format!("qwen2.{suffix}"), GGUFValue::Uint32(value));
        }
        for key in DROPPED_TOKENIZER_KEYS {
            source = source.without_metadata(key);
        }
        source = source
            .with_metadata("tokenizer.ggml.model", text("gpt2"))
            .with_metadata("tokenizer.ggml.pre", text("qwen2"))
            .with_metadata("tokenizer.ggml.tokens", tokens())
            .with_metadata("tokenizer.ggml.bos_token_id", GGUFValue::Uint32(BOS))
            .with_metadata("tokenizer.ggml.eos_token_id", GGUFValue::Uint32(EOS))
            .with_metadata("tokenizer.ggml.add_bos_token", GGUFValue::Bool(false));
        for name in ["token_embd.weight", "output.weight"] {
            if let Some(tensor) = tiny_tensor(name, &[VOCAB, 4], 0) {
                source = source.with_tensor(name, tensor);
            }
        }
        for (salt, (name, len)) in BIASES.into_iter().enumerate() {
            if let Some(tensor) = tiny_tensor(name, &[len], salt) {
                source = source.with_tensor(name, tensor);
            }
        }
        source
    }
}

/// The stock valid 1-block qwen2: see [`TinySource::qwen2`].
#[must_use]
pub fn tiny_qwen2_source() -> InMemorySource {
    TinySource::qwen2().build()
}

#[cfg(test)]
#[path = "../testing_qwen2_tests.rs"]
mod tests;
