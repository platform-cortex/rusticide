// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A tiny valid llama model for tests, with tokenizer metadata.
//!
//! [`tiny_source`] is the stock model; [`TinySource`] is a fluent builder that
//! overrides or drops individual metadata keys and tensors before building.
//!
//! Weights are deterministic: a matrix entry is `0.05 * (1 + (i + salt) % 5)`
//! where `i` is the flat index and `salt` the tensor's position in the layout;
//! norm weights are `1.0`.

use rusticide_core::Tensor;
use rusticide_io::{GGUFValue, Metadata};
use rusticide_model::InMemorySource;

mod qwen2;

pub use qwen2::tiny_qwen2_source;

/// Number of tokens in the tiny vocabulary.
pub const TINY_VOCAB: usize = 7;

const EMBEDDING: usize = 4;
const FEED_FORWARD: usize = 8;
const KV_DIM: usize = 2;

const PIECES: [&str; TINY_VOCAB] = ["<unk>", "<s>", "</s>", "\u{2581}", "x", "h", "i"];
const TOKEN_TYPES: [i32; TINY_VOCAB] = [2, 3, 3, 1, 1, 1, 1];

fn text(value: &str) -> GGUFValue {
    GGUFValue::String(value.to_owned())
}

fn metadata_pairs() -> Vec<(String, GGUFValue)> {
    let count = |n: u32| GGUFValue::Uint32(n);
    let pairs = [
        ("general.architecture", text("llama")),
        ("llama.context_length", count(16)),
        ("llama.embedding_length", count(4)),
        ("llama.block_count", count(1)),
        ("llama.feed_forward_length", count(8)),
        ("llama.attention.head_count", count(2)),
        ("llama.attention.head_count_kv", count(1)),
        ("tokenizer.ggml.model", text("llama")),
        ("tokenizer.ggml.bos_token_id", count(1)),
        ("tokenizer.ggml.eos_token_id", count(2)),
        (
            "tokenizer.ggml.tokens",
            GGUFValue::Array(PIECES.iter().map(|piece| text(piece)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            GGUFValue::Array(
                [0.0_f32, 0.0, 0.0, -1.0, -2.0, -2.0, -2.0]
                    .into_iter()
                    .map(GGUFValue::Float32)
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            GGUFValue::Array(TOKEN_TYPES.into_iter().map(GGUFValue::Int32).collect()),
        ),
    ];
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

fn layout() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("token_embd.weight", vec![TINY_VOCAB, EMBEDDING]),
        ("blk.0.attn_norm.weight", vec![EMBEDDING]),
        ("blk.0.attn_q.weight", vec![EMBEDDING, EMBEDDING]),
        ("blk.0.attn_k.weight", vec![KV_DIM, EMBEDDING]),
        ("blk.0.attn_v.weight", vec![KV_DIM, EMBEDDING]),
        ("blk.0.attn_output.weight", vec![EMBEDDING, EMBEDDING]),
        ("blk.0.ffn_norm.weight", vec![EMBEDDING]),
        ("blk.0.ffn_gate.weight", vec![FEED_FORWARD, EMBEDDING]),
        ("blk.0.ffn_up.weight", vec![FEED_FORWARD, EMBEDDING]),
        ("blk.0.ffn_down.weight", vec![EMBEDDING, FEED_FORWARD]),
        ("output_norm.weight", vec![EMBEDDING]),
        ("output.weight", vec![TINY_VOCAB, EMBEDDING]),
    ]
}

/// The stock tensor named `name` with the given shape; `salt` varies the data.
#[must_use]
pub fn tiny_tensor(name: &str, shape: &[usize], salt: usize) -> Option<Tensor> {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = if name.ends_with("_norm.weight") {
        vec![1.0; len]
    } else {
        (0..len)
            .map(|index| 0.05 * f32::from(1 + u8::try_from((index + salt) % 5).unwrap_or(0)))
            .collect()
    };
    Tensor::from_slice(&data, shape).ok()
}

/// A fluent builder over the stock tiny model.
#[derive(Debug, Default)]
pub struct TinySource {
    metadata: Vec<(String, Option<GGUFValue>)>,
    tensors: Vec<(String, Option<Tensor>)>,
}

impl TinySource {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets (or adds) a metadata key.
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: GGUFValue) -> Self {
        self.metadata.push((key.to_owned(), Some(value)));
        self
    }

    /// Removes a metadata key.
    #[must_use]
    pub fn without_metadata(mut self, key: &str) -> Self {
        self.metadata.push((key.to_owned(), None));
        self
    }

    /// Replaces (or adds) a tensor.
    #[must_use]
    pub fn with_tensor(mut self, name: &str, tensor: Tensor) -> Self {
        self.tensors.push((name.to_owned(), Some(tensor)));
        self
    }

    /// Removes a tensor.
    #[must_use]
    pub fn without_tensor(mut self, name: &str) -> Self {
        self.tensors.push((name.to_owned(), None));
        self
    }

    /// Applies the overrides in the order they were given.
    #[must_use]
    pub fn build(self) -> InMemorySource {
        let mut metadata = metadata_pairs();
        for (key, value) in self.metadata {
            metadata.retain(|(existing, _)| *existing != key);
            if let Some(value) = value {
                metadata.push((key, value));
            }
        }
        let mut source = InMemorySource::new(metadata.into_iter().collect::<Metadata>());
        let mut tensors: Vec<(String, Option<Tensor>)> = layout()
            .into_iter()
            .enumerate()
            .map(|(salt, (name, shape))| (name.to_owned(), tiny_tensor(name, &shape, salt)))
            .collect();
        tensors.extend(self.tensors);
        let mut finished: Vec<(String, Option<Tensor>)> = Vec::new();
        for (name, tensor) in tensors {
            finished.retain(|(existing, _)| *existing != name);
            finished.push((name, tensor));
        }
        for (name, tensor) in finished {
            if let Some(tensor) = tensor {
                source = source.with_tensor(&name, tensor);
            }
        }
        source
    }
}

/// The stock valid 1-block llama: embedding 4, 2 heads, 1 kv head,
/// feed-forward 8, context 16, [`TINY_VOCAB`] tokens, untied output weight.
#[must_use]
pub fn tiny_source() -> InMemorySource {
    TinySource::new().build()
}

#[cfg(test)]
#[path = "testing_tests.rs"]
mod tests;
