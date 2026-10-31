// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::ModelSource;
use rusticide_tokenizer::Tokenizer;

use crate::testing::{TinySource, tiny_qwen2_source};

#[test]
fn qwen2_source_feeds_the_gpt2_tokenizer() {
    let source = tiny_qwen2_source();
    let tokenizer = Tokenizer::from_gguf(source.metadata()).unwrap();
    assert_eq!(tokenizer.vocab_size(), 258);
    assert_eq!((tokenizer.bos_id(), tokenizer.eos_id()), (256, 257));
    assert_eq!(tokenizer.encode("hi").unwrap(), [104, 105]);
    assert_eq!(tokenizer.decode(&[104, 105]).unwrap(), "hi");
}

#[test]
fn qwen2_source_declares_the_architecture_and_biases() {
    let source = TinySource::qwen2().build();
    assert_eq!(source.metadata().architecture(), Some("qwen2"));
    assert_eq!(source.tensor_shape("blk.0.attn_q.bias"), Some(vec![4]));
    assert_eq!(source.tensor_shape("blk.0.attn_k.bias"), Some(vec![2]));
    assert_eq!(source.tensor_shape("token_embd.weight"), Some(vec![258, 4]));
}
