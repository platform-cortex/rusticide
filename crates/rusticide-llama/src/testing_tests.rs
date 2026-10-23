// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::GGUFValue;
use rusticide_model::ModelSource;
use rusticide_tokenizer::Tokenizer;

use super::{TINY_VOCAB, TinySource, tiny_source, tiny_tensor};

#[test]
fn tiny_source_feeds_the_tokenizer() {
    let source = tiny_source();
    let tokenizer = Tokenizer::from_gguf(source.metadata()).unwrap();
    assert_eq!(tokenizer.vocab_size(), TINY_VOCAB);
    assert_eq!((tokenizer.bos_id(), tokenizer.eos_id()), (1, 2));
    assert_eq!(tokenizer.encode("x").unwrap(), [1, 3, 4]);
    assert_eq!(tokenizer.encode("hi").unwrap(), [1, 3, 5, 6]);
}

#[test]
fn tiny_source_weights_are_small_and_non_zero() {
    let source = tiny_source();
    let values = source
        .tensor("blk.0.attn_q.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(values.len(), 16);
    assert!(values.iter().all(|v| (0.05..=0.25).contains(v)));
    let norm = source
        .tensor("output_norm.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(norm, [1.0; 4]);
}

#[test]
fn builder_overrides_and_drops_metadata_and_tensors() {
    let replacement = tiny_tensor("blk.0.attn_k.weight", &[1, 4], 0).unwrap();
    let source = TinySource::new()
        .with_metadata("llama.block_count", GGUFValue::Uint32(9))
        .with_metadata("extra.key", GGUFValue::Bool(true))
        .without_metadata("llama.context_length")
        .with_tensor("blk.0.attn_k.weight", replacement)
        .without_tensor("output.weight")
        .build();
    assert_eq!(source.metadata().block_count(), Some(9));
    assert!(source.metadata().get("extra.key").is_some());
    assert_eq!(source.metadata().context_length(), None);
    assert_eq!(source.tensor_shape("blk.0.attn_k.weight"), Some(vec![1, 4]));
    assert_eq!(source.tensor_shape("output.weight"), None);
    assert_eq!(source.tensor_shape("token_embd.weight"), Some(vec![7, 4]));
}

#[test]
fn a_later_override_wins_over_an_earlier_one() {
    let source = TinySource::new()
        .without_tensor("output.weight")
        .with_tensor(
            "output.weight",
            tiny_tensor("output.weight", &[7, 4], 0).unwrap(),
        )
        .build();
    assert_eq!(source.tensor_shape("output.weight"), Some(vec![7, 4]));
}
