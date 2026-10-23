// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};

use super::Registry;
use crate::error::{ModelError, Result};
use crate::model::Model;
use crate::source::{InMemorySource, ModelSource};

struct Echo {
    vocab: usize,
}

impl Model for Echo {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn context_length(&self) -> usize {
        8
    }

    fn forward(&mut self, token: u32, _position: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0.0; self.vocab];
        logits[token as usize] = 1.0;
        Ok(logits)
    }

    fn reset(&mut self) {}
}

#[allow(clippy::unnecessary_wraps)]
fn load_echo(_source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    Ok(Box::new(Echo { vocab: 4 }))
}

fn source(architecture: Option<&str>) -> InMemorySource {
    let metadata: Metadata = architecture
        .map(|name| {
            (
                "general.architecture".to_owned(),
                GGUFValue::String(name.into()),
            )
        })
        .into_iter()
        .collect();
    InMemorySource::new(metadata)
}

fn two_entries() -> Registry {
    Registry::new()
        .with("echo", load_echo)
        .with("llama2", load_echo)
}

#[test]
fn registry_loads_by_architecture() {
    let mut model = two_entries().load(&source(Some("echo"))).unwrap();
    assert_eq!(model.vocab_size(), 4);
    assert_eq!(model.context_length(), 8);
    assert_eq!(model.forward(2, 0).unwrap(), [0.0, 0.0, 1.0, 0.0]);
    model.reset();
}

#[test]
fn registry_without_architecture_key_is_missing_metadata() {
    let error = two_entries().load(&source(None)).err().unwrap();
    assert!(matches!(
        error,
        ModelError::MissingMetadata { ref key } if key == "general.architecture"
    ));
}

#[test]
fn registry_rejects_an_unknown_architecture() {
    let error = two_entries().load(&source(Some("gptj"))).err().unwrap();
    assert!(matches!(
        error,
        ModelError::UnknownArchitecture { ref name, ref supported }
            if name == "gptj" && supported == "echo, llama2"
    ));
}

#[test]
fn registry_lists_architectures() {
    assert_eq!(two_entries().architectures(), ["echo", "llama2"]);
    assert_eq!(Registry::new().architectures(), [] as [&str; 0]);
}
