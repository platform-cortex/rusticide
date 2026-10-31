// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the llama and qwen2 loaders against arbitrary GGUF bytes.
//!
//! `rusticide_llama::load` and `load_qwen2` trust hyperparameters and tensor
//! shapes from an untrusted file: dimensions may be zero, overflow, or
//! disagree with the tensors. Loading must return a `ModelError` rather than
//! panic, and a model that does load must survive one forward pass.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rusticide_io::GgufFile;

fuzz_target!(|data: &[u8]| {
    // Below the 24-byte header there is nothing to parse.
    if data.len() < 24 {
        return;
    }
    let Ok(file) = GgufFile::from_bytes(data.to_vec()) else {
        return;
    };
    for load in [rusticide_llama::load, rusticide_llama::load_qwen2] {
        if let Ok(mut model) = load(&file) {
            let _ = model.forward(0, 0);
        }
    }
});
