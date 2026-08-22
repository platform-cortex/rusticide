// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the GGUF reader against arbitrary bytes.
//!
//! A malformed model file is untrusted input, so `Model::load` must return a
//! `ModelError` for anything it cannot parse -- never panic, and never read
//! past the end of the buffer. This target asserts that by feeding it random
//! bytes and, when a load unexpectedly succeeds, walking every accessor and
//! materializing every tensor (where an out-of-range offset or length in the
//! header would actually bite).

#![no_main]

mod common;

use common::ScratchFile;
use libfuzzer_sys::fuzz_target;
use xllm_model::Model;

fuzz_target!(|data: &[u8]| {
    // Below the 24-byte header there is nothing to parse; skip the file I/O.
    if data.len() < 24 {
        return;
    }

    let Ok(scratch) = ScratchFile::new("load", data) else {
        return;
    };

    // Only a clean parse is interesting past this point; errors are the
    // expected outcome for almost every input.
    let Ok(model) = Model::load(scratch.path()) else {
        return;
    };

    // Metadata accessors must tolerate whatever the header claimed.
    let _ = model.architecture();
    let _ = model.context_length();
    let _ = model.embedding_length();
    let _ = model.block_count();
    let _ = model.head_count();
    let _ = model.head_count_kv();
    let _ = model.feed_forward_length();
    let _ = model.rope_freq_base();
    let _ = model.rms_norm_eps();

    // Tensor materialization is where a bogus offset/length pair would cause
    // an out-of-bounds read, so exercise every one the header advertises.
    for name in model.tensor_names() {
        let _ = model.tensor_info(&name);
        let _ = model.tensor(&name);
    }
    for index in 0..model.tensor_count() {
        let _ = model.tensor_by_index(index);
    }
});
