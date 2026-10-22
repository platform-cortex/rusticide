// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the GGUF reader against arbitrary bytes.
//!
//! A malformed model file is untrusted input, so `GgufFile::from_bytes` must
//! return a `GgufError` for anything it cannot parse -- never panic, and never
//! read past the end of the buffer. When a parse unexpectedly succeeds, this
//! walks every metadata accessor and materializes every tensor, where an
//! out-of-range offset or length in the header would actually bite.

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

    let metadata = file.metadata();
    let _ = metadata.architecture();
    let _ = metadata.context_length();
    let _ = metadata.embedding_length();
    let _ = metadata.block_count();
    let _ = metadata.head_count();
    let _ = metadata.head_count_kv();
    let _ = metadata.feed_forward_length();
    let _ = metadata.rope_freq_base();
    let _ = metadata.rms_norm_eps();

    for name in file.tensor_names() {
        let _ = file.tensor_info(name);
        let _ = file.tensor(name);
    }
    for index in 0..file.tensor_count() {
        let _ = file.tensor_by_index(index);
    }
});
