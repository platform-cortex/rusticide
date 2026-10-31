// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the tokenizer built from arbitrary GGUF metadata.
//!
//! `Tokenizer::from_gguf` trusts vocabulary, scores and special-token ids that
//! came out of an untrusted file: the vocab may be empty, the score array may
//! disagree with the token count, and bos/eos ids may point outside the vocab.
//! Encoding and decoding under those conditions must still return a `Result`
//! rather than panicking or indexing out of bounds.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rusticide_io::GgufFile;
use rusticide_tokenizer::Tokenizer;

/// Exercises the byte-fallback path, multi-byte UTF-8, and leading whitespace.
const PROBES: [&str; 5] = [
    "",
    " ",
    "Hello, world",
    "  leading space",
    "\u{4f60}\u{597d}",
];

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    let Ok(file) = GgufFile::from_bytes(data.to_vec()) else {
        return;
    };
    let Ok(tokenizer) = Tokenizer::from_gguf(file.metadata()) else {
        return;
    };

    // The input itself is also text: GPT-2 pre-tokenization runs regexes on it.
    let fuzzed = String::from_utf8_lossy(data);
    let probes = PROBES.into_iter().chain([fuzzed.as_ref()]);
    for probe in probes {
        let Ok(tokens) = tokenizer.encode(probe) else {
            continue;
        };
        // Round-trip: ids produced by encode must always be decodable.
        let _ = tokenizer.decode(&tokens);
        let _ = tokenizer.decode_detokenized(&tokens);
    }

    // Ids that no encode call would produce must be rejected, not panic.
    let _ = tokenizer.decode(&[0, 1, 2, u32::MAX]);

    // Ids derived from the input hit arbitrary, mostly out-of-range, pieces.
    let derived: Vec<u32> = data
        .iter()
        .take(16)
        .map(|&byte| u32::from(byte) * 257)
        .collect();
    let _ = tokenizer.decode(&derived);
    let _ = tokenizer.decode_detokenized(&derived);
});
