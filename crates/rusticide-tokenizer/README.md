<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-tokenizer

BPE and SentencePiece tokenizer for LLM text processing.

Builds a `Tokenizer` from the `tokenizer.ggml.*` keys of a GGUF
`rusticide_io::Metadata` store, encodes text into token IDs and decodes token
IDs back to text. Supports the `llama` (SentencePiece unigram, Viterbi) and
`gpt2` (byte-level BPE) models, compatible with llama.cpp.

## License

MIT OR Apache-2.0
