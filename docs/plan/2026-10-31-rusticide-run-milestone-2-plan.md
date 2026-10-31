<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide run, Milestone 2 (Qwen2.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rusticide run qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "..."`
streams text whose first 32 greedy tokens match llama.cpp, released as 0.4.0.

**Architecture:** No new crates. `rusticide-quantize` gains a table of block
formats (the format contract) plus Q8_0 and Q5_0; `rusticide-io` stops
rejecting a file over one unknown tensor type and looks formats up in that
table. `rusticide-tokenizer` gets a working GPT-2 byte-level BPE path: the
byte-to-unicode table, a pre-tokenizer chosen by `tokenizer.ggml.pre`, and an
end-of-generation set. `rusticide-llama` covers `qwen2` as a variant of the
same family (half-split RoPE, Q/K/V biases), registered in the CLI's one
registry line.

**Tech Stack:** Rust 2024 (MSRV 1.96), rayon, fancy-regex (new, pure Rust),
clap, thiserror/anyhow, serde_json (dev only), llama.cpp `llama-server`
0.8.0 as the reference (outside the repo).

**Spec:** `docs/plan/2026-10-22-rusticide-run-design.md`. Read it first,
including "Facts the builder needs" (the Qwen2.5 file's metadata and tensor
types). This plan implements its milestone 2 and nothing from milestone 3.
Milestone 1's plan (`docs/plan/2026-10-22-rusticide-run-milestone-1-plan.md`)
shows the shape of the code this one extends.

## Global Constraints

- AGENTS.md applies in full: strict TDD (watch each test fail first), 100%
  tarpaulin coverage, no `unwrap`/`expect` outside `#[cfg(test)]`, no
  `unsafe`, files under 300 lines tests included, ASCII only, the three-line
  SPDX header on every new `.rs` file, thiserror in libraries and anyhow in
  the binary, GPG-signed conventional commits with no trailers.
- MSRV 1.96: no std APIs newer than 1.96. `fancy-regex` must be a release
  whose own `rust-version` is at most 1.96.
- ASCII only applies to source: the GPT-2 byte table and every non-ASCII
  test string are written with `\u{...}` escapes, never literal characters.
- The only new dependency is `fancy-regex` (`[workspace.dependencies]`,
  then `fancy-regex.workspace = true` in `rusticide-tokenizer`).
  `mise run security` (cargo audit + cargo deny) must pass with it.
- No panics on hostile input: file-controlled sizes use checked arithmetic;
  an unknown tensor type is an error only when that tensor is materialized.
- Weights are dequantized to f32 once at load; the file is read whole.
- One implementation per kernel (spec anti-goal): the new RoPE variant is a
  kernel in `rusticide-kernels`, not a copy inside `rusticide-llama`.
- No `if arch == "qwen2"` outside `rusticide-llama`'s variant type and the
  CLI's registry line (spec anti-goal).
- Success criterion (spec): prompt tokens equal the reference's; the first
  **32** generated greedy tokens equal the reference's, except that a
  divergence passes when rusticide's own top-1 and top-2 logits at that
  position differ by less than **0.1** (comparison stops there).
- Canonical model: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`,
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`, SHA-256
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
- Reference tool: llama.cpp 0.8.0 (build 11960, commit `8642541b2`), from
  Homebrew on the M1 Pro. Pre-tokenizer patterns and the end-of-generation
  token texts are copied verbatim from `src/llama-vocab.cpp` at that commit.
- GGML type ids: F32 0, F16 1, Q4_0 2, Q5_0 6, Q8_0 8, Q4_K 12, Q6_K 14.

## Review Focus

1. **A GPT-2 file without `tokenizer.ggml.pre`** -> llama.cpp's default
   pre-tokenizer, as the spec requires; an unrecognized name ->
   `TokenizerError::UnsupportedPreTokenizer(name)` at load, never a silent
   fallback. Tests: Task 5 `absent_pre_uses_the_default_patterns`,
   `unknown_pre_is_unsupported`.
2. **Whitespace runs, newlines and digits in a prompt** -> pieces exactly as
   llama.cpp splits them (`\s+(?!\S)` keeps the last space for the next
   word; qwen2 splits digits one by one). Tests: Task 5
   `qwen2_splits_*`.
3. **The model emits `<|endoftext|>` rather than the declared EOS** ->
   generation stops without printing it. Tests: Task 6
   `endoftext_ends_generation`, `generation_stops_at_any_end_of_generation_token`.
4. **A GPT-2 vocabulary missing one of the 256 byte tokens** ->
   `TokenizerError::MissingByteToken { byte }` at load, never a panic or a
   dropped byte at encode. Test: Task 4
   `from_gguf_rejects_a_vocab_missing_a_byte_token`.
5. **A truncated or misaligned Q5_0/Q8_0 tensor, or an unknown type on an
   unused tensor** -> `DataTooShort`/`UnalignedShape` naming the format, or
   `UnsupportedTensorType` naming the tensor only when it is loaded; the
   rest of the file stays usable. Tests: Task 2 `q5_0_rejects_*`,
   `q8_0_rejects_*`; Task 3 `unknown_type_fails_only_when_materialized`.

---

### Task 1: rusticide-quantize, the block-format table

**Files:**

- Create: `crates/rusticide-quantize/src/format.rs` (+ `format_tests.rs`)
- Modify: `crates/rusticide-quantize/src/{lib,error,q4_0,q4_k,q6_k}.rs`,
  `crates/rusticide-quantize/Cargo.toml` (description and keywords)

**Interfaces:**

- Produces:
  - `pub struct BlockFormat { pub name: &'static str, pub type_id: u32,
    pub block_size: usize, pub block_bytes: usize, pub dequantize_block:
    fn(block: &[u8], out: &mut [f32]) }` -- `dequantize_block` receives
    exactly `block_bytes` bytes and `block_size` outputs.
  - `impl BlockFormat { pub const fn byte_len(&self, numel: usize) ->
    Option<usize>; pub fn dequantize(&self, data: &[u8], shape: &[usize]) ->
    Result<Tensor, QuantizeError> }` -- `byte_len` is `None` on overflow;
    `dequantize` checks alignment, then length, then fills a fallibly
    reserved buffer block by block (the logic `dequantize_q4_k` has today,
    written once).
  - `pub fn block_format(type_id: u32) -> Option<&'static BlockFormat>`
    over one `static FORMATS: [BlockFormat; N]`.
  - `QuantizeError::{DataTooShort, UnalignedShape}` take
    `format: &'static str` (the `name`); `QuantizationType` is deleted.
    Display stays `"{format} data too short: {got} bytes, need {needed}"`.
- Removed: `dequantize_q4_k`, `dequantize_q6_k`,
  `Quantizer::dequantize_q4_0` (their bodies become each module's
  `dequantize_block`). `Quantizer::quantize_q4_0` and the
  `*_BLOCK_SIZE`/`*_BLOCK_BYTES` constants stay.
- [ ] **Step 1: Write the failing tests** in `format_tests.rs`:

```rust
#[test] fn table_knows_the_existing_formats() {
    assert_eq!(block_format(2).map(|f| f.name), Some("Q4_0"));
    assert_eq!(block_format(12).map(|f| (f.block_size, f.block_bytes)), Some((256, 144)));
    assert_eq!(block_format(14).map(|f| f.name), Some("Q6_K"));
}
#[test] fn unknown_type_id_has_no_format() { assert!(block_format(99).is_none()); }
#[test] fn byte_len_counts_whole_blocks() { assert_eq!(block_format(2).unwrap().byte_len(64), Some(36)); }
#[test] fn byte_len_overflow_is_none() { assert_eq!(block_format(12).unwrap().byte_len(usize::MAX), None); }
#[test] fn dequantize_rejects_a_partial_block()
    // Q4_K, shape [255] -> UnalignedShape { format: "Q4_K", numel: 255, block: 256 }
#[test] fn dequantize_rejects_short_data()
    // Q6_K, 10 bytes for [256] -> DataTooShort { format: "Q6_K", got: 10, needed: 210 }
```

  The existing per-format tests move to calling
  `block_format(id).unwrap().dequantize(..)` with unchanged expected values.

- [ ] **Step 2: Run `cargo test -p rusticide-quantize`; expect FAIL.**
- [ ] **Step 3: Implement** `format.rs`; make `q4_0`, `q4_k`, `q6_k` expose
  `pub(crate) fn dequantize_block`; update `rusticide-io/src/gguf/load.rs`
  to call `block_format(raw).dequantize` for the three ids (Task 3 finishes
  the io side; keep its tests green here).
- [ ] **Step 4: Run `cargo test -p rusticide-quantize -p rusticide-io`;
  expect PASS. `mise run lint`, `mise run test` (100%).**
- [ ] **Step 5: Commit**
  `refactor(quantize): look block formats up in one table`.

---

### Task 2: rusticide-quantize, Q8_0 and Q5_0

**Files:**

- Create: `crates/rusticide-quantize/src/q8_0.rs`,
  `crates/rusticide-quantize/src/q5_0.rs` (tests in each)
- Create: `crates/rusticide-fuzz/fuzz_targets/dequantize.rs`; modify
  `crates/rusticide-fuzz/Cargo.toml` (new `[[bin]]`, path dependency on
  `rusticide-quantize`)
- Modify: `crates/rusticide-quantize/src/{lib,format}.rs` (two table rows)

**Interfaces:**

- Produces: table rows `Q8_0` (type id 8, block 32, 34 bytes) and `Q5_0`
  (type id 6, block 32, 22 bytes); `pub const Q8_0_BLOCK_BYTES`,
  `Q5_0_BLOCK_BYTES`.
- Block layouts (ggml `dequantize_row_q8_0` / `dequantize_row_q5_0`):

```text
Q8_0: d = f16 le [0..2]; qs = 32 x i8 [2..34];  y[j] = d * qs[j]
Q5_0: d = f16 le [0..2]; qh = u32 le [2..6]; qs = 16 bytes [6..22]
      for j in 0..16:
        hi0 = ((qh >> j) << 4) & 0x10;  hi1 = (qh >> (j + 12)) & 0x10
        y[j]      = d * (((qs[j] & 0x0F) | hi0) as i32 - 16)
        y[j + 16] = d * (((qs[j] >> 4)   | hi1) as i32 - 16)
```

- [ ] **Step 1: Write the failing tests:**

```rust
#[test] fn q8_0_scales_signed_bytes()      // d = 0.5, qs = [-128, -1, 0, 1, 127, 0...] -> [-64, -0.5, 0, 0.5, 63.5, 0...]
#[test] fn q5_0_without_high_bits_is_nibble_minus_16()
    // d = 1, qh = 0, qs[0] = 0x21 -> y[0] = 1 - 16 = -15, y[16] = 2 - 16 = -14
#[test] fn q5_0_high_bits_add_sixteen()
    // d = 1, qh = 1 | (1 << 16), qs[0] = 0x00 -> y[0] = 0, y[16] = 0 (16 - 16); y[1] = -16
#[test] fn q5_0_uses_every_high_bit()
    // d = 1, qh = 0xFFFF_FFFF, qs all 0xFF -> every y == 15
#[test] fn q8_0_rejects_short_data()      // DataTooShort { format: "Q8_0", got: 33, needed: 34 }
#[test] fn q5_0_rejects_a_partial_block() // UnalignedShape { format: "Q5_0", numel: 31, block: 32 }
```

- [ ] **Step 2: Run `cargo test -p rusticide-quantize`; expect FAIL.**
- [ ] **Step 3: Implement both `dequantize_block`s and the two rows.**
- [ ] **Step 4: Run; PASS. `mise run lint`, `mise run test`.**
- [ ] **Step 5: Fuzz target `dequantize`:** the first byte picks a format
  (`FORMATS` index modulo its length, through a `pub fn formats() ->
  &'static [BlockFormat]` accessor added here), the rest is data, shape
  `[rest.len() / block_bytes * block_size]`; result ignored. Run it for
  60 s per `crates/rusticide-fuzz` conventions; no crash. Commit the target.
- [ ] **Step 6: Commit** `feat(quantize): add Q8_0 and Q5_0`.

---

### Task 3: rusticide-io, lazy rejection of tensor types

**Files:**

- Modify: `crates/rusticide-io/src/gguf/{tensor_info,load,mod}.rs`,
  `crates/rusticide-io/src/{error,lib}.rs` and their `*_tests.rs` /
  `test_support.rs`

**Interfaces:**

- Produces:
  - `pub struct TensorInfo { pub name: String, pub shape: Vec<usize>,
    pub offset: u64, pub type_id: u32 }` -- the raw GGML type id, recorded
    whatever it is. `GgmlType` and the `dtype` field are deleted.
  - `GgufError::UnsupportedTensorType { name: String, type_id: u32 }`,
    message `tensor {name} has unsupported type {type_id}` (replaces the
    tuple variant).
- Behaviour: parsing never looks at the type id. `tensor(name)` decodes
  type 0 as F32 and 1 as F16 (as today), any other id through
  `block_format(type_id)`, and `None` there is `UnsupportedTensorType`.
  Byte length: `numel * 4`, `numel * 2`, or `BlockFormat::byte_len`, all
  checked.
- [ ] **Step 1: Write the failing tests:**

```rust
#[test] fn parse_records_an_unknown_type_id()        // tensor info type_id == 99
#[test] fn unknown_type_fails_only_when_materialized()
    // file with "good" (F32) and "odd" (type 99): metadata() readable, tensor("good") Ok,
    // tensor("odd") -> UnsupportedTensorType { name: "odd", type_id: 99 }
#[test] fn q8_0_tensor_loads_through_the_table()      // one Q8_0 block, values as Task 2
#[test] fn unsupported_type_message_names_the_tensor()
    // "tensor odd has unsupported type 99"
```

- [ ] **Step 2: Run `cargo test -p rusticide-io`; expect FAIL.**
- [ ] **Step 3: Implement.** The `gguf_load` and `gguf_tokenizer` fuzz
  targets must still compile; adjust them only if they named `GgmlType`.
- [ ] **Step 4: Run; PASS. `mise run lint`, `mise run test`.
  `target/release/rusticide run <qwen file> --prompt x` now fails with
  `unknown architecture qwen2; supported: llama` (types no longer stop it).**
- [ ] **Step 5: Commit**
  `feat(io): reject unsupported tensor types only when loaded`.

---

### Task 4: rusticide-tokenizer, GPT-2 byte-level encoding and decoding

**Files:**

- Create: `crates/rusticide-tokenizer/src/byte_unicode.rs` (+ tests)
- Modify: `crates/rusticide-tokenizer/src/{bpe,decode,from_gguf,tokenizer,
  error,spm,lib}.rs` and `bpe_tests.rs`, `decode_tests.rs`,
  `from_gguf_tests.rs`, `test_support.rs`

**Interfaces:**

- Produces:
  - `pub(crate) const fn byte_to_char(byte: u8) -> char` and
    `pub(crate) fn char_to_byte(c: char) -> Option<u8>`: GPT-2's
    `bytes_to_unicode`. Bytes `0x21..=0x7E`, `0xA1..=0xAC` and
    `0xAE..=0xFF` map to the same code point; the other 68 bytes, in
    ascending order, map to `U+0100`, `U+0101`, ...
  - `Tokenizer` field `byte_tokens: [Option<u32>; 256]`: for GPT-2, the id
    of the one-character token `byte_to_char(b)` (built in both `new` and
    `from_gguf`).
  - `TokenizerError::MissingByteToken { byte: u8 }`, message
    `gpt2 vocabulary has no token for byte 0x{byte:02X}`, returned by
    `from_gguf` for a GPT-2 vocabulary lacking any of the 256.
- Behaviour: `encode_gpt2` becomes fallible. Each pre-tokenized piece (still
  `split_whitespace` until Task 5 replaces it) starts as
  `byte_tokens[b]` per byte (`None` -> `EncodeError`), then merges by rank
  as `best_merge` does today. GPT-2 decode maps each character of a piece
  through `char_to_byte`; a character outside the table (special tokens)
  passes through as its UTF-8 bytes.
- [ ] **Step 1: Write the failing tests:**

```rust
#[test] fn printable_bytes_map_to_themselves()   // byte_to_char(b'A') == 'A', 0xE9 -> '\u{e9}'
#[test] fn space_and_newline_are_shifted()       // b' ' -> '\u{120}', b'\n' -> '\u{10a}', 0x00 -> '\u{100}', 0xAD -> '\u{143}'
#[test] fn every_byte_round_trips()              // char_to_byte(byte_to_char(b)) == Some(b) for all 256
#[test] fn unmapped_char_has_no_byte()           // char_to_byte('\u{2581}') == None
#[test] fn encode_maps_bytes_through_the_table() // vocab = 256 byte chars, "hi" -> [id('h'), id('i')]
#[test] fn encode_merges_by_rank()               // merges ["h i"] with "hi" in vocab -> [id("hi")]
#[test] fn decode_reverses_the_byte_mapping()    // tokens of "\u{120}ol\u{c3}\u{a1}" -> " ol\u{e1}"
#[test] fn decode_passes_special_tokens_through() // "<|im_end|>" decodes to itself
#[test] fn from_gguf_rejects_a_vocab_missing_a_byte_token() // MissingByteToken { byte: 0x00 }
```

  A `test_support` helper builds the 256-token GPT-2 vocabulary (plus
  extra tokens and merges) so later tasks reuse it.

- [ ] **Step 2: Run `cargo test -p rusticide-tokenizer`; expect FAIL.**
- [ ] **Step 3: Implement.** Rewrite the existing `bpe_tests.rs` cases that
  assumed "byte `b` is token `b`" against the helper vocabulary; delete
  any that only pinned that bug.
- [ ] **Step 4: Run; PASS. `mise run lint`, `mise run test`.**
- [ ] **Step 5: Commit**
  `fix(tokenizer): map gpt2 bytes through the byte-to-unicode table`.

---

### Task 5: rusticide-tokenizer, the pre-tokenizer

**Files:**

- Create: `crates/rusticide-tokenizer/src/pre_tokenizer.rs` (+ tests)
- Modify: `Cargo.toml` (workspace dependency `fancy-regex`),
  `crates/rusticide-tokenizer/{Cargo.toml,src/bpe.rs,src/from_gguf.rs,
  src/tokenizer.rs,src/error.rs}`

**Interfaces:**

- Produces:
  - `#[derive(Clone, Copy)] pub(crate) struct PreTokenizer { patterns:
    &'static [&'static str] }` with
    `pub(crate) fn from_name(name: Option<&str>) -> Result<Self>` and
    `pub(crate) fn split<'t>(self, text: &'t str) -> Result<Vec<&'t str>>`.
  - Names: `"gpt2"`, `"qwen2"`, and `None` (key absent) for llama.cpp's
    `LLAMA_VOCAB_PRE_TYPE_DEFAULT` pattern list. Any other name is
    `TokenizerError::UnsupportedPreTokenizer(String)`, message
    `unsupported pre-tokenizer: {0}`. The qwen2 pattern, for orientation
    (copy the exact string from the reference commit):

```text
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

- `split` applies the patterns in order, as llama.cpp's
  `unicode_regex_split` does: each pattern splits every current piece
  into its matches and the gaps between them, and gaps are kept as
  pieces. Patterns are compiled per call (prompts are encoded once per
  run), which keeps construction infallible and needs no static; a
  compile or match error maps to `EncodeError`.
- `Tokenizer` field `pre_tokenizer: Option<PreTokenizer>`: read from
  `tokenizer.ggml.pre` for GPT-2, `None` for SentencePiece;
  `Tokenizer::new` with GPT-2 uses `"gpt2"`.
- [ ] **Step 1: Write the failing tests:**

```rust
#[test] fn qwen2_splits_words_with_their_leading_space()  // "Hello world" -> ["Hello", " world"]
#[test] fn qwen2_splits_a_space_run_before_a_word()       // "a  b" -> ["a", " ", " b"]
#[test] fn qwen2_splits_newline_runs()                    // "x\n\ny" -> ["x", "\n\n", "y"]
#[test] fn qwen2_splits_digits_one_by_one()               // "2024" -> ["2", "0", "2", "4"]
#[test] fn qwen2_splits_contractions()                    // "don't" -> ["don", "'t"]
#[test] fn qwen2_keeps_accented_letters_together()        // "Ol\u{e1}, meu" -> ["Ol\u{e1}", ",", " meu"]
#[test] fn gpt2_keeps_digit_runs()                        // "2024" -> ["2024"]
#[test] fn absent_pre_uses_the_default_patterns()         // from_name(None) splits "a  b" like gpt2
#[test] fn unknown_pre_is_unsupported()                   // UnsupportedPreTokenizer("mystery")
#[test] fn encode_keeps_whitespace()                      // gpt2 tokenizer: decode(encode("a  b\n")) == "a  b\n"
```

- [ ] **Step 2: Run `cargo test -p rusticide-tokenizer`; expect FAIL.**
- [ ] **Step 3: Implement; replace `gpt2_pre_tokenize`'s
  `split_whitespace` with `PreTokenizer::split`.**
- [ ] **Step 4: Run; PASS. `mise run lint`, `mise run test`,
  `mise run security` (the new dependency passes cargo deny and audit).**
- [ ] **Step 5: Fuzz:** extend `gguf_tokenizer` (or add a target) so a
  fuzzed GPT-2 tokenizer also `encode`s a fuzzed string; 60 s, no crash.
- [ ] **Step 6: Commit**
  `feat(tokenizer): pre-tokenize gpt2 text with llama.cpp's patterns`.

---

### Task 6: End-of-generation tokens

**Files:**

- Modify: `crates/rusticide-tokenizer/src/{tokenizer,from_gguf}.rs` (+
  tests), `crates/rusticide-model/src/generate.rs`, `generate_tests.rs`

**Interfaces:**

- Produces: `pub fn is_end_of_generation(&self, token: u32) -> bool` on
  `Tokenizer`, true for `eos_id`, for `tokenizer.ggml.eot_token_id` when
  present, and for any vocabulary token whose text is in llama.cpp's
  end-of-generation text list at the reference commit (it includes
  `<|endoftext|>` and `<|im_end|>`). Stored as a sorted `Vec<u32>`;
  `Tokenizer::new` yields `vec![eos_id]`.
- Behaviour: `generate` stops on `is_end_of_generation(next)` instead of
  `next == eos_id()`; the stopping token is never emitted.
- [ ] **Step 1: Write the failing tests:**
  `eos_ends_generation`, `endoftext_ends_generation` (a GPT-2 vocabulary
  holding `<|endoftext|>` whose EOS is another id),
  `eot_token_id_ends_generation`, `ordinary_tokens_do_not_end_generation`,
  and in `rusticide-model`
  `generation_stops_at_any_end_of_generation_token` (scripted model emits
  `<|endoftext|>`; output excludes it).
- [ ] **Step 2: Run; FAIL. Step 3: Implement. Step 4: PASS; lint;
  coverage.**
- [ ] **Step 5: Commit** `feat(tokenizer): stop generation on every end-of-generation token`.

---

### Task 7: The qwen2 variant, configuration and half-split RoPE

**Files:**

- Create: `crates/rusticide-llama/src/variant.rs` (+ tests)
- Modify: `crates/rusticide-kernels/src/{rope,lib}.rs`,
  `crates/rusticide-llama/src/{config,attention,model,lib}.rs` and their
  tests

**Interfaces:**

- Produces:
  - `pub fn rope_half_split(row: &mut [f32], position: usize, theta: f32,
    dim_pairs: usize)` in `rusticide-kernels`: rotates `(p, p + row.len() /
    2)` for `p < min(dim_pairs, row.len() / 2)` by angle
    `position * theta^(-2p / row.len())` (llama.cpp rope mode NEOX).
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Variant
    { Llama, Qwen2 }` with `pub const fn architecture(self) -> &'static str`
    (`"llama"`, `"qwen2"`), `pub const fn rope(self) -> RopeStyle`, and
    `pub const fn requires_attention_bias(self) -> bool` (Qwen2 only).
  - `pub enum RopeStyle { Adjacent, HalfSplit }` with
    `pub fn apply(self, row: &mut [f32], position: usize, theta: f32,
    dim_pairs: usize)` calling the matching kernel.
  - `LlamaConfig` gains `pub variant: Variant`;
    `LlamaConfig::from_source(source: &dyn ModelSource, variant: Variant)`.
    Missing-key errors name `{architecture}.<key>` (e.g.
    `MissingMetadata { key: "qwen2.context_length" }`), and the RoPE
    dimension key is `{architecture}.rope.dimension_count`.
  - `Llama::from_source(source, variant)`; `load` passes `Variant::Llama`.
- [ ] **Step 1: Write the failing tests:**

```rust
#[test] fn rope_half_split_rotates_halves() {
    let mut row = [1.0, 1.0, 0.0, 0.0]; rope_half_split(&mut row, 1, 10000.0, 2);
    let slow = 10000.0_f32.powf(-2.0 / 4.0);
    assert_close(&row, &[1.0_f32.cos(), slow.cos(), 1.0_f32.sin(), slow.sin()]);
}
#[test] fn rope_half_split_at_position_zero_is_identity()
#[test] fn variants_name_their_architecture_and_rope()
#[test] fn qwen2_config_reads_qwen2_keys()             // tiny qwen2 metadata -> variant Qwen2, sizes
#[test] fn qwen2_config_names_qwen2_keys_when_missing() // MissingMetadata { key: "qwen2.context_length" }
#[test] fn qwen2_attention_uses_half_split_rope()
    // one head, q = k = identity-like: scores at position 1 equal those computed with rope_half_split
```

- [ ] **Step 2: Run `cargo test -p rusticide-kernels -p rusticide-llama`;
  expect FAIL. Step 3: Implement. Step 4: PASS; lint; coverage.**
- [ ] **Step 5: Commit** `feat(llama): parameterize the family by variant and rope style`.

---

### Task 8: The qwen2 variant, attention biases, loader and registration

**Files:**

- Modify: `crates/rusticide-llama/src/{weights,attention,model,lib,
  testing}.rs` and tests,
  `crates/rusticide-fuzz/fuzz_targets/llama_load.rs`,
  `crates/rusticide/src/cli/registry.rs`, `crates/rusticide/src/cli/tests.rs`

**Interfaces:**

- Produces:
  - `BlockWeights` gains `attn_q_bias: Option<Vec<f32>>` (length E),
    `attn_k_bias` and `attn_v_bias` (length KV*D), loaded from
    `blk.{i}.attn_{q,k,v}.bias` when present; absent with
    `variant.requires_attention_bias()` is `MissingTensor`.
  - Attention adds each present bias right after its projection, before
    RoPE (llama.cpp's `qwen2` graph).
  - `pub fn load_qwen2(source: &dyn ModelSource) -> Result<Box<dyn Model>>`.
  - `testing::TinySource::qwen2()` and `pub fn tiny_qwen2_source() ->
    InMemorySource`: one block, architecture `qwen2` with `qwen2.*` keys,
    the three biases, and GPT-2 tokenizer metadata (the 256 byte tokens
    plus `<|endoftext|>` as BOS and `<|im_end|>` as EOS,
    `tokenizer.ggml.pre = "qwen2"`, `add_bos_token = false`, no merges).
  - CLI registry: `Registry::new().with("llama", rusticide_llama::load)
    .with("qwen2", rusticide_llama::load_qwen2)`.
- [ ] **Step 1: Write the failing tests:**
  - `qwen2_adds_the_value_bias`: zero `attn_v`, non-zero
    `attn_v.bias`, identity-like `attn_output`, zero FFN -> the attention
    delta equals the bias (computed in the test from literal numbers);
  - `qwen2_requires_attention_biases` (`MissingTensor { name:
    "blk.0.attn_q.bias" }`);
  - `llama_without_biases_still_loads` (the stock tiny llama);
  - `load_qwen2_returns_a_boxed_model`;
  - CLI: `run_streams_text_from_a_tiny_qwen2_model` (greedy, out
    non-empty), and the unknown-architecture test now expects
    `supported: llama, qwen2`.
- [ ] **Step 2: Run; FAIL. Step 3: Implement. Step 4: PASS; lint;
  coverage.**
- [ ] **Step 5: Fuzz:** `llama_load` also calls `load_qwen2`; 60 s, no
  crash.
- [ ] **Step 6: Commit** `feat(llama): run qwen2 models`.

---

### Task 9: Reference parity with llama.cpp on Qwen2.5

**Files:**

- Create: `crates/rusticide-llama/tests/fixtures/qwen2.5-0.5b-instruct-q4_k_m.json`
- Modify: `crates/rusticide-llama/tests/reference.rs`, `mise.toml`
  (`test:reference` description), `AGENTS.md` (the `RUSTICIDE_MODELS`
  line names both files)

**Interfaces:**

- Fixture JSON: milestone 1's schema; `reference.version` is
  `llama-server --version`; `settings` `"temperature 0, n_predict 32,
  add_special true"`.
- Prompts: `"The capital of France is"`, `"Once upon a time,"`,
  `"Ol\u00e1, meu nome \u00e9"`, and `"Line one\n\n  2024 has 365 days."`
  (exercises newlines, a space run and single digits).
- [ ] **Step 1: Check the model** in `$RUSTICIDE_MODELS`: `shasum -a 256`
  equals the Global Constraints hash.
- [ ] **Step 2: Capture the fixture** with `llama-server -m <file>` (no
  script kept in the repo), exactly as milestone 1 Task 8 Step 2. llama.cpp
  stops at an end-of-generation token, so a case may hold fewer than 32
  generated tokens; record what it returns.
- [ ] **Step 3: Generalize the reference test.** One
  `fn check_fixture(fixture: &str) -> TestResult` loads the file named by
  the fixture through `Registry::new().with("llama", load).with("qwen2",
  load_qwen2)` (so architecture detection is tested too); tests
  `tinyllama_matches_llama_cpp_greedy` and `qwen2_matches_llama_cpp_greedy`
  each skip when `RUSTICIDE_MODELS` is unset. When a case has fewer than 32
  generated tokens, assert that rusticide's top-1 at the next position is
  an end-of-generation token (`is_end_of_generation`).
- [ ] **Step 4: Run** `RUSTICIDE_MODELS=~/models cargo test --release -p
  rusticide-llama --test reference -- --nocapture`. A failure is a bug in
  Tasks 1-8: fix it there test-first, never by editing the fixture.
- [ ] **Step 5: Measure speed** on the M1 Pro:
  `target/release/rusticide run <qwen file> --prompt "Once upon a time," -n
  64 --temperature 0`, three runs; keep the `tokens/s` lines for Task 10.
- [ ] **Step 6: Commit** `test(llama): pin greedy parity with llama.cpp on Qwen2.5`.

---

### Task 10: Documentation and the 0.4.0 release preparation

**Files:**

- Modify: `README.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`,
  `docs/plan/2026-10-22-rusticide-run-design.md`
- [ ] **Step 1: Docs.**
  - `README.md` "Run a model": supported families `llama` and `qwen2`,
    both tested models, parity against llama.cpp 0.8.0 for Qwen2.5, and
    the Task 9 tokens/s with the machine it was measured on.
  - `docs/ARCHITECTURE.md`: the block-format table, lazy rejection and the
    GPT-2 tokenizer path.
  - `AGENTS.md`: "Current plan" points to this plan.
  - Design, "Left to the builder": `> **Resolved 2026-10-31.**` the
    pre-tokenizer uses `fancy-regex` with llama.cpp's patterns verbatim.
  - `mise run markdownlint` passes; ASCII only.
- [ ] **Step 2: Full gate:** `mise run ci:quick` passes.
- [ ] **Step 3: Commit** `docs: document qwen2 support`.
- [ ] **Step 4: Release prep** (human-approved step): `mise run
  release:prepare` (expect 0.4.0), `mise run changelog --tag v0.4.0`,
  commit `chore(release): prepare for v0.4.0`. Merging and pushing stay
  with the human.
