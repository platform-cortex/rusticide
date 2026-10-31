<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide run, Milestone 1 (TinyLlama) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rusticide run tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "..."`
streams text whose first 32 greedy tokens match llama.cpp, released as 0.3.0.

**Architecture:** Three new crates. `rusticide-kernels` holds slice-level f32
kernels shared by the existing graph crate and the models.
`rusticide-model` holds the family-independent runtime: the `Model` contract,
the architecture registry, the KV cache and the generation loop.
`rusticide-llama` implements the `llama` family. The CLI's `run` subcommand
only wires them together.

**Tech Stack:** Rust 2024 (MSRV 1.96), rayon, clap, thiserror/anyhow,
serde_json (dev only), llama.cpp `llama-server` as the reference (outside the
repo).

**Spec:** `docs/plan/2026-10-22-rusticide-run-design.md`. Read it first; this
plan implements its milestone 1 and nothing from milestones 2-3.

## Global Constraints

- AGENTS.md applies in full: strict TDD (watch each test fail first), 100%
  tarpaulin coverage, no `unwrap`/`expect` outside `#[cfg(test)]`, no
  `unsafe` (only `rusticide-alloc` may use it), files under 300 lines tests
  included, ASCII only, the three-line SPDX header on every `.rs` file,
  thiserror in libraries and anyhow in the binary, GPG-signed conventional
  commits with no trailers.
- MSRV 1.96: no std APIs newer than 1.96.
- New crates live under `crates/` (the workspace glob picks them up). Each
  gets a `[workspace.dependencies]` entry in the root `Cargo.toml` with
  `version = "0.2.0"` (the release task moves it), and copies the
  `[package]` block and `[lints] workspace = true` of
  `crates/rusticide-sampling/Cargo.toml`.
- No panics on hostile input: indexing that could go out of bounds on
  file-controlled sizes is a bug; validate at load time and use checked
  arithmetic on file-controlled numbers.
- Weights are dequantized to f32 once at load; the file is read whole
  (`GgufFile::load`). No mmap.
- Kernels have exactly one implementation each (spec anti-goal: no copies).
- Model files are never committed. Reference tests skip unless
  `RUSTICIDE_MODELS` names a directory holding the model file.
- Success criterion (spec): prompt tokens equal the reference's; the first
  **32** generated greedy tokens equal the reference's, except that a
  divergence passes when rusticide's own top-1 and top-2 logits at that
  position differ by less than **0.1** (comparison stops there).
- CLI defaults (spec): `-n/--max-tokens 128`, `--temperature 0.8` (0 means
  greedy), `--top-k 40`, `--top-p 0.95`, `--seed` random when absent.
- Canonical model: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`,
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`; its SHA-256 goes in the fixture.

## Review Focus

1. **Prompt at or past the context length** -> `ModelError::ContextFull`
   before any forward pass; generation that reaches the context length
   stops cleanly with the tokens so far. Tests: Task 4
   `prompt_filling_the_context_is_context_full`,
   `generation_stops_at_the_context_length`.
2. **Hostile metadata** (zero heads, embedding not divisible by heads, KV
   heads not dividing heads, odd head dim, absurd sizes) -> `InvalidConfig`,
   never a panic or a giant allocation. Tests: Task 5 `config_rejects_*`;
   fuzz target in Task 6.
3. **Tensor shape disagreeing with metadata** -> `ShapeMismatch` naming the
   tensor. Tests: Task 2 `matrix_with_wrong_shape_is_shape_mismatch`, Task 5
   `weights_reject_a_mis_shaped_projection`.
4. **A multi-byte UTF-8 character split across tokens while streaming**
   (Portuguese prompts) -> the stream never prints U+FFFD for a character
   that later completes. Test: Task 4
   `streaming_holds_back_an_incomplete_character`.
5. **Empty prompt** -> generation starts from BOS alone and works. Test:
   Task 4 `empty_prompt_generates_from_bos`.

---

### Task 1: rusticide-kernels, the shared f32 kernels

**Files:**

- Create: `crates/rusticide-kernels/Cargo.toml`, `src/lib.rs`, `src/dot.rs`,
  `src/norm.rs`, `src/rope.rs`, `src/elementwise.rs` (tests in each file's
  `#[cfg(test)] mod tests` or a sibling `*_tests.rs` to stay under 300 lines)
- Modify: `Cargo.toml` (workspace dependency),
  `crates/rusticide-ggml/Cargo.toml`, `crates/rusticide-ggml/src/ops/norm.rs`,
  `crates/rusticide-ggml/src/ops/rope.rs`,
  `crates/rusticide-ggml/src/ops/unary.rs`

**Interfaces:**

- Produces (crate `rusticide_kernels`, depends only on rayon):
  - `pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32`
  - `pub fn matvec(weight: &[f32], cols: usize, input: &[f32], output: &mut [f32])`
    -- `weight` is row-major `[output.len(), cols]`; row `r` of the result
    is `dot(weight row r, input)`; rows in parallel with rayon.
  - `pub fn rms_norm(row: &mut [f32], eps: f32)` -- divides by
    `sqrt(mean(x^2) + eps)`; empty row is a no-op.
  - `pub fn softmax(row: &mut [f32])` -- max-subtracted; empty row no-op.
  - `pub fn silu(value: f32) -> f32`
  - `pub fn mul_assign(target: &mut [f32], factor: &[f32])`,
    `pub fn add_assign(target: &mut [f32], addend: &[f32])`
  - `pub fn rope_adjacent(row: &mut [f32], position: usize, theta: f32, dim_pairs: usize)`
    -- rotates pairs `(2p, 2p+1)` for `p < min(dim_pairs, row.len() / 2)` by
    angle `position * theta^(-2p / row.len())`.
- Mismatched slice lengths never panic: kernels iterate with `zip`, so the
  shorter slice bounds the work. Document that callers validate lengths.
- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn dot_of_known_vectors() { assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0); }
#[test] fn dot_spans_more_than_one_lane_block() {
    let a: Vec<f32> = (1..=19).map(|v| v as f32).collect();
    assert_eq!(dot(&a, &vec![1.0; 19]), 190.0);
}
#[test] fn matvec_multiplies_each_row() {
    let mut out = [0.0; 2];
    matvec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, &[1.0, 0.0, -1.0], &mut out);
    assert_eq!(out, [-2.0, -2.0]);
}
#[test] fn matvec_with_zero_columns_writes_zeros() {
    let mut out = [9.0; 2]; matvec(&[], 0, &[], &mut out); assert_eq!(out, [0.0, 0.0]);
}
#[test] fn rms_norm_divides_by_root_mean_square() {
    let mut row = [1.0, 2.0, 3.0, 4.0]; rms_norm(&mut row, 0.0);
    let rms = 7.5_f32.sqrt();
    assert_eq!(row, [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]);
}
#[test] fn softmax_sums_to_one_and_survives_large_inputs() {
    let mut row = [1000.0, 1000.0]; softmax(&mut row); assert_eq!(row, [0.5, 0.5]);
}
#[test] fn rope_adjacent_rotates_by_position() {
    let mut row = [1.0, 0.0, 1.0, 0.0]; rope_adjacent(&mut row, 1, 10000.0, 2);
    let slow = 10000.0_f32.powf(-2.0 / 4.0);
    assert_close(&row, &[1.0_f32.cos(), 1.0_f32.sin(), slow.cos(), slow.sin()]);
}
#[test] fn rope_at_position_zero_is_identity() { /* row unchanged */ }
#[test] fn silu_of_zero_is_zero() { assert_eq!(silu(0.0), 0.0); }
#[test] fn mul_assign_and_add_assign_are_elementwise() { /* [1,2]*[3,4]=[3,8]; +[1,1]=[4,9] */ }
```

- [ ] **Step 2: Run `cargo test -p rusticide-kernels`; expect FAIL (crate
  missing / functions undefined).**

- [ ] **Step 3: Implement the kernels.** `dot` uses 8 independent
  accumulators over `chunks_exact(8)` plus a scalar tail, so the compiler
  can vectorize the reduction; `matvec` uses
  `output.par_iter_mut().zip(weight.par_chunks(cols))` and fills zeros when
  `cols == 0`.

- [ ] **Step 4: Run `cargo test -p rusticide-kernels`; expect PASS.**

- [ ] **Step 5: Move, don't copy.** Make `rusticide-ggml`'s `softmax`,
  `rms_norm`, `rope` (per row, `position` = row index, `dim_pairs` as
  today) and `silu` call these kernels; delete their local math. All
  existing `rusticide-ggml` tests must pass unchanged:
  `cargo test -p rusticide-ggml`.

- [ ] **Step 6: Verify and commit.** `mise run lint`, `mise run test`
  (100%).

```bash
git commit -S -m "feat(kernels): add shared f32 kernels and route ggml through them"
```

---

### Task 2: rusticide-model, errors, sources, weights and the KV cache

**Files:**

- Create: `crates/rusticide-model/Cargo.toml`, `src/lib.rs`, `src/error.rs`,
  `src/source.rs`, `src/weights.rs`, `src/cache.rs` (+ tests)
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: `rusticide_io::{GgufFile, Metadata, GgufError}`,
  `rusticide_core::{Tensor, DType}`, `rusticide_kernels::matvec`.
- Produces:
  - `pub enum ModelError` (thiserror), variants exactly:
    `Gguf(#[from] GgufError)`, `Tensor(#[from] rusticide_core::Error)`,
    `Tokenizer(#[from] TokenizerError)`, `Sampling(#[from] SamplingError)`,
    `MissingTensor { name: String }`, `MissingMetadata { key: String }`,
    `InvalidConfig { reason: &'static str }`,
    `ShapeMismatch { name: String, expected: Vec<usize>, got: Vec<usize> }`,
    `UnknownArchitecture { name: String, supported: String }`,
    `ContextFull { limit: usize }`,
    `CachePosition { expected: usize, got: usize }`.
    Messages lowercase; `UnknownArchitecture` reads
    `unknown architecture {name}; supported: {supported}`.
  - `pub type Result<T> = std::result::Result<T, ModelError>;`
  - `pub trait ModelSource { fn metadata(&self) -> &Metadata; fn tensor_shape(&self, name: &str) -> Option<Vec<usize>>; fn tensor(&self, name: &str) -> Result<Tensor>; }`
    implemented for `GgufFile` (absent name -> `MissingTensor`).
  - `pub struct InMemorySource` with `pub fn new(metadata: Metadata) -> Self`
    and `#[must_use] pub fn with_tensor(self, name: &str, tensor: Tensor) -> Self`.
  - `pub struct Matrix` with `pub fn load(source: &dyn ModelSource, name: &str, rows: usize, cols: usize) -> Result<Self>`,
    `pub const fn rows(&self) -> usize`, `pub const fn cols(&self) -> usize`,
    `pub fn row(&self, index: usize) -> Option<&[f32]>`,
    `pub fn matvec(&self, input: &[f32], output: &mut [f32])`.
  - `pub fn load_vector(source: &dyn ModelSource, name: &str, len: usize) -> Result<Vec<f32>>`
  - `pub struct KvCache` with `pub fn new(layers: usize, width: usize, capacity: usize) -> Self`,
    `pub fn store(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) -> Result<()>`,
    `pub fn keys(&self, layer: usize) -> &[f32]`,
    `pub fn values(&self, layer: usize) -> &[f32]`, `pub fn clear(&mut self)`.
    Storage grows on `store` (no up-front allocation of `capacity`).
- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn gguf_source_reports_a_missing_tensor() // MissingTensor { name: "nope" }
#[test] fn in_memory_source_returns_its_tensors()
#[test] fn matrix_loads_rows_and_multiplies()   // 2x3 matrix, row(1) == Some(&[4.,5.,6.]), matvec
#[test] fn matrix_with_wrong_shape_is_shape_mismatch()
    // expected [3, 2], got [2, 3] -> ShapeMismatch { name, expected: vec![3, 2], got: vec![2, 3] }
#[test] fn matrix_widens_f16_weights()          // F16 tensor loads as the same f32 values
#[test] fn load_vector_checks_length()          // ShapeMismatch on wrong length
#[test] fn cache_appends_per_layer()            // store(0,0,..), store(0,1,..) -> keys(0) has 2 rows
#[test] fn cache_rejects_a_skipped_position()   // CachePosition { expected: 0, got: 1 }
#[test] fn cache_rejects_a_full_context()       // capacity 1: store at 1 -> ContextFull { limit: 1 }
#[test] fn cache_rejects_an_unknown_layer()     // InvalidConfig { reason: "layer out of range" }
#[test] fn cache_clear_empties_every_layer()
```

- [ ] **Step 2: Run `cargo test -p rusticide-model`; expect FAIL.**
- [ ] **Step 3: Implement.** `Matrix::load` checks `tensor_shape(name) ==
  [rows, cols]` before materializing; reads F32 with `to_vec::<f32>()` and
  F16 with `to_vec::<half::f16>()` widened; any other dtype is
  `rusticide_core::Error` via `?`.
- [ ] **Step 4: Run tests; expect PASS. `mise run lint`, `mise run test`.**
- [ ] **Step 5: Commit**
  `feat(model): add model sources, f32 weights and the KV cache`.

---

### Task 3: rusticide-model, the Model contract and the registry

**Files:**

- Create: `crates/rusticide-model/src/model.rs`,
  `crates/rusticide-model/src/registry.rs` (+ tests)

**Interfaces:**

- Produces:
  - `pub trait Model { fn vocab_size(&self) -> usize; fn context_length(&self) -> usize; fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>>; fn reset(&mut self); }`
    -- `forward` returns logits for the token after `token` at `position`;
    positions must arrive in order from 0 after `reset`.
  - `pub type Loader = fn(&dyn ModelSource) -> Result<Box<dyn Model>>;`
  - `pub struct Registry` with `pub const fn new() -> Self`,
    `#[must_use] pub fn with(self, architecture: &'static str, loader: Loader) -> Self`,
    `pub fn architectures(&self) -> Vec<&'static str>`,
    `pub fn load(&self, source: &dyn ModelSource) -> Result<Box<dyn Model>>`.
- [ ] **Step 1: Failing tests** (with a test-only `Echo` model whose logits
  put 1.0 at index `token`):
  `registry_loads_by_architecture`,
  `registry_without_architecture_key_is_missing_metadata`
  (`MissingMetadata { key: "general.architecture" }`),
  `registry_rejects_an_unknown_architecture`
  (`UnknownArchitecture { name: "gptj", supported: "echo, llama2" }` for a
  registry with two entries, joined `", "` in registration order),
  `registry_lists_architectures`.
- [ ] **Step 2: Run; expect FAIL. Step 3: Implement. Step 4: Run; PASS;
  lint and coverage.**
- [ ] **Step 5: Commit** `feat(model): add the Model contract and architecture registry`.

---

### Task 4: rusticide-model, the generation loop

**Files:**

- Create: `crates/rusticide-model/src/generate.rs`,
  `crates/rusticide-model/src/stream.rs` (+ tests)

**Interfaces:**

- Consumes: `Model`, `rusticide_tokenizer::Tokenizer` (`encode`, `decode`,
  `eos_id`), `rusticide_sampling::{Sampler, SamplingParams}`.
- Produces:
  - `pub struct GenerateOptions { pub max_tokens: usize, pub params: SamplingParams }`
  - `pub struct Generation { pub prompt_tokens: Vec<u32>, pub generated_tokens: Vec<u32>, pub elapsed: std::time::Duration }`
  - `pub fn generate(model: &mut dyn Model, tokenizer: &Tokenizer, prompt: &str, options: &GenerateOptions, sampler: &mut Sampler, emit: &mut dyn FnMut(&str) -> Result<()>) -> Result<Generation>`
- Behaviour: encode the prompt (BOS handling is the tokenizer's); a prompt
  with `len >= model.context_length()` is `ContextFull { limit }` before
  any forward; `reset`, then forward every prompt token in order; then up
  to `max_tokens` times: pick the next token (`params.temperature == 0.0`
  -> `sample_greedy`, else `sample`), stop without emitting on
  `tokenizer.eos_id()`, emit the newly decoded text, stop when the next
  position would equal the context length, else forward it.
- Streaming (`stream.rs`): keep the text emitted so far; after each token,
  decode all generated tokens and emit only the new suffix, holding back
  a trailing U+FFFD until a later token completes it (emit it at the end
  if it never does).
- [ ] **Step 1: Failing tests** (tiny tokenizer built with
  `Tokenizer::new` and a scripted test model returning fixed logits):

```rust
#[test] fn greedy_generation_follows_the_argmax()
#[test] fn generation_stops_at_eos_without_emitting_it()
#[test] fn generation_stops_after_max_tokens()
#[test] fn prompt_filling_the_context_is_context_full()   // ContextFull { limit }
#[test] fn generation_stops_at_the_context_length()        // Ok, fewer than max_tokens
#[test] fn empty_prompt_generates_from_bos()
#[test] fn emit_errors_propagate()
#[test] fn streaming_holds_back_an_incomplete_character()
    // tokens whose bytes are 0xC3 then 0xA1 ("a" acute): emitted pieces == ["\u{e1}"], no "\u{fffd}"
#[test] fn streaming_flushes_a_never_completed_character()
```

- [ ] **Step 2: Run; FAIL. Step 3: Implement. Step 4: PASS; lint;
  coverage.**
- [ ] **Step 5: Commit** `feat(model): add the streaming generation loop`.

---

### Task 5: rusticide-llama, configuration and weights

**Files:**

- Create: `crates/rusticide-llama/Cargo.toml`, `src/lib.rs`,
  `src/config.rs`, `src/weights.rs`, `src/testing.rs` (+ tests)
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: `ModelSource`, `Matrix`, `load_vector`, `ModelError`.
- Produces:
  - `pub struct LlamaConfig { pub vocab_size: usize, pub embedding_length: usize, pub block_count: usize, pub feed_forward_length: usize, pub head_count: usize, pub head_count_kv: usize, pub head_dim: usize, pub context_length: usize, pub rope_freq_base: f32, pub rms_norm_eps: f32 }`
    with `pub fn from_source(source: &dyn ModelSource) -> Result<Self>`.
  - `pub struct LlamaWeights` with `pub fn load(source: &dyn ModelSource, config: &LlamaConfig) -> Result<Self>`;
    fields: `token_embd`, per-block `attn_norm`, `attn_q`, `attn_k`,
    `attn_v`, `attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`,
    `ffn_down`, then `output_norm` and `output: Option<Matrix>` (`None`
    when `output.weight` is absent: tied to `token_embd`).
  - Feature `testing` (and `cfg(test)`): `pub mod testing` with
    `pub fn tiny_source() -> InMemorySource` -- a valid 1-block llama with
    `embedding_length 4, head_count 2, head_count_kv 1, feed_forward_length 8,
    context_length 16`, vocab 6, deterministic small weights, and the
    tokenizer metadata `Tokenizer::from_gguf` needs (`tokenizer.ggml.model =
    "llama"`, tokens, scores, token types, BOS 1, EOS 2).
- Rules: `vocab_size` comes from `token_embd.weight`'s first dim;
  `head_dim = embedding_length / head_count`; `rope_freq_base` defaults to
  `10000.0`, `rms_norm_eps` to `1e-5`; `llama.rope.dimension_count`, when
  present, must equal `head_dim`. Tensor shapes are `[rows = outputs,
  cols = inputs]`: `attn_q [E, E]`, `attn_k`/`attn_v [KV*D, E]`,
  `attn_output [E, E]`, `ffn_gate`/`ffn_up [F, E]`, `ffn_down [E, F]`,
  `token_embd`/`output [V, E]`, norms `[E]` (E embedding, F feed-forward,
  KV kv heads, D head dim, V vocab). Names are
  `blk.{i}.attn_norm.weight` etc. as in GGUF.
- [ ] **Step 1: Failing tests:** `config_reads_tiny_source`,
  `config_defaults_rope_base_and_eps`, `config_requires_context_length`
  (`MissingMetadata { key: "llama.context_length" }`),
  `config_rejects_zero_heads`, `config_rejects_embedding_not_divisible_by_heads`,
  `config_rejects_kv_heads_not_dividing_heads`, `config_rejects_odd_head_dim`,
  `config_rejects_a_mismatched_rope_dimension` (each the exact
  `InvalidConfig { reason }` string you choose, asserted verbatim),
  `weights_load_tiny_source`, `weights_reject_a_mis_shaped_projection`
  (`ShapeMismatch` for `blk.0.attn_k.weight`),
  `missing_output_weight_ties_to_token_embedding`.
- [ ] **Step 2: Run; FAIL. Step 3: Implement with checked arithmetic on
  every metadata-derived size. Step 4: PASS; lint; coverage.**
- [ ] **Step 5: Commit** `feat(llama): read llama configuration and weights`.

---

### Task 6: rusticide-llama, the forward pass

**Files:**

- Create: `crates/rusticide-llama/src/attention.rs`, `src/feed_forward.rs`,
  `src/model.rs` (+ tests)
- Create: `crates/rusticide-fuzz/fuzz_targets/llama_load.rs`; modify
  `crates/rusticide-fuzz/Cargo.toml` (new `[[bin]]`, path dependency on
  `rusticide-llama` and `rusticide-model`)

**Interfaces:**

- Consumes: Task 5, `KvCache`, kernels (`matvec` via `Matrix`, `rms_norm`,
  `mul_assign`, `add_assign`, `rope_adjacent`, `softmax`, `silu`, `dot`).
- Produces: `pub struct Llama` implementing `Model`, and
  `pub fn load(source: &dyn ModelSource) -> Result<Box<dyn Model>>` (the
  registry `Loader`).
- Forward for one token at `position` (llama.cpp's `llama` graph, rope
  mode NORM):

```text
x = token_embd.row(token)                     (token >= vocab -> InvalidConfig "token out of range")
for each block:
  h = rms_norm(x) * attn_norm
  q = attn_q h; k = attn_k h; v = attn_v h
  rope_adjacent each D-wide head of q and k (position, rope_freq_base, D/2)
  cache.store(block, position, k, v)          (ContextFull / CachePosition surface here)
  for head i: kv = i / (head_count / head_count_kv)
     scores_t = dot(q_i, k_t[kv]) / sqrt(D) for t in 0..=position; softmax
     out_i = sum_t scores_t * v_t[kv]
  x += attn_output (concat out_i)
  h = rms_norm(x) * ffn_norm
  x += ffn_down( silu(ffn_gate h) * (ffn_up h) )
logits = (output or token_embd) (rms_norm(x) * output_norm)
```

  Scratch buffers live in `Llama` and are reused across calls.

- [ ] **Step 1: Failing tests** (built from `testing::tiny_source` with
  weights overridden per test):
  - `zero_blocks_leave_the_normed_embedding`: all attention and FFN weights
    zero, so logits equal `output * (x / sqrt(mean(x^2) + eps) * output_norm)`,
    computed in the test from the literal numbers;
  - `second_position_averages_values_when_queries_are_zero`: `attn_q`
    zero gives uniform attention, so the attention output at position 1 is
    the mean of the two value vectors (assert through an identity-like
    `attn_output` and zero FFN);
  - `grouped_query_heads_share_their_kv_head`: two query heads, one KV
    head, both heads read the same cached value;
  - `forward_out_of_order_is_cache_position`,
  - `forward_past_the_context_is_context_full`,
  - `reset_restarts_at_position_zero` (same logits after `reset`),
  - `token_out_of_range_is_invalid_config`,
  - `load_returns_a_boxed_model` (vocab 6, context 16).
- [ ] **Step 2: Run; FAIL. Step 3: Implement. Step 4: PASS; lint;
  coverage.**
- [ ] **Step 5: Fuzz target** `llama_load`: `GgufFile::from_bytes(data)`
  then `rusticide_llama::load(&file)`, results ignored. Run it per
  `crates/rusticide-fuzz` conventions (`mise run fuzz llama_load` for
  60 s); no crash. Commit the target only.
- [ ] **Step 6: Commit** `feat(llama): run the llama forward pass with a KV cache`.

---

### Task 7: The `run` subcommand

**Files:**

- Create: `crates/rusticide/src/cli/cmd/run.rs`,
  `crates/rusticide/src/cli/registry.rs`
- Modify: `crates/rusticide/Cargo.toml`, `crates/rusticide/src/lib.rs`,
  `crates/rusticide/src/cli/mod.rs`, `crates/rusticide/src/cli/cmd/mod.rs`,
  `crates/rusticide/src/cli/tests.rs` (split if it passes 300 lines)

**Interfaces:**

- Facade features: `models = ["io", "dep:rusticide-model",
  "dep:rusticide-llama", "dep:rusticide-tokenizer", "dep:rusticide-sampling"]`;
  `cli` adds `"models"`. `pub use rusticide_model as model;` under
  `models`. Dev-dependency: `rusticide-llama` with `features = ["testing"]`.
- `pub(crate) fn registry() -> Registry` in `cli/registry.rs`:
  `Registry::new().with("llama", rusticide_llama::load)` -- the one place
  families are registered.
- `Command::Run { model: PathBuf, #[arg(long)] prompt: String,
  #[arg(short = 'n', long, default_value_t = 128)] max_tokens: usize,
  #[arg(long, default_value_t = 0.8)] temperature: f32,
  #[arg(long, default_value_t = 40)] top_k: u32,
  #[arg(long, default_value_t = 0.95)] top_p: f32,
  #[arg(long)] seed: Option<u64> }`.
- `execute` gains the `err` writer. `cmd::run::run(args, out, err)` loads
  the file (`GgufFile::load`) and calls
  `cmd::run::run_source(source: &dyn ModelSource, args, out, err) -> anyhow::Result<()>`,
  which builds the tokenizer (`Tokenizer::from_gguf(source.metadata())`),
  the model (`registry().load(source)`), the sampler (`seed` or one derived
  from the system clock), streams to `out`, then writes one line to `err`:
  `prompt: {p} tokens, generated: {g} tokens, {rate:.2} tokens/s`.
- [ ] **Step 1: Failing tests:**
  `run_streams_text_from_a_tiny_model` (`run_source` on `tiny_source`,
  `--temperature 0`, out non-empty, err contains `tokens/s`, exit path Ok),
  `run_is_deterministic_with_a_seed`,
  `run_reports_a_missing_file` (`run(["rusticide","run","/nonexistent.gguf","--prompt","x"])`
  exits 1, err starts with `error:`),
  `run_reports_a_non_gguf_file` (a temp file with junk bytes),
  `run_reports_an_unknown_architecture`
  (err contains `unknown architecture gptj; supported: llama`),
  `run_requires_a_prompt` (exit 2), `help_lists_run`.
- [ ] **Step 2: Run; FAIL. Step 3: Implement. Step 4: PASS; lint;
  coverage; `cargo build --release -p rusticide --features cli` and
  `mise run release:snapshot` still pass (smoke runs `--version`).**
- [ ] **Step 5: Commit** `feat(rusticide): add the run subcommand`.

---

### Task 8: Reference parity with llama.cpp on TinyLlama

**Files:**

- Create: `crates/rusticide-llama/tests/reference.rs`,
  `crates/rusticide-llama/tests/fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.json`
- Modify: `crates/rusticide-llama/Cargo.toml` (dev-deps `serde`,
  `serde_json`, `rusticide-tokenizer`), `mise.toml` (task `test:reference`)

**Interfaces:**

- Fixture JSON (ASCII only; escape non-ASCII as `\uXXXX`):

```json
{
  "model": {"repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "sha256": "<hex>"},
  "reference": {"tool": "llama.cpp llama-server", "version": "<llama-server --version>",
                "settings": "temperature 0, n_predict 32, add_special true"},
  "cases": [{"prompt": "...", "prompt_tokens": [1, ...], "generated_tokens": [32 ids]}]
}
```

- Prompts: `"The capital of France is"`, `"Once upon a time,"`,
  `"Ol\u00e1, meu nome \u00e9"`.
- [ ] **Step 1: Get the model** into a directory outside the repo (e.g.
  `~/models`): download the canonical file from Hugging Face, record
  `shasum -a 256`. Confirm every tensor type is supported by loading it
  with `GgufFile::load` and `rusticide_llama::load`; if a type is
  unsupported, stop and report (it changes milestone scope).
- [ ] **Step 2: Capture the fixture** with `llama-server -m <file>` (not
  committed, no script kept in the repo): `POST /tokenize` with
  `{"content": prompt, "add_special": true}` for `prompt_tokens`, and
  `POST /completion` with `{"prompt": <prompt_tokens>, "n_predict": 32,
  "temperature": 0, "cache_prompt": false, "return_tokens": true}` for
  `generated_tokens`. Write the JSON.
- [ ] **Step 3: Write the reference test** `matches_llama_cpp_greedy`:
  returns early with a printed skip when `RUSTICIDE_MODELS` is unset;
  otherwise for each case asserts `tokenizer.encode(prompt) ==
  prompt_tokens`, then feeds the prompt and walks 32 positions: rusticide's
  top-1 must equal the fixture token, except that a mismatch where
  `top1_logit - top2_logit < 0.1` ends the case as a pass (print the
  position). Feed the fixture token at each step.
- [ ] **Step 4: Run** `RUSTICIDE_MODELS=~/models cargo test --release -p
  rusticide-llama --test reference -- --nocapture`. Any failure is a bug in
  Tasks 1-6 or the SentencePiece tokenizer: fix it there test-first (a
  failing unit test reproducing it, then the fix), never by editing the
  fixture.
- [ ] **Step 5: mise task** `test:reference` (alias `tref`), following the
  AGENTS.md task template (usage with `-v --verbose`), running the command
  above with `RUSTICIDE_MODELS` from the environment.
- [ ] **Step 6: Measure speed:** `target/release/rusticide run <file>
  --prompt "Once upon a time," -n 64 --temperature 0`; keep the
  `tokens/s` line for Task 9. Target 2 tokens/s; not a blocker.
- [ ] **Step 7: Commit** `test(llama): pin greedy parity with llama.cpp on TinyLlama`.

---

### Task 9: Documentation and the 0.3.0 release preparation

**Files:**

- Modify: `README.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`,
  `docs/plan/2026-08-21-rusticide-framework-design.md`
- [ ] **Step 1: Docs.**
  - `README.md`: a "Run a model" section with the `rusticide run` example,
    the supported family (`llama`, TinyLlama tested), and the measured
    tokens/s from Task 8 Step 6 with the machine it was measured on.
  - `docs/ARCHITECTURE.md`: the three new crates in the crate graph and
    layers.
  - `AGENTS.md`: "Current plan" points to this plan; mention
    `mise run test:reference` and `RUSTICIDE_MODELS`.
  - Framework design, "Sub-project decomposition": an
    `> **Amended 2026-10-22.**` note that the run design's three milestones
    come before SP2-SP7, linking `2026-10-22-rusticide-run-design.md`.
  - `mise run markdownlint` passes; ASCII only.
- [ ] **Step 2: Full gate:** `mise run ci:quick` passes.
- [ ] **Step 3: Commit** `docs: document rusticide run and the reordered roadmap`.
- [ ] **Step 4: Release prep** (human-approved step): `mise run
  release:prepare` (expect 0.3.0), `mise run changelog --tag v0.3.0`,
  commit `chore(release): prepare for v0.3.0`. Merging and pushing stay
  with the human.
