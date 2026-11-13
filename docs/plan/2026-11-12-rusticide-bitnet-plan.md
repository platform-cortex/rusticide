<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# BitNet b1.58 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rusticide run` runs BitNet b1.58 2B 4T (`i2_s`) and matches
bitnet.cpp's greedy tokens, with the decoder blocks shared with `llama`
and `qwen2` through a new `rusticide-transformer` crate.

**Architecture:** reference fixtures first (go/no-go). Then a pure move of
the decoder (attention, feed-forward, norms, batch helpers, the layer
loop) from `rusticide-llama` into `rusticide-transformer`, proven by
identical logits. Then `i2_s`: a per-row int8 activation quantization, a
format with a per-tensor scale trailer, a portable ternary kernel behind
`Matrix`/`Weights`. Then two hooks in the shared decoder (squared ReLU,
sub-norms) and a thin `rusticide-bitnet` family, registered in the facade.

**Tech Stack:** Rust 2024, rayon, `rusticide-bench`, bitnet.cpp (legacy
fork, built outside the repo) as the reference.

**Spec:** `docs/plan/2026-11-12-rusticide-bitnet-design.md` (read it; its
"Facts from recon" section and the ledger's `recon.md` carry the byte
layout and formulas every `i2_s` task relies on).

## Global Constraints

- MSRV 1.98, toolchain pin unchanged; no API newer than 1.98.
- `unsafe` only in `rusticide-alloc`/`rusticide-simd`; this plan adds none.
- No `unwrap`/`expect` in production code; 100% coverage; no panic on a
  hostile GGUF file.
- ASCII only, files under 300 lines (tests included), SPDX headers,
  bilingual docs, conventional signed commits (`git commit -S`), no
  attribution trailers.
- Never `target-cpu=native` in committed config.
- Ledger: `.superpowers/sdd/2026-11-12-rusticide-bitnet-plan/`
  (`progress.md`, `recon.md`, git-ignored).
- Model files live in `$RUSTICIDE_MODELS` (`~/models` on both machines):
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`, and (Task 1)
  `bitnet-b1.58-2b-4t-i2_s.gguf`, SHA-256
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
- The iMac Pro is `ssh elio@192.168.18.146` (tools need `zsh -lc '...'`).
  Sync the worktree with `rsync -a --delete --exclude target --exclude .git
  --exclude mise.local.toml ./ elio@192.168.18.146:~/rusticide-wt/` and
  build there. Never commit or push from the iMac Pro.
- Refactor tasks (3-5) and hook task (9) must leave TinyLlama and Qwen2.5
  logits bit-identical: the logit digest (Task 2) matches on the M1.
- `bench:gate` on the M1 after every task that touches kernels, the
  decoder or scheduling; quiet machine, AC power.

## Review Focus

1. A hostile `i2_s` tensor: a byte span shorter than `numel/4 + 32`, or a
   row width not a multiple of 128: a typed error at load, never a panic
   or an out-of-bounds read. Task 7 tests, fuzz target extended.
2. An all-zero activation row: the `1e-5` floor gives a finite scale,
   every quant is 0 and the product is exactly 0.0, never NaN. Task 6.
3. An activation exactly halfway between two quants: rounds half to even
   (`2.5 -> 2`, `3.5 -> 4`), as bitnet.cpp's legacy `nearest_int`. Task 6.
4. Code `3` (unused by the format) in a hostile file: decodes as
   `code - 1 = +2` in both dequantize and kernel, no panic. Tasks 7-8.
5. A `bitnet-b1.58` file missing a sub-norm, or an empty or over-long
   prompt: `MissingTensor`, `InvalidConfig` and `ContextFull` as for
   `llama`. Task 10.

---

## Part 1: the reference (go/no-go)

### Task 1: BitNet reference fixtures

**Files:**

- Create: `crates/rusticide/tests/fixtures/bitnet-b1.58-2b-4t-i2_s.json`
- Ledger: `fixtures-capture.md` (commands, versions, outputs)

**Interfaces:**

- Produces: a fixture with the schema of
  `crates/rusticide-llama/tests/fixtures/*.json` (`model.repo`,
  `model.file` = `bitnet-b1.58-2b-4t-i2_s.gguf`, `model.sha256`,
  `reference.tool`/`version`/`settings`, `cases[].prompt`,
  `prompt_tokens`, `generated_tokens` of 32).
- [ ] **Step 1:** On the iMac Pro, clone `microsoft/BitNet` outside any
  repo (`~/bitnet-ref`), check out `3b04140` with its submodule
  (`Eddie-Wang1120/llama.cpp`), and build per `setup_env.py` with
  `-q i2_s` and clang from `/usr/local/opt/llvm/bin`. Point it at the
  existing HF-cache file (copy or symlink as
  `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`); conversion must be
  skipped. Record both commits.
- [ ] **Step 2:** Find how the legacy binaries print token ids (e.g.
  `llama-cli --verbose-prompt`, `llama-tokenize`, or `llama-server`
  `/completion` with `n_probs`/token output). Capture, with
  `--temp 0 -n 32 -c 2048 -ngl 0` and BOS prepended, at least these
  prompts: `The capital of France is`, `def fibonacci(n):`,
  and a Portuguese prompt with accents, written in the JSON as
  `"A capital do Brasil \u00e9 Bras\u00edlia, e a maior cidade \u00e9"`
  (escapes keep the file ASCII; the tool receives the accented UTF-8).
- [ ] **Step 3:** Run the same capture twice: token ids must repeat. Then
  download the file to `~/models/bitnet-b1.58-2b-4t-i2_s.gguf` on both
  machines and check the SHA-256.
- [ ] **Step 4 (go/no-go):** If the legacy build fails or does not
  repeat, try BitNet `765741d` on the M1 (NEON, same 128 layout); if both
  fail or disagree, stop and report to the owner. No other oracle.
- [ ] **Step 5: Commit** `test(bitnet): capture the bitnet.cpp fixtures`.

## Part 2: `rusticide-transformer` (pure move)

### Task 2: logit digest and the facade-level reference test

**Files:**

- Ledger: `logit_digest.rs` (never committed)
- Move: `crates/rusticide-llama/tests/reference.rs` and `fixtures/` ->
  `crates/rusticide/tests/reference.rs`, `crates/rusticide/tests/fixtures/`
- Create: `crates/rusticide/src/models.rs`
- Modify: `crates/rusticide/src/cli/cmd/run.rs:69`, `mise.toml`
  (`test:reference` runs `-p rusticide --features models --test reference`)

**Interfaces:**

- Produces: `rusticide::models::registry() -> rusticide_model::Registry`
  (feature `models`; for now `rusticide_llama::registry()`), used by the
  CLI and the reference test. Task 10 adds BitNet here.
- Produces: the digest harness, an integration test kept in the ledger and
  copied into `crates/rusticide/tests/` only while it runs: for each
  TinyLlama and Qwen2.5 fixture prompt, `forward_batch(prompt)` then 8
  greedy `forward` steps, FNV-1a 64 over every logit's `to_bits()`;
  `RUSTICIDE_DIGEST=record` writes `logit-digest-<machine>.txt` to the
  ledger, `check` compares and fails on any difference.
- [ ] **Step 1:** Move the reference test and fixtures; switch it and
  `run.rs` to `rusticide::models::registry()`; `test:reference` green
  (2/2) on the M1.
- [ ] **Step 2:** Record the digest on the M1 twice; the two must be
  equal (determinism), else stop: the proof would be void.
- [ ] **Step 3: Commit** `test: run the reference parity from the facade`.

### Task 3: the crate and its leaf helpers

**Files:**

- Create: `crates/rusticide-transformer/` (`Cargo.toml` from
  `rusticide-llama`'s, `src/lib.rs`)
- Move from `rusticide-llama/src`: `batch.rs` (`for_each_row`,
  `zip_rows`, `PARALLEL_MIN_TOKENS`), `layout.rs` (`to_token_major`),
  `RopeStyle` from `variant.rs`, `normalize`/`normalize_rows` from
  `model.rs` (new `norm.rs`), with their tests.

**Interfaces:**

- Produces: `rusticide_transformer::{for_each_row, zip_rows,
  PARALLEL_MIN_TOKENS, to_token_major, RopeStyle, normalize,
  normalize_rows}`, same signatures as today, now `pub`.
- [ ] **Step 1:** Move; `rusticide-llama` imports them. No body changes.
- [ ] **Step 2:** `cargo test -p rusticide-transformer -p rusticide-llama`,
  clippy, digest `check` on the M1.
- [ ] **Step 3: Commit** `refactor(transformer): host the decoder helpers`.

### Task 4: attention and feed-forward

**Files:**

- Move: `attention.rs`, `feed_forward.rs` and their tests to
  `rusticide-transformer`.
- Create: `crates/rusticide-transformer/src/config.rs`,
  `src/weights.rs`

**Interfaces:**

- Produces: `pub struct DecoderConfig { embedding_length, block_count,
  feed_forward_length, head_count, head_count_kv, head_dim,
  context_length, vocab_size: usize; rope_freq_base, rms_norm_eps: f32;
  rope: RopeStyle; activation: Activation }` with `kv_dim()`.
- Produces: `pub enum Activation { Silu }` (Task 9 adds `ReluSquared`).
- Produces: `pub struct BlockWeights` (today's fields, moved) plus
  `attn_sub_norm: Option<Vec<f32>>`, `ffn_sub_norm: Option<Vec<f32>>`,
  both `None` until Task 9 reads them.
- `Attention::new(&DecoderConfig)`, `FeedForward::new(&DecoderConfig)`,
  `forward` signatures unchanged otherwise.
- [ ] **Step 1:** Move; `LlamaConfig` gains
  `fn decoder(&self) -> DecoderConfig`. No body changes.
- [ ] **Step 2:** Tests, clippy, digest `check`.
- [ ] **Step 3: Commit** `refactor(transformer): host attention and the
  feed-forward`.

### Task 5: the decoder model

**Files:**

- Move: the `Llama` struct, `forward_on_pool`, `embed` and the `Model`
  impl to `crates/rusticide-transformer/src/decoder.rs` as `Decoder`;
  `LlamaWeights` to `DecoderWeights`.
- Keep in `rusticide-llama`: `LlamaConfig` (metadata parsing), `Variant`,
  weight loading by tensor name, `load`/`load_qwen2`, `registry`.

**Interfaces:**

- Produces: `pub struct Decoder`; `Decoder::new(config: DecoderConfig,
  weights: DecoderWeights) -> rusticide_model::Result<Self>`;
  `impl Model for Decoder`. `pub struct DecoderWeights { token_embd:
  Matrix, blocks: Vec<BlockWeights>, output_norm: Vec<f32>, output:
  Option<Matrix> }`.
- [ ] **Step 1:** Move; `rusticide_llama::load` builds a `Decoder`.
- [ ] **Step 2:** `mise run ci:quick` (100%), digest `check` on the M1,
  `test:reference` 2/2 on the M1 and (synced) the iMac Pro, `bench:gate`
  exit 0 on both machines.
- [ ] **Step 3:** `rusticide-llama` holds no attention, norm or batch
  code: `grep -rn "rms_norm\|softmax\|matmul_batch"
  crates/rusticide-llama/src` prints nothing.
- [ ] **Step 4: Commit** `refactor(transformer): host the decoder model`.

## Part 3: `i2_s`

### Task 6: per-row activation quantization

**Files:**

- Modify: `crates/rusticide-kernels/src/q8_input.rs` (+ tests file)

**Interfaces:**

- Produces: `Q8Kind::Row`: one scale for the whole row, the bitnet.cpp
  legacy formula; `scales` holds `s` (a divisor) repeated per 32-block,
  `sums` per block as today; `dequantize` divides by `s` for `Row`.

```rust
// max in f64 from 1e-5; s = (127 / max) as f32; q = round-half-even(x * s)
// clamped to [-128, 127]
```

- [ ] **Step 1: Failing tests:** `row_scale_follows_the_legacy_formula`
  (`[0.5, -2.0, 1.0]` -> `s == (127.0_f64 / 2.0) as f32`, quants
  `[32, -127, 64]`); `an_all_zero_row_quantizes_to_zeros`
  (`s == (127.0 / 1e-5) as f32`, quants 0); `halfway_rounds_to_even`
  (an input built so `x * s` is exactly `2.5`/`3.5` -> `2`/`4`);
  `row_clamps_to_i8` (`-128` reachable only by clamping); sums equal the
  per-block quant sums.
- [ ] **Step 2:** Implement; round half to even with
  `f32::round_ties_even`.
- [ ] **Step 3:** Tests, clippy; prove each test by breaking its line.
- [ ] **Step 4: Commit** `feat(kernels): quantize activations per row`.

### Task 7: the `i2_s` format

**Files:**

- Create: `crates/rusticide-quantize/src/i2_s.rs` (+ tests)
- Modify: `crates/rusticide-quantize/src/format.rs` (contract),
  `crates/rusticide-io` extent sizing if it does not already go through
  `byte_len`, `crates/rusticide-fuzz` (a target loading an `i2_s` tensor)

**Interfaces:**

- Produces: `BlockFormat` gains `trailer_bytes: usize` (0 for every
  existing format, 32 for `I2_S`) and
  `apply_trailer: Option<fn(trailer: &[u8], values: &mut [f32])>`;
  `byte_len(numel) = blocks * block_bytes + trailer_bytes`;
  `pub const fn trailer_bytes(&self) -> usize`.
- Produces: table entry `I2_S`, type id 36, block size 128, block bytes 32,
  `dequantize_block` writes `code - 1` per the layout (byte `k` -> weights
  `k`, `k+32`, `k+64`, `k+96`, high bits first), `apply_trailer`
  multiplies by the f32 at trailer offset 0.
- Produces: `pub fn i2_s_scale(trailer: &[u8]) -> Option<f32>`.
- [ ] **Step 1: Failing tests:** a hand-built 128-weight block whose byte
  `k` encodes weights `k..k+96` decodes to the expected `-1/0/+1`
  sequence; code `3` decodes to `+2`; `byte_len(6_553_600) ==
  1_638_432`; a tensor one byte short -> `DataTooShort`; numel not a
  multiple of 128 -> `UnalignedShape`; dequantize applies the trailer
  scale.
- [ ] **Step 2:** Implement; existing formats' tests unchanged.
- [ ] **Step 3:** Fuzz target runs 60 s clean (`mise run fuzz` per its
  task); tests, clippy, coverage of the new lines.
- [ ] **Step 4: Commit** `feat(quantize): read the i2_s format`.

### Task 8: the portable ternary kernel

**Files:**

- Create: `crates/rusticide-kernels/src/portable/ternary.rs` (+ tests)
- Modify: `crates/rusticide-kernels/src/matmul.rs`,
  `crates/rusticide-model/src/weights.rs`, `crates/rusticide-bench`
  (kernel rows `kernel/I2_S/row`, `kernel/I2_S/rows8`)

**Interfaces:**

- Produces: `Weights::Ternary { data: &[u8], row_bytes: usize, scale:
  f32 }`; the scheduler quantizes each token with `Q8Kind::Row` and
  writes `(dot as f32) / s * scale` per row (the legacy order: integer
  first, then divide by `s`, then multiply by the weight scale).
- Produces: `pub(crate) fn ternary_dot(row: &[u8], input: &Q8Input) -> i32`,
  the exact integer `sum((code - 1) * q)`.
- `Matrix::load` builds `Storage::Ternary` when the format has a trailer
  (scale read once at load), `copy_row` decodes `code - 1` times scale.
- [ ] **Step 1: Failing tests:** `ternary_dot` equals a naive dequantize-
  then-integer-dot on random rows (all four codes); a `Matrix` from a
  synthetic `i2_s` tensor gives `matvec` equal to `(int / s) * scale`
  computed in the test; a zero row yields 0.0; multi-token batches equal
  single-token calls bit for bit.
- [ ] **Step 2:** Implement (rows in parallel like the other formats).
- [ ] **Step 3:** Bench rows appear; `bench:gate` on the M1 shows no
  existing row regressing (new rows are not in the baseline yet).
- [ ] **Step 4: Commit** `feat(kernels): run i2_s products portably`.

## Part 4: BitNet

### Task 9: squared ReLU and sub-norms in the decoder

**Files:**

- Modify: `crates/rusticide-kernels/src/portable/elementwise.rs`,
  `facade.rs`; `crates/rusticide-transformer` attention and feed-forward.

**Interfaces:**

- Produces: `rusticide_kernels::relu_squared_assign(values: &mut [f32])`
  (`max(x, 0)^2`, `0.0` for negative and `-0.0`).
- Produces: `Activation::ReluSquared`; the feed-forward applies it in
  place of SiLU. `attn_sub_norm` (RMS with the block's eps, then weight)
  on the mixed heads before `attn_output`; `ffn_sub_norm` on the gated
  hidden row before `ffn_down`; each only when `Some`.
- [ ] **Step 1: Failing tests:** `relu_squared_assign` on
  `[-2.0, -0.0, 0.0, 1.5, 3.0]` -> `[0.0, 0.0, 0.0, 2.25, 9.0]`; a small
  decoder with sub-norms equals a hand-composed forward; with both `None`
  the outputs equal Task 5's.
- [ ] **Step 2:** Implement; digest `check` on the M1 (llama path
  untouched).
- [ ] **Step 3: Commit** `feat(transformer): add squared ReLU and
  sub-norms`.

### Task 10: the `bitnet-b1.58` family

**Files:**

- Create: `crates/rusticide-bitnet/` (`config.rs`, `weights.rs`,
  `lib.rs`, tests, `testing.rs` for a synthetic tiny `bitnet-b1.58`
  source like `rusticide-llama`'s)
- Modify: `crates/rusticide/src/models.rs`, facade `Cargo.toml`

**Interfaces:**

- Produces: `rusticide_bitnet::load(source: &dyn ModelSource) ->
  Result<Box<dyn Model>>`, building a `Decoder` with `RopeStyle` NEOX
  (half-split), `Activation::ReluSquared`, sub-norms read from
  `blk.N.attn_sub_norm.weight` / `blk.N.ffn_sub_norm.weight`, tied
  `token_embd` (F16) as output. Metadata keys under `bitnet-b1.58.`.
- `rusticide::models::registry()` gains
  `.with("bitnet-b1.58", rusticide_bitnet::load)`.
- [ ] **Step 1: Failing tests:** config from a synthetic source (keys,
  defaults, errors); a missing sub-norm -> `MissingTensor`; the registry
  lists `bitnet-b1.58`; empty prompt and over-long prompt give the same
  errors as `llama`.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** With the real file on the M1: the fixture prompts
  tokenize to the fixture's ids (all of them).
- [ ] **Step 4 (checkpoint):** `rusticide run` on the M1, 64 tokens:
  record decode tokens/s. Below 2 tok/s: report to the owner (NEON
  `i2_s` into this milestone or not) before Task 11.
- [ ] **Step 5: Commit** `feat(bitnet): add the bitnet-b1.58 family`.

### Task 11: parity

**Files:**

- Modify: `crates/rusticide/tests/reference.rs` (a third case),
  `mise.toml` (`test:reference` description), `AGENTS.md`
  (`RUSTICIDE_MODELS` holds three files)
- [ ] **Step 1:** `test:reference` 3/3 on the M1 (`neon` and
  `RUSTICIDE_KERNELS=portable`) and the iMac Pro (`avx2`, `portable`).
  A divergence beyond the near-tie rule is a bug: debug (layer-by-layer
  against bitnet.cpp's intermediate values if needed), do not relax.
- [ ] **Step 2: Commit** `test(bitnet): check greedy parity with
  bitnet.cpp`.

## Finish

### Task 12: bench, results, docs, release

- [ ] **Step 1:** `rusticide-bench` gains `e2e/bitnet/prefill` and
  `e2e/bitnet/decode` when the file is present; re-record both baselines
  (quiet, AC), each its own commit `perf(bench): re-record the <machine>
  baseline`.
- [ ] **Step 2:** Fill the design's Results (EN and pt-BR): parity per
  prompt, BitNet prefill/decode per machine vs bitnet.cpp's numbers
  (`llama-bench` from the Task 1 build), the refactor proof, follow-ups
  (SIMD `i2_s`, attention/RoPE). Status done. `AGENTS.md` "Current
  plan" points at the next step.
- [ ] **Step 3:** `mise run ci:quick`, `security`, `markdownlint` clean.
- [ ] **Step 4: Commit** `docs: record the BitNet results`.
- [ ] **Step 5:** `mise run release:prepare` (0.10.0) after the owner
  approves the merge; `chore(release): prepare for v0.10.0`.
