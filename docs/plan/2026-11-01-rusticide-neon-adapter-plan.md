<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# NEON adapter milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close most of the prefill gap with llama.cpp on the M1 Pro in
three gated parts: per-token parallel work, top-k sampling, and a NEON
kernel adapter.

**Architecture:** Parts 1 and 2 are safe Rust in `rusticide-kernels`,
`rusticide-llama`, `rusticide-sampling` and `rusticide-model`. Part 3 adds
the `rusticide-simd` crate (the only new `unsafe`), a `neon` adapter in
`rusticide-kernels` selected at build time, and a row-group weight layout
that the adapter prepares in place at load.

**Tech Stack:** Rust 2024 (MSRV 1.98), rayon, `core::arch::aarch64` NEON
and dotprod intrinsics, bytemuck (`aarch64_simd`), cargo-tarpaulin, Miri,
GitHub Actions (`ubuntu-24.04-arm`).

**Spec:** `docs/plan/2026-11-01-rusticide-neon-adapter-design.md`

## Global Constraints

- Everything in AGENTS.md: strict TDD (see the test fail first), 100%
  coverage (`mise run test`), files under 300 lines including tests,
  ASCII only, the three-line SPDX header on every new `.rs` file,
  `-D warnings`, no `unwrap`/`expect` outside `#[cfg(test)]`,
  `cargo fmt` before every commit.
- MSRV 1.98: no std APIs newer than 1.98.
- `unsafe` only in `crates/rusticide-simd`, under `docs/UNSAFE.md`: a
  `// SAFETY:` comment on every block, safe public API, a README
  justification, an entry in `docs/UNSAFE.md`, Miri in CI.
- Never add `target-cpu=native` or `+dotprod` to committed build or
  release config; `+dotprod` appears only in the CI `arm` job's env.
- Within one adapter, a batched prefill stays bit-identical to running
  its tokens one at a time (the existing `matmul_batch` and model batch
  tests keep passing unchanged).
- `mise run test:reference` passes at NEAR_TIE 0.25 after every part.
- `mise run bench:gate` passes before each part's merge point; a baseline
  is re-recorded only in the tasks that say so (a reviewed change), on a
  quiet machine on AC power.
- Commits: GPG-signed (`git commit -S`), conventional, subject at most 50
  characters, body lines at most 72, no trailers of any kind.
- Bilingual docs: any `.md` changed under `docs/` changes its `.pt-BR.md`
  sibling in the same commit (ASCII, no accents).

## Review Focus

1. A batch of one token (decode) must not pay for new parallel regions:
   every parallel path in part 1 keeps a serial branch for `tokens == 1`,
   and the decode e2e rows of `bench:gate` must not regress.
2. Sampling with seeded randomness must keep producing the same tokens
   for the same seed and logits: part 2 is pinned against the current
   algorithm by an oracle test over random logits, ties included.
3. Logits with NaN, all `-inf`, or `top_k` larger than the vocabulary must
   behave as today (`NoValidTokens` or the full vocabulary), never panic.
4. A quantized matrix whose rows are not a multiple of the `neon` group,
   or the tied output/embedding matrix, must still load, run and decode
   rows correctly (fall back to `portable`, and `copy_row` reads the
   original row from any layout).
5. `RUSTICIDE_KERNELS` set to an unknown value or to `neon` on a build
   without the module must not panic: the value is reported once on
   stderr and `portable` or the compiled default is used.

---

## Part 1: per-token parallel work

### Task 1: quantize a batch's activations in parallel

**Files:**

- Modify: `crates/rusticide-kernels/src/matmul.rs` (`Activations::new`)
- Test: `crates/rusticide-kernels/src/matmul_tests.rs`

**Interfaces:**

- Consumes: `Q8Input::quantize(&mut self, &[f32], Q8Kind)`.
- Produces: no new API; `matmul_batch` keeps its signature and its
  bit-identity guarantee.
- [ ] **Step 1: Write the failing test.** Add
  `batched_quantization_matches_per_token_quantization`: for 1, 2, 7 and
  64 tokens of width 256 with deterministic values, the `Q8Input` of each
  token built by `Activations::new` equals (field by field: `scales`,
  `quants`, `sums`) a `Q8Input::quantize` of that token alone, for both
  `Q8Kind`s. It passes today; prove it by breaking the per-token loop
  (quantize token 0 for every slot), seeing it fail, restoring.
- [ ] **Step 2: Implement.** In `Activations::new`, quantize the tokens
  with `par_iter` when `tokens > 1` and serially otherwise; collect in
  token order.
- [ ] **Step 3: Verify.** `cargo test -p rusticide-kernels` passes, and
  `cargo clippy -p rusticide-kernels --all-targets -- -D warnings` is
  clean.
- [ ] **Step 4: Commit** `perf(kernels): quantize batch tokens in parallel`.

### Task 2: run the batch's per-token f32 work on the pool

**Files:**

- Modify: `crates/rusticide-llama/src/model.rs` (`normalize_rows`, the
  two `add_assign` calls), `crates/rusticide-llama/src/layout.rs`
  (`to_token_major`), `crates/rusticide-llama/src/feed_forward.rs`
  (`silu_assign`, `mul_assign`)
- Create: `crates/rusticide-llama/src/batch.rs` (the parallel helpers)
- Test: `crates/rusticide-llama/src/batch.rs` tests module,
  `layout.rs` tests

**Interfaces:**

- Produces, in `batch.rs`:
  - `pub(crate) const PARALLEL_MIN_TOKENS: usize` (the builder picks the
    value from the gate; start at 2).
  - `pub(crate) fn for_each_row(rows: &mut [f32], width: usize, tokens:
    usize, op: impl Fn(usize, &mut [f32]) + Sync)`: calls
    `op(token, row)` for each `width`-wide row, on the pool when
    `tokens >= PARALLEL_MIN_TOKENS`, serially otherwise.
  - `pub(crate) fn zip_rows(rows: &mut [f32], other: &[f32], width:
    usize, tokens: usize, op: impl Fn(&mut [f32], &[f32]) + Sync)`: the
    same over row pairs.
- `normalize_rows`, the residual `add_assign`s, the feed-forward
  `silu_assign`/`mul_assign` and `to_token_major` use these helpers.
  `to_token_major` parallelizes over output tokens (each output row
  gathers its column), keeping the `tokens == 1` copy path.
- [ ] **Step 1: Write the failing tests.** In `batch.rs`:
  `for_each_row_visits_every_row_once_with_its_index` (records indices
  into a `Mutex<Vec>`; sorted equals `0..tokens` for tokens 1, 2, 9) and
  `zip_rows_pairs_matching_rows`. In `layout.rs`, keep the existing tests
  and add `large_batches_match_the_serial_gather` (64 tokens x 96 rows
  against a reference written in the test).
- [ ] **Step 2: Run them** and see them fail to compile (no module).
- [ ] **Step 3: Implement** `batch.rs` with `rayon::prelude`, then route
  the call sites above through it.
- [ ] **Step 4: Verify.** `cargo test -p rusticide-llama` passes,
  including the unchanged batch bit-identity tests in
  `model_batch_tests.rs`.
- [ ] **Step 5: Commit** `perf(llama): run per-token batch work on the pool`.

### Task 3: bias and RoPE per token in parallel

**Files:**

- Modify: `crates/rusticide-llama/src/attention.rs` (`Attention::forward`)
- Test: `crates/rusticide-llama/src/attention_tests.rs`

**Interfaces:**

- Consumes: `batch::PARALLEL_MIN_TOKENS`.
- The per-token loop splits in two: first `add_bias` and
  `heads.rotate` for every token (on the pool when
  `tokens >= PARALLEL_MIN_TOKENS`, using `par_chunks_exact_mut` over the
  zipped query/key/value rows with the position `start + offset`), then
  the `cache.store` calls serially in position order.
- [ ] **Step 1: Write the failing test.**
  `batched_rotation_matches_one_token_at_a_time`: a 12-token batch
  through `Attention::forward` writes the same output and the same cache
  rows as 12 single-token calls (bit-identical). It passes today; prove
  it by rotating every token at `start` instead of `start + offset`.
- [ ] **Step 2: Implement** the split.
- [ ] **Step 3: Verify** `cargo test -p rusticide-llama`.
- [ ] **Step 4: Commit** `perf(llama): rotate batch tokens in parallel`.

### Task 4: part 1 gate and baseline

**Files:**

- Modify: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`
- Modify: `docs/plan/2026-11-01-rusticide-neon-adapter-design.md` and
  `.pt-BR.md` (append a `## Results` table row for part 1)
- [ ] **Step 1:** `mise run test:reference` passes.
- [ ] **Step 2:** `mise run bench:gate` exits 0 (prefill rows improve,
  decode rows within tolerance). If a decode row regresses, raise
  `PARALLEL_MIN_TOKENS` and re-run; record the chosen value.
- [ ] **Step 3:** Measure the thread count: `RAYON_NUM_THREADS=6` and
  `=8` runs of `mise run bench` with `RUSTICIDE_MODELS`; record both e2e
  prefill and decode numbers in the Results table. Change no code.
- [ ] **Step 4:** `mise run bench:baseline`, then `mise run bench:gate`
  passes against it.
- [ ] **Step 5:** `mise run ci:quick` passes.
- [ ] **Step 6: Commit** `perf(bench): re-record the M1 baseline` with the
  baseline and the Results rows.

## Part 2: sampling on the top-k survivors

### Task 5: a slice sampler that works on the survivors

**Files:**

- Create: `crates/rusticide-sampling/src/candidates.rs` (the survivor set
  and its filters)
- Modify: `crates/rusticide-sampling/src/sampler.rs`,
  `crates/rusticide-sampling/src/filters.rs`
- Test: `crates/rusticide-sampling/src/candidates_tests.rs`

**Interfaces:**

- Produces:
  - `pub fn Sampler::sample_logits(&mut self, logits: &[f32], params:
    &SamplingParams) -> Result<u32>` and `pub fn
    Sampler::greedy_logits(&self, logits: &[f32]) -> Result<u32>`: the
    slice API, treating NaN as `-inf` without copying the input.
  - `Sampler::sample` and `Sampler::sample_greedy` keep their signatures
    and delegate to the slice API after `extract_logits`.
- Algorithm (the builder writes it; these are the decisions):
  - When `0 < top_k < len`, select the k largest with
    `select_nth_unstable_by` over (index, logit) pairs, keep every logit
    equal to the k-th (ties, as today), then sort the survivors by index
    so every later sum runs in index order.
  - Temperature, softmax, top-p and the multinomial draw then run on the
    survivors only. Because the dropped logits contribute `exp(-inf) =
    0` to today's sums, summing the survivors in index order gives
    bit-identical probabilities.
  - With `top_k == 0` or `top_k >= len` the survivors are every finite
    logit, in index order.
- [ ] **Step 1: Write the failing tests** in `candidates_tests.rs`:
  `survivors_reproduce_the_full_vocabulary_sampler`: for 200 seeds, a
  vocabulary of 1000 random logits with injected ties, NaN and `-inf`,
  and params from `{temperature 0.0, 0.7, 1.3} x {top_k 0, 1, 40, 1000,
  5000} x {top_p 0.5, 0.95, 1.0}`, `sample_logits` returns the same token
  as a copy of today's `Sampler::sample` kept in the test file as
  `reference_sample`. Plus `all_non_finite_logits_are_no_valid_tokens`
  and `top_k_beyond_the_vocabulary_keeps_everything`.
- [ ] **Step 2: Run** and see them fail to compile.
- [ ] **Step 3: Implement** `candidates.rs`, then make `sample` delegate.
  Delete filter functions no caller uses any more.
- [ ] **Step 4: Verify** `cargo test -p rusticide-sampling` (the existing
  tests and `non_finite_tests` unchanged).
- [ ] **Step 5: Commit** `perf(sampling): filter only the top-k survivors`.

### Task 6: generate without copying logits, part 2 gate

**Files:**

- Modify: `crates/rusticide-model/src/generate.rs` (`pick`)
- Modify: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`, the
  design's Results rows (both languages)

**Interfaces:**

- Consumes: `Sampler::sample_logits`, `Sampler::greedy_logits`.
- `pick(sampler, logits: &[f32], params)` calls them directly; no
  `Tensor` is built per token.
- [ ] **Step 1:** The existing generate tests pass unchanged after the
  switch (`cargo test -p rusticide-model`); prove one covers `pick` by
  swapping greedy and sampled branches and seeing it fail.
- [ ] **Step 2:** `mise run test:reference` passes.
- [ ] **Step 3:** Measure Qwen decode with sampling (`rusticide run
  ... --temperature 0.8 --seed 1 --max-tokens 256`, three runs, before
  and after) and record it in the Results table.
- [ ] **Step 4:** `mise run bench:gate`, then `mise run bench:baseline`
  and the gate again; `mise run ci:quick` passes.
- [ ] **Step 5: Commit** `perf(model): sample logits without a copy`
  (code) and `perf(bench): re-record the M1 baseline` (baseline and
  Results).

## Part 3: the `neon` adapter

### Task 7: the `rusticide-simd` crate and its CI

**Files:**

- Create: `crates/rusticide-simd/{Cargo.toml,README.md,src/lib.rs,
  src/dot.rs}`
- Modify: root `Cargo.toml` (`[workspace.dependencies]`:
  `bytemuck = { version = ">=1,<2.0.0" }`, `rusticide-simd` path entry),
  `docs/UNSAFE.md` (entry), `deny.toml` only if bytemuck's license needs
  it, `.github/workflows/cicd.yml` (jobs `miri` and `arm`, both added to
  `tag.needs`), `AGENTS.md` (the `unsafe` line names both crates)

**Interfaces:**

- `Cargo.toml` copies `rusticide-alloc`'s `[lints]` (with
  `unsafe_code = "allow"` and `undocumented_unsafe_blocks = "deny"`);
  depends on `bytemuck` with feature `aarch64_simd`.
- `src/lib.rs`: the module `neon` exists only under
  `#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]`;
  `pub const AVAILABLE: bool` is true exactly then.
- First kernel, the template every later one follows:
  `pub fn dot_i8x32(lhs: &[i8; 32], rhs: &[i8; 32]) -> i32` in
  `neon::dot`, a safe wrapper around a private
  `#[target_feature(enable = "neon,dotprod")] fn` using `vdotq_s32`,
  with the one `unsafe` call and its `// SAFETY:` comment citing the
  build-time cfg; under `cfg(miri)` the wrapper calls a scalar body
  instead.
- CI `miri` job (in `cicd.yml`, ubuntu-latest, job name
  `Miri (rusticide-simd)`): toolchain
  `nightly-2026-08-03` with `miri` and `rust-src`,
  `cargo miri test -p rusticide-simd --target aarch64-apple-darwin`,
  `MIRIFLAGS=-Zmiri-strict-provenance`.
- CI `arm` job: `runs-on: ubuntu-24.04-arm`, the same steps as `gates`,
  `env: RUSTFLAGS: "-D warnings -C target-feature=+dotprod"`, job name
  `Gates arm64 (mise run ci:quick)`.
- New actions are pinned by SHA with
  `uv run --script scripts/update-actions-pins.py --write`; no hand
  edits.
- [ ] **Step 1: Write the failing test** `dot_matches_the_scalar_sum`
  (all 256 `i8` edge values at each position, plus 1000 random pairs)
  and `available_matches_the_build_target`.
- [ ] **Step 2: Run** `cargo test -p rusticide-simd` and see it fail.
- [ ] **Step 3: Implement** the crate, README and `docs/UNSAFE.md` entry
  (and its `.pt-BR.md` if one exists).
- [ ] **Step 4: Verify:** `cargo test -p rusticide-simd`; `cargo +nightly-2026-08-03
  miri test -p rusticide-simd --target aarch64-apple-darwin` locally;
  `mise run security` (bytemuck passes deny); `actionlint` via
  `mise run ci:quick`; `mise run ci:quick`.
- [ ] **Step 5: Commit** `feat(simd): add the NEON kernel crate` (crate,
  docs) and `ci: gate Miri and arm64` (workflow).

### Task 8: the `neon` adapter and its selection

**Files:**

- Create: `crates/rusticide-kernels/src/neon/mod.rs`
- Modify: `crates/rusticide-kernels/src/adapter.rs`,
  `crates/rusticide-kernels/Cargo.toml` (depend on `rusticide-simd`)
- Test: `crates/rusticide-kernels/src/adapter_tests.rs` (new; move the
  existing adapter test there)

**Interfaces:**

- `neon::ADAPTER: Adapter` named `"neon"`, compiled only when
  `rusticide_simd::AVAILABLE` (same cfg), starting as `portable`'s
  kernels (`..portable::ADAPTER` fields) so behaviour is unchanged.
- `fn choose(requested: Option<&str>, compiled: &'static Adapter) ->
  (&'static Adapter, Option<String>)`: `None` -> compiled default;
  `Some("portable")` -> portable; `Some("neon")` -> the compiled default
  if it is `neon`, else portable plus a warning; anything else ->
  compiled default plus a warning `"RUSTICIDE_KERNELS=<value>: unknown
  adapter, using <name>"`. `active()` calls it once with
  `std::env::var("RUSTICIDE_KERNELS").ok()` and prints the warning to
  stderr.
- [ ] **Step 1: Write the failing tests** for every `choose` row above
  (the warning text exact), and `active_adapter_is_the_compiled_default`
  (on this M1 build: `"neon"`; elsewhere `"portable"`).
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** `cargo test --workspace`; on the M1
  `RUSTICIDE_KERNELS=portable cargo test -p rusticide-kernels` passes.
- [ ] **Step 5: Commit** `feat(kernels): select a neon adapter at build`.

### Task 9: the row-group layout port

**Files:**

- Modify: `crates/rusticide-kernels/src/quant_kernel.rs`,
  `crates/rusticide-kernels/src/matmul.rs`,
  `crates/rusticide-kernels/src/adapter.rs`,
  `crates/rusticide-model/src/weights.rs`
- Test: `quant_kernel_tests.rs`, `matmul_tests.rs`, `weights_tests.rs`

**Interfaces:**

- `QuantKernel` gains:
  - `pub(crate) group_rows: usize` (1 for every `portable` kernel);
  - `pub(crate) prepare: Option<fn(data: &mut [u8], row_bytes: usize) ->
    bool>`: repacks whole groups in place, `false` (data untouched) when
    `data.len()` is not a multiple of `row_bytes * group_rows`;
  - `pub(crate) read_row: Option<fn(data: &[u8], row_bytes: usize, index:
    usize, out: &mut [u8]) -> bool>`: writes row `index`'s original block
    bytes;
  - `pub(crate) dot_group: Option<fn(group: &[u8], inputs: &[Q8Input],
    out: &mut [f32])>`: one group against every input,
    `out[row * inputs.len() + token]`.
- `pub fn Adapter::prepare_matrix(&self, type_id: u32, data: &mut [u8],
  row_bytes: usize) -> Option<&'static QuantKernel>`: this adapter's
  kernel when its `prepare` succeeds (or it has none), else the
  `portable` kernel for `type_id`, else `None`.
- `pub fn QuantKernel::read_row(&self, data: &[u8], row_bytes: usize,
  index: usize, out: &mut [u8]) -> bool`: the identity copy for
  `group_rows == 1`.
- The scheduler rounds `rows_per_task` up to a multiple of `group_rows`
  and, for `group_rows > 1`, calls `dot_group` per group.
- `Matrix::load` calls `active().prepare_matrix` on its owned copy;
  `Matrix::copy_row` reads the row through `kernel.read_row` into a
  `row_bytes` buffer before `dequantize_row`.
- [ ] **Step 1: Write the failing tests** with a test-only kernel in
  `test_support.rs` that "interleaves" 2 rows by swapping them in place
  (`group_rows: 2`): `prepare_matrix_falls_back_on_an_odd_row_count`,
  `read_row_restores_every_original_row`,
  `grouped_kernels_match_row_kernels` (scheduler output identical for 1,
  3 and 64 tokens) and, in `weights_tests.rs`,
  `copy_row_reads_a_prepared_matrix`.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** `cargo test --workspace` and
  `mise run test:reference` (portable path unchanged).
- [ ] **Step 5: Commit** `feat(kernels): add the row-group layout port`.

### Task 10: `Q4_K` on NEON

**Files:**

- Create: `crates/rusticide-simd/src/neon/q4_k.rs`,
  `crates/rusticide-kernels/src/neon/q4_k.rs` (layout: `prepare`,
  `read_row`; safe Rust)
- Modify: `crates/rusticide-kernels/src/neon/mod.rs` (register the kernel)
- Test: next to each file

**Interfaces:**

- Layout: groups of 8 rows (`group_rows = 8`), each group's blocks
  interleaved as llama.cpp's `block_q4_Kx8` (the builder confirms the
  byte order from ggml's `repack.cpp`; any bijective layout that keeps
  8 x 144 bytes per block group is acceptable).
- `rusticide_simd::neon::q4_k_group(group: &[[u8; 1152]], inputs: &[
  Q8Superblocks], out: &mut [f32])` where the builder defines
  `Q8Superblocks` as the fixed-size view of a `Q8Kind::SuperBlock`
  input that the kernel needs (the kernels crate builds it from
  `Q8Input`; all slicing to fixed-size arrays happens in safe code
  before the call).
- Tolerance: the f32 output matches `portable`'s `q4_k_rows` within
  `1e-5` relative (or `1e-6` absolute near zero); the integer partial
  sums are exact.
- [ ] **Step 1: Write the failing tests:** `prepare_then_read_row_is_identity`
  on random blocks; `neon_q4_k_matches_portable` for 1, 2, 7 and 64
  tokens over random weights; in `rusticide-simd`, the Miri-scalar body
  equals the NEON body.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement** layout, kernel and registration.
- [ ] **Step 4: Verify:** `cargo test --workspace`; Miri as in Task 7;
  `mise run test:reference`; `mise run bench:gate` shows
  `kernel/Q4_K/rows8` and the TinyLlama prefill row improving (record the
  numbers in the task report).
- [ ] **Step 5: Commit** `feat(simd): run Q4_K on NEON`.

### Task 11: `Q6_K` on NEON

Same shape as Task 10, for `Q6_K` (`block_q6_Kx8` analogue, 8 x 210
bytes per group). Tests `prepare_then_read_row_is_identity`,
`neon_q6_k_matches_portable`, the Miri-scalar equality. Commit
`feat(simd): run Q6_K on NEON`.

### Task 12: `Q5_0` and `Q8_0` on NEON

Same shape for the two block formats Qwen2.5 uses (32-element blocks,
22 and 34 bytes). The builder may use a simpler 4-row group if it meets
Goal 1 for Qwen2.5; record the choice. Commit
`feat(simd): run Q5_0 and Q8_0 on NEON`.

### Task 13: part 3 gate, baseline, results and required checks

**Files:**

- Modify: the M1 baseline; the design's Results table and status (both
  languages); `AGENTS.md` "Current plan" pointer.
- [ ] **Step 1:** `mise run test:reference` passes with `neon` active and
  with `RUSTICIDE_KERNELS=portable`.
- [ ] **Step 2:** `RUSTICIDE_KERNELS=portable mise run bench:gate` passes
  against the part-2 baseline (the fallback did not regress).
- [ ] **Step 3:** `mise run bench:gate` with `neon`; record llama.cpp
  `pp128` (`llama-bench -ngl 0 -t 6`) next to ours for both models and
  check Goal 1 (at least 65%). If it is not met, stop and report with the
  profile instead of re-recording.
- [ ] **Step 4:** `mise run bench:baseline`, gate again, `mise run
  ci:quick`.
- [ ] **Step 5: Commit** `perf(bench): re-record the M1 baseline` and
  `docs: record the NEON adapter results`.
- [ ] **Step 6:** After merge (controller, not an implementer): add
  `Miri (rusticide-simd)` and `Gates arm64 (mise run ci:quick)` to the
  `main` ruleset's required checks with `gh api`.
