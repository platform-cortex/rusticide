<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Kernel Facade and Performance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the documents match 0.6.0, add a local performance gate with
a committed baseline, then move every kernel behind one facade in
`rusticide-kernels` with a single `portable` adapter, without losing speed.

**Architecture:** A new `rusticide-bench` crate measures kernels (no model)
and end-to-end tokens/s (with `RUSTICIDE_MODELS`) and compares against
`crates/rusticide-bench/baselines/<machine>.json`. The baseline is captured
from the 0.6.0 code before any refactor. Then `rusticide-kernels` gains an
`Adapter` (a struct of plain function pointers, chosen once per process,
after PyTorch's `DispatchStub`), absorbs `Q8Input`/`int_dot` from
`rusticide-quantize` and `matmul_batch` from `rusticide-model`, and
`BlockFormat` goes back to pure data.

**Tech Stack:** Rust 2024, rayon, serde/serde_json, clap, anyhow (binary
only), mise tasks, GitHub Actions.

**Spec:** `docs/plan/2026-11-01-rusticide-kernel-facade-design.md`

## Global Constraints

- AGENTS.md holds: strict TDD (red, then green), 100% coverage gate, no
  `unwrap`/`expect` outside `#[cfg(test)]`, files under 300 lines, ASCII
  only, the three-line SPDX header on every new `.rs` file, `-D warnings`,
  MSRV 1.96, no `unsafe` anywhere in this plan.
- Commits: `git commit -S`, conventional, first line at most 50 characters,
  no attribution trailers. Run `cargo fmt` before each commit.
- Every new or changed document has a `.pt-BR.md` sibling kept in sync;
  links inside a `.pt-BR.md` point to `.pt-BR.md` siblings when they exist.
- Anti-goal: no decode or prefill figure on the M1 Pro below the committed
  baseline beyond its tolerance. From Task 5 on, every task ends with
  `mise run bench:gate` passing.
- Anti-goal: no indirect call per block. Ports are called per row run or
  per batch.
- Anti-goal: `mise run test:reference` passes at NEAR_TIE 0.25.
- The dispatcher (`rusticide-dispatch`) is not touched.

## Review Focus

1. Gating against a baseline file for another machine, or none: a clear
   error and a non-zero exit, never a pass (Task 2).
2. Running the gate without `RUSTICIDE_MODELS` when the baseline holds
   end-to-end metrics: each missing metric is reported and fails the gate
   (Task 2).
3. A degenerate measurement (no runs, zero elapsed time, zero work): an
   error, never an infinite or NaN throughput (Task 2).
4. A GGUF tensor whose format has no kernel in the adapter: `Matrix::load`
   returns `ModelError::MissingKernel`, not a panic (Task 6).
5. A matrix product with zero tokens or a job with zero rows after the
   move: no work and no panic, as today (Task 7).

---

### Task 1: Make the documents match 0.6.0

**Files:**

- Modify: `docs/plan/2026-10-22-rusticide-run-design.md` (amendment after
  "Non-goals" and in "Memory" and "Quantization formats")
- Modify: `docs/plan/2026-08-21-rusticide-framework-design.md` (decision
  under the dispatcher/`rusticide-simd` sections)
- Create: `docs/plan/2026-10-31-rusticide-inference-performance.pt-BR.md`
- Modify: `docs/plan/2026-11-01-rusticide-kernel-facade-design.pt-BR.md`
  (link the notes' new `.pt-BR.md`)
- Modify: `AGENTS.md` ("Current plan" points to this plan and its design)

**Steps:**

- [ ] **Step 1:** In the run design, add a block in its existing amendment
  style, `> **Amended 2026-11-01.**`, under "Non-goals", "Memory" and
  "Quantization formats": quantized matmul kernels and quantized weights in
  memory shipped in 0.6.0 (cite the notes); the format contract is name,
  type id, block size, block bytes and dequantizer, with kernels registered
  in `rusticide-kernels` (cite the facade design). Do not delete the
  original text.
- [ ] **Step 2:** In the framework design, record the two-level selection
  (dispatcher per operator; `DispatchStub`-style adapter per process in
  `rusticide-kernels`) and that native adapters build on `rusticide-simd`.
- [ ] **Step 3:** Translate the 0.6.0 notes into
  `2026-10-31-rusticide-inference-performance.pt-BR.md` (ASCII, no
  accents, same structure) and point the facade design's `.pt-BR.md` link
  at it.
- [ ] **Step 4:** Run `rumdl check docs AGENTS.md` and
  `LC_ALL=C grep -nP '[^\x00-\x7F]' docs AGENTS.md -r`. Expected: no
  issues, no output.
- [ ] **Step 5:** Commit: `docs: record 0.6.0 decisions in the specs`.

### Task 2: Benchmark results, baselines and comparison

**Files:**

- Create: `crates/rusticide-bench/Cargo.toml` (`publish = false`; deps
  `serde`, `serde_json`, `thiserror`; lints from workspace)
- Create: `crates/rusticide-bench/src/lib.rs`, `src/error.rs`,
  `src/stats.rs`, `src/results.rs`, `src/baseline.rs`, plus `*_tests.rs`
  siblings where a file would pass 300 lines
- The crate is a leaf: no `[workspace.dependencies]` entry. Its version
  follows `version.workspace`, so `release:prepare` bumps it like the rest.

**Interfaces:**

- Produces (all `pub` from `rusticide_bench`):
  - `BenchError` (thiserror): `NoRuns`, `ZeroElapsed`, `ZeroWork`,
    `WrongMachine { expected: String, got: String }`, `Json(serde_json::Error)`.
  - `pub struct Sample { pub median: f64, pub spread: f64 }` (serde);
    `spread` is `(max - min) / median`.
  - `pub fn summarize(throughputs: &[f64]) -> Result<Sample, BenchError>`.
  - `pub fn throughput(work: f64, elapsed: Duration) -> Result<f64, BenchError>`.
  - `pub struct Results { pub machine: String, pub metrics: BTreeMap<String, Sample> }`
    (serde), `Results::to_markdown(&self) -> String` (one table row per
    metric: name, median, spread in percent).
  - `pub const TOLERANCE_FLOOR: f64 = 0.05;`
  - `pub struct Expected { pub median: f64, pub tolerance: f64 }` and
    `pub struct Baseline { pub machine: String, pub metrics: BTreeMap<String, Expected> }`
    (serde); `Baseline::from_results(&Results) -> Baseline` with
    `tolerance = spread.max(TOLERANCE_FLOOR)`.
  - `pub enum Status { Within, Regressed, Missing, New }`,
    `pub struct Row { pub metric: String, pub status: Status, pub baseline: Option<f64>, pub measured: Option<f64> }`,
    `pub fn compare(baseline: &Baseline, results: &Results) -> Result<Vec<Row>, BenchError>`
    (`WrongMachine` when machines differ), `pub fn passed(rows: &[Row]) -> bool`
    (false on any `Regressed` or `Missing`), `pub fn rows_to_markdown(rows: &[Row]) -> String`.

**Steps:**

- [ ] **Step 1: Write the failing tests** (in the module test files):
  - `summarize_takes_the_median_and_relative_spread`:
    `summarize(&[9.0, 10.0, 12.0])` is `Sample { median: 10.0, spread: 0.3 }`.
  - `summarize_of_no_runs_is_an_error`: `NoRuns`.
  - `throughput_rejects_zero_elapsed_and_zero_work`: `ZeroElapsed`, `ZeroWork`.
  - `baseline_tolerance_is_the_spread_with_a_floor`: spreads 0.02 and 0.12
    give tolerances 0.05 and 0.12.
  - `compare_flags_a_drop_beyond_tolerance`: baseline 100.0 / 0.05;
    measured 96.0 is `Within`, 94.0 is `Regressed`.
  - `compare_reports_missing_and_new_metrics`: baseline `{a, b}`, results
    `{a, c}` gives `b` `Missing`, `c` `New`; `passed` is false.
  - `compare_refuses_another_machine`: `WrongMachine`.
  - `results_round_trip_through_json` and `baseline_round_trip_through_json`.
- [ ] **Step 2:** `cargo test -p rusticide-bench`. Expected: FAIL to compile
  (items missing).
- [ ] **Step 3:** Implement the interfaces above. `New` never fails the
  gate.
- [ ] **Step 4:** `cargo test -p rusticide-bench` and
  `cargo clippy -p rusticide-bench --all-targets -- -D warnings`. Expected:
  PASS, no warnings.
- [ ] **Step 5:** Commit: `feat(bench): compare results with a baseline`.

### Task 3: Kernel and end-to-end measurements

**Files:**

- Modify: `crates/rusticide-quantize/src/format.rs` (+ tests): add
  `BlockFormat::synthetic_blocks(&self, blocks: usize, seed: u32) -> Vec<u8>`,
  full-range pseudo-random bytes with every f16 scale field set to a
  finite value (offsets: 0 for Q4_0/Q5_0/Q8_0, 0 and 2 for Q4_K, 208 for
  Q6_K). Replace `random_blocks` in `format_tests.rs` with it.
- Create: `crates/rusticide-bench/src/kernels.rs`, `src/e2e.rs` (new deps:
  `rusticide-quantize`, `rusticide-model`)
- Modify: `crates/rusticide-llama/src/lib.rs`: add
  `pub fn registry() -> Registry` (moved from
  `crates/rusticide/src/cli/registry.rs`, which now calls it; its test
  moves too)

**Interfaces:**

- Consumes: Task 2 `summarize`, `throughput`, `Sample`.
- Produces:
  - `pub struct Plan { pub runs: usize, pub budget: Duration }`; one
    warm-up run is discarded before `runs`.
  - `pub fn kernel_metrics(plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError>`
    with keys `kernel/<format>/row` (one row of 4096 columns against one
    input, elements/s, single thread), `kernel/<format>/rows8` (same rows
    against 8 inputs, element-products/s) for `Q4_0`, `Q5_0`, `Q8_0`,
    `Q4_K`, `Q6_K`, and `kernel/q8/block`, `kernel/q8/superblock`
    (activation quantization, elements/s). Matrix: 256 rows x 4096 columns
    per format. Each run repeats its work until `budget` elapses.
  - `pub const DECODE_PROMPT: &str = "The history of the Roman Empire";`
  - `pub const DECODE_TOKENS: usize = 64;`
  - `pub const PREFILL_PROMPT: &str` -- a fixed English paragraph of
    about 120 words (builder writes it; changing it later invalidates
    baselines).
  - `pub fn e2e_metrics(name: &str, model: &mut dyn Model, decode_prompt: &[u32], prefill_prompt: &[u32], plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError>`
    with keys `e2e/<name>/decode` (greedy argmax for `DECODE_TOKENS` after
    the prompt, tokens/s of the decode steps only) and
    `e2e/<name>/prefill` (one `forward_batch` of the prefill prompt,
    tokens/s). `model.reset()` before each run. Model errors map to a new
    `BenchError::Model(ModelError)`.

**Steps:**

- [ ] **Step 1: Write the failing tests:**
  - `synthetic_blocks_dequantize_to_finite_values` for every format.
  - `kernel_metrics_cover_every_format_and_both_activations`: with
    `Plan { runs: 2, budget: Duration::from_millis(1) }`, the 12 expected
    keys exist and every median is positive and finite.
  - `e2e_metrics_time_decode_and_prefill_on_a_model`: a test `Model`
    (vocab 4, logits favouring token 1) yields both keys, positive medians,
    and was reset once per run, warm-up included.
  - `e2e_metrics_propagate_model_errors`: a model whose `forward` fails
    gives `BenchError::Model`.
  - `registry_lists_llama_and_qwen2` in `rusticide-llama`.
- [ ] **Step 2:** Run the new tests. Expected: FAIL to compile.
- [ ] **Step 3:** Implement. Kernel metrics call today's API
  (`BlockFormat::dot_row_q8`, `dot_rows_q8`, `Q8Input::quantize`) on a
  single thread, no rayon.
- [ ] **Step 4:** `cargo test -p rusticide-bench -p rusticide-quantize -p rusticide-llama -p rusticide`.
  Expected: PASS.
- [ ] **Step 5:** Commit: `feat(bench): measure kernels and tokens/s`.

### Task 4: Bench binary, tasks, CI summary and the first baseline

**Files:**

- Create: `crates/rusticide-bench/src/main.rs` (clap + anyhow; the only
  untested file, excluded by the existing `crates/*/src/main.rs` rule in
  `tarpaulin.toml`; it holds the GGUF and tokenizer loading glue and
  nothing else)
- Modify: `mise.toml` (tasks below, using the task template)
- Modify: `.github/workflows/cicd.yml` (job `bench`)
- Create: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`
- Modify: `AGENTS.md` (commands, the gate rule)

**Interfaces:**

- Consumes: Tasks 2 and 3.
- Produces: binary `rusticide-bench` with subcommands
  - `run --machine <id> [--models <dir>] [--runs N] [--budget-ms N] --out <file>`
    (kernels always; end-to-end for each reference model file found in
    `<dir>`, names `tinyllama` and `qwen2.5`; defaults 5 runs, 200 ms)
  - `report --results <file>` (markdown table to stdout)
  - `gate --results <file> --baseline <file>` (table to stdout, exit 1 on
    failure, exit 2 with a message naming the expected path when the
    baseline file is missing)
  - `baseline --results <file> --out <file>`
- mise tasks (each with alias, description, `usage` with `--verbose`):
  - `bench` (alias `bn`): release build, `run` then `report`.
  - `bench:gate` (alias `bng`): `run` then `gate` against
    `crates/rusticide-bench/baselines/<machine>.json`.
  - `bench:baseline` (alias `bnb`): `run` then `baseline`, writing that
    file.
  - Machine id: lowercase CPU brand with non-alphanumerics as `-`, plus
    `-<logical cpus>` (`sysctl -n machdep.cpu.brand_string` and
    `sysctl -n hw.logicalcpu` on macOS, `/proc/cpuinfo` and `nproc` on
    Linux); the M1 Pro gives `apple-m1-pro-8`. `RUSTICIDE_MODELS` is
    passed as `--models` when set.
- CI job `bench` (kebab-case id, `needs: [version]`, same setup action as
  `gates`): runs `mise run bench` without models and appends the report to
  `$GITHUB_STEP_SUMMARY`. It fails only if the tool crashes.

**Steps:**

- [ ] **Step 1:** Implement `main.rs` and the tasks; `mise run bench`
  locally. Expected: a table with 12 kernel rows and, with
  `RUSTICIDE_MODELS` set, 4 end-to-end rows.
- [ ] **Step 2:** `mise run bench:gate` with no baseline file. Expected:
  exit 2 and the expected path in the message.
- [ ] **Step 3:** On the M1 Pro, plugged in, other heavy apps closed:
  `RUSTICIDE_MODELS=~/models mise run bench:baseline`, then
  `mise run bench:gate` twice. Expected: both pass. Record the end-to-end
  medians in the commit body and check them against the 0.6.0 notes
  (decode about 117 and 82 tokens/s); a difference beyond 10% stops the
  task for investigation.
- [ ] **Step 4:** AGENTS.md: add the three commands to "Commands" and the
  rule "run `mise run bench:gate` before merging any change to kernels,
  scheduling or the forward pass; rewriting a baseline is a reviewed
  change".
- [ ] **Step 5:** `mise run ci:quick`. Expected: PASS at 100%.
- [ ] **Step 6:** Commit: `feat(bench): gate tokens/s on a baseline`.

### Task 5: The adapter and the f32 ports

**Files:**

- Create: `crates/rusticide-kernels/src/adapter.rs` (types and `active`)
- Create: `crates/rusticide-kernels/src/portable/` (`mod.rs`, and the
  current `dot.rs`, `elementwise.rs`, `norm.rs`, `rope.rs` moved here
  unchanged with their tests)
- Modify: `crates/rusticide-kernels/src/lib.rs`

**Interfaces:**

- Produces:
  - `pub(crate) struct F32Kernels` (crate-private: adapters are modules of
    this crate, so native ones later need no public constructor) with one
    `fn` pointer field per current free
    function: `dot`, `add_assign`, `mul_assign`, `silu`, `rms_norm`,
    `softmax`, `rope_adjacent`, `rope_half_split` (same signatures as
    today).
  - `pub struct Adapter { name: &'static str, f32: F32Kernels }` (fields
    private; `pub const fn name(&self) -> &'static str`).
  - `pub fn active() -> &'static Adapter` -- chosen once through a
    `std::sync::OnceLock`; returns `&portable::ADAPTER` (`name` is
    `"portable"`).
  - The existing free functions keep their names and signatures and call
    through `active()`, so no caller changes.

**Steps:**

- [ ] **Step 1: Write the failing tests:** `active_adapter_is_portable`
  (`active().name() == "portable"`, and two calls return the same
  address); `free_functions_route_through_the_adapter` (an in-module test
  builds an `Adapter` whose `dot` returns a constant and checks the
  routing helper used by the free functions with it).
- [ ] **Step 2:** `cargo test -p rusticide-kernels`. Expected: FAIL.
- [ ] **Step 3:** Implement; the free functions take the adapter through
  one private `fn adapter() -> &'static Adapter` so the routing is
  testable.
- [ ] **Step 4:** `cargo test --workspace`, `mise run bench:gate`.
  Expected: PASS.
- [ ] **Step 5:** Commit: `refactor(kernels): route f32 kernels via adapter`.

### Task 6: Quantized kernels move into the facade

**Files:**

- Move: `crates/rusticide-quantize/src/q8_input.rs`, `int_dot.rs`,
  `int_dot/rows.rs` to `crates/rusticide-kernels/src/` (`q8_input.rs`
  shared; `int_dot*` under `portable/`), with their tests and the
  kernel tests from `format_tests.rs`
- Modify: `crates/rusticide-quantize/src/format.rs`, `lib.rs`
- Modify: `crates/rusticide-kernels/Cargo.toml` (deps `half`,
  `rusticide-quantize`), `src/adapter.rs`, `src/lib.rs`
- Modify: `crates/rusticide-model/src/weights.rs`, `batch.rs`,
  `error.rs`; `crates/rusticide-bench/src/kernels.rs`

**Interfaces:**

- Produces:
  - `rusticide_kernels::{Q8_BLOCK, Q8Input, Q8Kind}` (moved; same API).
    `Q8Input::quantize` calls `active()`'s `quantize` slot; the type
    stays in `q8_input.rs`, its portable body moves to
    `portable/q8.rs`.
  - `pub struct QuantKernel { type_id, activation: Q8Kind, dot, dot_rows }`
    with `pub const fn type_id(&self) -> u32`,
    `pub const fn activation(&self) -> Q8Kind`,
    `pub fn dot_row(&self, row: &[u8], input: &Q8Input) -> f32`,
    `pub fn dot_rows(&self, row: &[u8], inputs: &[Q8Input], out: &mut [f32])`
    (bodies of today's `dot_row_q8`/`dot_rows_q8`).
  - `Adapter` gains `quantize: fn(&mut Q8Input, &[f32], Q8Kind)` and
    `quantized: &'static [QuantKernel]`;
    `pub fn quant_kernel(&self, type_id: u32) -> Option<&'static QuantKernel>`.
  - `BlockFormat` drops `dot_q8`, `dot_rows`, `activation` and their
    methods; `rusticide-quantize` no longer exports `Q8*`.
  - `ModelError::MissingKernel { format: &'static str }`, message
    `no kernel for format {format}`.

**Steps:**

- [ ] **Step 1: Write the failing tests:**
  `portable_has_a_kernel_for_every_block_format` (every
  `rusticide_quantize::formats()` entry has a `quant_kernel` with the
  activation the old table had: `Block` for Q4_0/Q5_0/Q8_0, `SuperBlock`
  for Q4_K/Q6_K); the moved kernel tests against the dequantized reference
  using `synthetic_blocks`; `a_format_without_a_kernel_fails_to_load` in
  `rusticide-model` (the private lookup helper given `None` returns
  `MissingKernel { format: "Q4_K" }`).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Move the code; `Matrix` stores `&'static QuantKernel`
  next to the bytes, resolved once in `Matrix::load`.
- [ ] **Step 4:** `cargo test --workspace`, `mise run test:reference`,
  `mise run bench:gate`. Expected: PASS.
- [ ] **Step 5:** Commit: `refactor(kernels): own the quantized kernels`.

### Task 7: The matrix scheduler moves into the facade

**Files:**

- Move: `crates/rusticide-model/src/batch.rs` and `batch_tests.rs` to
  `crates/rusticide-kernels/src/matmul.rs` and `matmul_tests.rs`
- Modify: `crates/rusticide-kernels/Cargo.toml` (dep `rayon`),
  `crates/rusticide-model/src/{weights.rs,lib.rs}`,
  `crates/rusticide-model/Cargo.toml` (drop `rayon`),
  `crates/rusticide-llama/src/{attention.rs,feed_forward.rs,layout.rs}`

**Interfaces:**

- Produces:
  - `pub enum Weights<'a> { Dense { data: &'a [f32], cols: usize }, Blocks { kernel: &'static QuantKernel, data: &'a [u8], row_bytes: usize } }`
    with `pub fn rows(&self) -> usize`.
  - `pub fn matmul_batch(inputs: &[f32], tokens: usize, jobs: &mut [(Weights<'_>, &mut [f32])])`
    and `pub fn matvec_batch(input: &[f32], jobs: &mut [(Weights<'_>, &mut [f32])])`,
    same semantics and output layout (`out[row * tokens + token]`) as
    today; `TASKS_PER_THREAD` stays 16.
  - `Matrix::weights(&self) -> Weights<'_>`; `rusticide-model` stops
    exporting `matmul_batch`/`matvec_batch`.

**Steps:**

- [ ] **Step 1: Write the failing tests:** the moved batch tests rewritten
  against `Weights`, plus `a_job_with_no_rows_does_no_work` and the
  existing zero-token case.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Move; llama builds jobs with `matrix.weights()`.
- [ ] **Step 4:** `cargo test --workspace`, `mise run test:reference`,
  `mise run bench:gate`. Expected: PASS.
- [ ] **Step 5:** Commit: `refactor(kernels): schedule matrix products`.

### Task 8: Close the milestone

**Files:**

- Modify: both design files (status done, measured results), both
  0.6.0 notes files (kernel locations now in `rusticide-kernels`),
  `crates/rusticide-kernels/Cargo.toml` and `README.md` (description:
  the kernel facade), `crates/rusticide-quantize/Cargo.toml` description
  if it mentions kernels, `AGENTS.md` (current plan)

**Steps:**

- [ ] **Step 1:** `mise run ci:quick`, `mise run test:reference`,
  `mise run bench:gate`, `mise run security`. Expected: all PASS, coverage
  100%.
- [ ] **Step 2:** Update the documents with the gate's final table.
- [ ] **Step 3:** `rumdl check` and the ASCII grep from Task 1. Expected:
  clean.
- [ ] **Step 4:** Commit: `docs: close the kernel facade milestone`.
