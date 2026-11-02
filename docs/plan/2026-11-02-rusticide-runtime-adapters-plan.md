<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Runtime-selected kernel adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** one release binary per target that picks, at run time, the
fastest kernel adapter its CPU can run: `neon` on aarch64 with `dotprod`,
`avx2` on x86_64 with AVX2, else `portable`.

**Architecture:** `rusticide-simd` gains zero-sized proof tokens
(`Dotprod`, `Avx2`) that only detection constructs; every kernel entry
point takes one, and its single `unsafe` call cites it. `rusticide-kernels`
detects once, builds the ordered list of runnable adapters, and resolves
`RUSTICIDE_KERNELS` against it. The AVX2 adapter mirrors NEON's structure:
a row-group layout prepared at load and a group kernel per format.

**Tech Stack:** Rust 2024, `std::arch` (aarch64 NEON, x86_64 AVX2/AVX-512),
`is_*_feature_detected!`, Miri, cargo-tarpaulin, `rusticide-bench`,
llama.cpp `llama-bench` as reference.

**Spec:** `docs/plan/2026-11-02-rusticide-runtime-adapters-design.md`
(read it; its Findings section carries facts every task relies on).

## Global Constraints

- MSRV 1.98, toolchain pin unchanged; no API newer than 1.98.
- `unsafe` only in `rusticide-simd`: one block per kernel entry point, a
  `// SAFETY:` comment naming the token; safe public API.
- AVX2 (and AVX-512) kernels are bit-identical to portable for every input
  the safe API accepts (`to_bits` equality in tests); no FMA where portable
  multiplies and adds separately.
- Never `target-cpu=native`, never a per-machine build in committed config.
- Detection cost: at most a cached load per kernel call, none per block.
- No `unwrap`/`expect` in production code; 100% coverage; Miri passes.
- ASCII only, files under 300 lines, SPDX headers, bilingual docs,
  conventional signed commits (`git commit -S`), no attribution trailers.
- Blocking floors on the iMac Pro: each AVX2 kernel at least 2x its
  portable kernel (rows8, one thread); prefill at least 65% of llama.cpp's
  CPU backend on both reference models. Two optimisation rounds after a
  miss, then stop and report to the owner.
- The iMac Pro is `ssh elio@192.168.18.146` (passwordless sudo; tools need
  a login shell: `zsh -lc '...'`). Its clone lives at
  `~/Development/github.com/platform-cortex/rusticide`; Task 4 adds a
  worktree of it at `~/rusticide-wt` (any branch; only its files matter).
  Sync the M1 worktree into it with `rsync -a --delete --exclude target
  --exclude .git --exclude mise.local.toml ./
  elio@192.168.18.146:~/rusticide-wt/` and build there. Never commit or
  push from the iMac Pro; measurements there are of the synced tree.
- `bench:gate` on the M1 after every task that touches kernels or
  selection: no regression.

## Review Focus

1. `RUSTICIDE_KERNELS=avx2` (or `neon`) on a CPU that lacks the feature:
   a one-line warning and the best runnable adapter; the SIMD kernel is
   never reached. Tested in Task 3 through the injected adapter list.
2. A `Q8` activation holding `-128` built through the public API: the
   views refuse it and the result still equals portable's. Task 1.
3. A matrix whose row count is not a multiple of the group (or zero-byte
   rows) under the AVX2 layout: the remainder stays row-major and the
   portable kernel reads it, as NEON does. Task 5.
4. A kernel wrapper reached before selection stored its token: it returns
   the portable result, never an `unsafe` call. Task 3 (NEON) and Task 4
   (AVX2), from the in-module tests.
5. Activations and weights at odd addresses: unaligned loads only, the
   odd-address checks NEON's `test_support` already has. Tasks 5-7.

---

## Part 1: run-time selection

### Task 1: arch-neutral group views that refuse `-128`

**Files:**

- Create: `crates/rusticide-simd/src/group.rs` (moved from
  `neon/group.rs`), tests beside it.
- Modify: `crates/rusticide-simd/src/lib.rs`, every `neon/*` user of
  `neon::group`, `crates/rusticide-kernels/src/neon/group_dot.rs`.

**Interfaces:**

- Produces `rusticide_simd::group::{ROWS, TOKENS, GroupKernel,
  Q8Blocks, Q8Superblocks}`, compiled on every target.
- `Q8Blocks::new(scales, quants) -> Option<Self>` and
  `Q8Superblocks::new(scales, quants, sums) -> Option<Self>`: `None` when
  any covered quant is `-128`; otherwise as today (the shortest slice
  bounds the view).
- `group_dot::dot_group` computes a token whose view is `None` with the
  format's portable `dot` for each row of the group (same values).
- [ ] **Step 1: Write the failing tests:**
  `blocks_refuse_minus_128`, `superblocks_refuse_minus_128`,
  `views_accept_minus_127` (simd); in kernels
  `a_minus_128_activation_matches_portable` (a hand-built `Q8Input` with
  one `-128`, every NEON format, `to_bits` equal to portable).
- [ ] **Step 2: Run** `cargo test -p rusticide-simd -p rusticide-kernels`;
  see them fail (compile errors count).
- [ ] **Step 3: Move and implement.** Keep the module docs.
- [ ] **Step 4: Verify** `cargo test --workspace`, `mise run lint`,
  Miri (`mise run` task the `miri` CI job runs, or
  `cargo +nightly-2026-08-03 miri test -p rusticide-simd`).
- [ ] **Step 5: Commit** `refactor(simd): share the group views`.

### Task 2: the `Dotprod` token and a per-architecture `neon` module

**Files:**

- Create: `crates/rusticide-simd/src/token.rs` (tokens and detection).
- Modify: `crates/rusticide-simd/src/lib.rs`, every `neon/*.rs` entry
  point, `crates/rusticide-simd/README.md`, `docs/UNSAFE.md`,
  `mise.toml` task `test`, the `rusticide-kernels` call sites.

**Interfaces:**

- `#[derive(Clone, Copy, Debug)] pub struct Dotprod(())` on
  `target_arch = "aarch64"`; `Dotprod::detect() -> Option<Dotprod>` uses
  `is_aarch64_feature_detected!("dotprod")` (NEON is baseline there);
  under `cfg(miri)` it returns `Some` (the stand-ins need no hardware).
- `rusticide_simd::neon` compiles on every `aarch64` target; each public
  kernel takes `token: Dotprod` first, e.g.
  `q4_k_group(token: Dotprod, group, inputs, out)`. `AVAILABLE` is
  removed.
- The `test` task excludes `crates/*/src/neon/*` when `rustc --print cfg`
  (with `RUSTFLAGS`) lacks `target_arch="aarch64"`, and
  `crates/*/src/avx2/*` when it lacks `target_arch="x86_64"` (replaces the
  `dotprod` check). Detection's `None` arm is covered through
  `Dotprod::from_detected(bool) -> Option<Dotprod>` (private, called by
  `detect`, tested both ways).
- `docs/UNSAFE.md` `rusticide-simd` entry: the soundness argument becomes
  "a token exists only after detection proved the feature; every entry
  point takes one".
- [ ] **Step 1: Failing tests:** `from_detected_true_yields_a_token`,
  `from_detected_false_yields_none`, and `detect_matches_std` (equals
  `is_aarch64_feature_detected!` off Miri).
- [ ] **Step 2: Run**, see them fail.
- [ ] **Step 3: Implement**; update each `// SAFETY:` comment to cite the
  token.
- [ ] **Step 4: Verify** workspace tests, lint, Miri, and on the M1
  `mise run test` (100%); also `cargo check --target
  x86_64-apple-darwin --workspace` (install the target with rustup).
- [ ] **Step 5: Commit** `feat(simd): gate NEON on a detected token`.

### Task 3: run-time adapter selection

**Files:**

- Modify: `crates/rusticide-kernels/src/adapter.rs`, `adapter_tests.rs`,
  `neon/mod.rs` and the `neon/*_dot.rs` wrappers, `lib.rs`;
  `crates/rusticide/src/cli/cmd/version.rs`; `rusticide-bench` results;
  `.github/workflows/cicd.yml` (`arm` job); `AGENTS.md` CI section.

**Interfaces:**

- `fn available() -> Vec<&'static Adapter>`: best first, `portable` last;
  on aarch64 `neon` when `Dotprod::detect()` is `Some`. It stores the
  token in `neon`'s `static TOKEN: OnceLock<Dotprod>` before returning.
- `fn choose(requested: Option<&str>, available: &[&'static Adapter]) ->
  (&'static Adapter, Option<String>)`: `None` -> `available[0]`; a name in
  the list -> it; `"portable"` -> portable; a known name not in the list
  -> `available[0]` plus `"RUSTICIDE_KERNELS=<value>: not supported by
  this CPU, using <name>"`; unknown -> `available[0]` plus
  `"RUSTICIDE_KERNELS=<value>: unknown adapter, using <name>"`. Known
  names: `portable`, `neon`, `avx2`.
- NEON wrappers read `TOKEN`; when it is unset they compute with the
  portable kernel (Review Focus 4).
- `rusticide version` prints a second line `kernels: <name>`.
- Bench results record `"adapter": "<name>"`; the gate compares only
  results of the same adapter (a baseline from another adapter is
  "missing", exit 2).
- `arm` job: `RUSTFLAGS` loses `-C target-feature=+dotprod`; a step after
  `ci:quick` runs `cargo run --release -p rusticide --features cli --
  version` and fails unless a line is exactly `kernels: neon`.
- [ ] **Step 1: Failing tests** for every `choose` row (exact texts),
  `available_ends_with_portable`, `version_names_the_kernels`, the bench
  adapter field round-trip and the cross-adapter gate exit, and
  `wrapper_without_token_matches_portable`.
- [ ] **Step 2: Run**, see them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** workspace tests, `mise run test`, lint,
  `mise run test:reference` on the M1 (and with
  `RUSTICIDE_KERNELS=portable`), `mise run bench:gate` on the M1 (no
  regression; re-record the M1 baseline only if the adapter field forces
  it, as its own commit `perf(bench): re-record the M1 baseline`).
- [ ] **Step 5: Commit** `feat(kernels): select the adapter at run time`.

## Part 2: the AVX2 adapter

### Task 4: the `Avx2` token, an empty `avx2` adapter, the iMac Pro baseline

**Files:**

- Modify: `crates/rusticide-simd/src/token.rs`, `lib.rs` (new `avx2`
  module, x86_64 only); create `crates/rusticide-kernels/src/avx2/mod.rs`;
  `adapter.rs`; `.github/workflows/cicd.yml` (`miri` job: a second step
  for `x86_64-unknown-linux-gnu`); create
  `crates/rusticide-bench/baselines/<imac machine>.json`.

**Interfaces:**

- `Avx2` mirrors `Dotprod` (`detect`, `from_detected`), detecting `avx2`
  (add `fma` only if a kernel uses it -- see Global Constraints).
- `avx2::ADAPTER` named `"avx2"`, `..portable::ADAPTER` (no kernels yet),
  ranked first on x86_64 when detected; `static TOKEN: OnceLock<Avx2>` as
  for NEON.
- iMac Pro setup: the `~/rusticide-wt` worktree (Global Constraints),
  `mise trust` and `mise install` there; copy
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` from the M1's `~/models` with `scp`;
  run with `RUSTICIDE_MODELS=~/models`.
- [ ] **Step 1: Failing tests** mirroring Task 2's token tests and Task
  3's `available` test for x86_64 (run on the iMac Pro over ssh after
  rsync).
- [ ] **Step 2: Run**, see them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** on the iMac Pro: `cargo test --workspace`,
  `rusticide version` says `kernels: avx2`, `mise run bench:baseline`
  (writes the iMac baseline: the portable kernels under the `avx2` name,
  the reference for the 2x floors). Measure llama.cpp there, CPU backend
  only, threads 10 and 20, `-r 5`, pp128/pp512/tg64 on both models; also
  with BLAS. Record both command lines and numbers in the ledger (they
  go into Results in Task 11). On the M1: workspace tests and bench:gate.
- [ ] **Step 5: Commit** `feat(kernels): add the avx2 adapter shell` and,
  separately, `perf(bench): record the iMac Pro baseline`.

### Task 5: `Q4_K` on AVX2

**Files:**

- Create: `crates/rusticide-simd/src/avx2/{q4_k.rs, q4_k/vector.rs,
  q4_k/scalar.rs, q4_k/tests.rs}` and shared helpers as needed;
  `crates/rusticide-kernels/src/avx2/{q4_k.rs, q4_k_dot.rs, tests}`.

**Interfaces:**

- `rusticide_simd::avx2::q4_k::{UNIT, q4_k_group(token: Avx2, group:
  &[[u8; UNIT]], inputs: &[Q8Superblocks], out: &mut [[f32; ROWS]])}`;
  the unit's byte order is AVX2's own and documented in the module.
- `avx2::q4_k_dot::KERNEL: QuantKernel` with `layout: Some(Layout {
  group_rows: ROWS, prepare, read_row, dot_group })`, registered in
  `avx2::ADAPTER`. Generic layout and dot helpers move from `neon/` to a
  shared kernels module when AVX2 needs them too (no copies).
- [ ] **Step 1: Failing tests:** bit-identity to portable on random,
  all-extreme and odd-address inputs, partial groups and zero-byte rows
  (Review Focus 3, 5), `read_row` round-trip after `prepare`; Miri runs
  the scalar stand-ins.
- [ ] **Step 2: Run** on the iMac Pro, see them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** on the iMac Pro: tests, `mise run test` (100%:
  the `avx2` files are measured only on x86_64), `test:reference`
  (TinyLlama and Qwen use `Q4_K`), `mise run bench` kernel rows: `Q4_K` rows8 at
  least 2x the baseline's portable number (blocking). On the M1: tests,
  lint, Miri, bench:gate.
- [ ] **Step 5: Commit** `feat(simd): run Q4_K on AVX2`.

### Task 6: `Q6_K` on AVX2

Same shape as Task 5 for `Q6_K` (`q6_k_group`, `avx2::q6_k_dot::KERNEL`),
reusing Task 5's helpers. Floor: `Q6_K` rows8 at least 2x portable.
Commit `feat(simd): run Q6_K on AVX2`.

### Task 7: `Q5_0` and `Q8_0` on AVX2

Same shape for both, on `Q8Blocks`. `Q8_0` uses the sign trick with `|w|`
unsigned (exact because the views refuse `-128`, Task 1); its tests
include all-`-128` weight blocks against `+-127` activations. Floors: both
at least 2x portable. Commit `feat(simd): run Q5_0 and Q8_0 on AVX2`.

### Task 8: the AVX2 gate

- [ ] **Step 1:** On the iMac Pro, quiet, AC: `mise run bench` prefill and
  decode for both models (threads as the bench picks), against the
  llama.cpp CPU-backend numbers from Task 4.
- [ ] **Step 2:** If prefill is at least 65% on both models and every
  kernel floor holds, record and go to Task 9. Otherwise run an
  optimisation round (profile on the iMac Pro with `cargo flamegraph` or
  `sample`, fix the largest cost, re-measure); at most two rounds. Still
  short: stop, write the numbers and the profile summary to the ledger,
  and report to the owner. Do not start Task 9.
- [ ] **Step 3:** `test:reference` on the iMac Pro with the `avx2`
  adapter and with `RUSTICIDE_KERNELS=portable`.
- [ ] **Step 4: Commit** any optimisation as `perf(simd): <what>`.

## Part 3: AVX-512, conditional

### Task 9: the AVX-512 spike

Throwaway, in `.worktrees/avx512-spike` on a branch off
`feat/runtime-adapters`; never merged. Port the `Q4_K` group kernel to
512-bit (`avx512bw`, `avx512vl`) and measure on the iMac Pro: kernel rows8
and prefill on both models against Task 8's AVX2 numbers. Also decide how
CI could run AVX-512 tests on every run (a runner guaranteeing AVX-512,
or an emulator such as Intel SDE; note its licence and supply-chain cost).
Ledger the numbers and a ruling: build it (prefill gain at least 10% on
both models AND a CI answer; or AVX2 missed a floor and this closes it,
CI answer still required) or record it as a follow-up. Remove the spike
worktree and branch after ledgering.

### Task 10: the AVX-512 adapter (only if Task 9 ruled "build")

An `avx512` adapter ranked above `avx2`, with its own token
(`avx512f`, `avx512bw`, `avx512vl`), kernels for the formats the spike
showed gain on (the rest from `avx2`), the CI step Task 9 chose, and the
same tests and floors as Tasks 5-7. Skipped otherwise; the ledger says
so.

## Finish

### Task 11: baselines, results, docs

- [ ] **Step 1:** Re-record both baselines (`bench:baseline` on the M1 and
  on the iMac Pro; quiet, AC), each its own commit
  `perf(bench): re-record the <machine> baseline`.
- [ ] **Step 2:** `test:reference` on both machines with the default
  adapter and with `RUSTICIDE_KERNELS=portable`.
- [ ] **Step 3:** Fill the design's Results (EN and pt-BR): kernel rows
  portable vs SIMD per format and machine, prefill/decode per model and
  machine, llama.cpp CPU-backend and BLAS numbers with their command
  lines, the AVX-512 ruling, follow-ups. Set Status to done.
- [ ] **Step 4:** `AGENTS.md`: CI section (`arm` without `+dotprod`, the
  version assertion, Miri on two targets), the `unsafe` line if it
  changed, and the "Current plan" pointer back to the framework spec's
  next step. Keep `docs/UNSAFE.md` in sync.
- [ ] **Step 5:** `mise run ci:quick`, `mise run security`,
  `mise run markdownlint` clean.
- [ ] **Step 6: Commit** `docs: record the runtime adapters results`.
