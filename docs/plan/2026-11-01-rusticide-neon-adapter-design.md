<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# NEON kernel adapter (Design)

Status: done, 2026-11-02. Follows the
[kernel facade](2026-11-01-rusticide-kernel-facade-design.md) (release
0.7.0) and the GGUF hardening patch (release 0.7.1). Branch
`perf/prefill`.

## Intent

Prompt prefill runs at about half of llama.cpp's CPU speed on the M1 Pro:
Qwen2.5 0.5B about 350 tokens/s against 650-735, TinyLlama about 148
against 280-320 (llama.cpp 0.8.0 `llama-bench -ngl 0`, best thread count;
its profile shows no BLAS, the work is CPU kernels). A prefill profile of
TinyLlama puts 62% of the time in the quantized matrix kernels, about 16%
in attention and about 14% in thread waits; the serial per-layer work is
small on its own but leaves the other workers idle.

Three throwaway spikes (2026-11-01) showed the gap cannot be closed in
safe Rust:

- llama.cpp's kernels (`ggml_gemm_q4_K_8x4_q8_K`) interleave several
  weight rows so one `sdot` accumulates a lane per row, with no horizontal
  reduction. Written as scalar safe Rust, that layout compiles to widening
  multiplies: 5.2 G multiply-adds/s against 31.5 for today's shape.
- Tiling rows x tokens in today's layout helps plain `i8` dots by up to
  42%, unreliably, and not at all once the real `Q4_K` block arithmetic is
  in: today's one-row-many-tokens kernel is already the fastest shape.

The owner chose the native adapter the facade was built for: NEON
intrinsics behind safe wrappers in `rusticide-simd`, and a `neon` adapter
in `rusticide-kernels`, compiled in for every target that guarantees NEON
and dotprod.

## Sequencing

The milestone lands in three parts, each behind `bench:gate`:

1. **Per-token parallel work** (safe Rust). The serial per-layer sections
   (Q8 activation quantization, `to_token_major`, RoPE, `rms_norm`,
   copies) cost about 15-17% of prefill with seven workers idle. Each is
   independent per token and runs on the pool.
2. **Sampling on the top-k survivors** (safe Rust). `apply_top_p` sorts
   the whole vocabulary every token (about 15% of Qwen decode with
   sampling on); after top-k, temperature, softmax, top-p and the draw run
   on the k kept (index, logit) pairs, and the logits are not copied.
3. **The `neon` adapter**, the rest of this document.

Parts 1 and 2 come first because they are cheap, and because after part 3
the serial share grows: with kernels 1.5x faster it would be about 22% of
prefill.

## Goals

1. **Matrix kernels at llama.cpp's speed on the M1 Pro**: the `Q4_K`,
   `Q6_K`, `Q5_0` and `Q8_0` prefill kernels reach the per-thread
   throughput of llama.cpp's repacked kernels (about 51 G multiply-adds/s
   per thread, against 33 today). Because the kernels are 62% of prefill,
   this alone moves prefill from about 50% to about 65% of llama.cpp;
   *my call:* 65% of llama.cpp's best `pp128` is the end-to-end floor
   after part 3, parts 1 and 2 add to it, and attention (the next
   milestone) is out of scope.
2. **Decode not slower**, and faster where the same kernels apply.
3. **Every release binary still runs everywhere**: the `neon` adapter is
   compiled only for targets that guarantee NEON and dotprod at build
   time; everywhere else `portable` runs, as today.
4. **`unsafe` stays inside `rusticide-simd`** under the existing policy
   (`docs/UNSAFE.md`): a safe public API, a `// SAFETY:` comment per block,
   a README justification, an entry in `docs/UNSAFE.md`, Miri in CI with
   scalar fallbacks under `cfg(miri)`.

## Non-goals

- AVX2, AVX-512 or any x86 adapter (the Intel server): the next milestone,
  on the same seam.
- SVE, i8mm (`smmla`), or other ARM extensions beyond dotprod.
- Faster attention or exp, and RoPE tables, except where a kernel change
  carries them for free.
- Sharing weights with the file buffer instead of copying them.
- GPU or Metal.

## Anti-goals

- **`unsafe` anywhere but `rusticide-simd`.** `rusticide-kernels` and every
  model crate keep `unsafe_code = "deny"`.
- **A panic or undefined behaviour reachable from a caller's input.** The
  safe API makes out-of-bounds loads unrepresentable (fixed-size array
  references), and the one call into code compiled for a feature rests on
  the build target guaranteeing that feature.
- **Weaker parity.** `mise run test:reference` passes at today's NEAR_TIE
  (0.25) with the `neon` adapter active, and within one adapter a batched
  prefill stays bit-identical to running the tokens one at a time.
- **A slower `portable`.** The gate keeps protecting the fallback path.

## Architecture

- **`rusticide-simd`** (new crate, `unsafe` permitted): the NEON kernels,
  compiled with `#[target_feature(enable = "neon,dotprod")]`, behind safe
  entry points that take fixed-size block arrays and output slices. Its
  only `unsafe` is the call into each feature-compiled kernel (see
  "Findings"). On other targets the module does not exist. Under
  `cfg(miri)` each kernel has a scalar body with the same integer results,
  so its tests run under Miri.
- **The adapter owns the weight layout.** The `neon` adapter repacks each
  quantized matrix once at load into its interleaved layout (several rows
  side by side, as llama.cpp's repacked formats do), so the facade needs
  one new port: preparing a matrix for the active adapter. `portable`'s
  preparation is the identity. Decode uses a matrix-vector kernel on the
  same layout, so there is one copy of the weights, not two. Today
  `Matrix::load` already copies each tensor out of the whole-file buffer
  (`data.to_vec()`); the repack writes into that copy (an interleaved
  group of N rows is exactly N blocks' bytes, as in llama.cpp's
  `block_q4_Kx8`), so it adds no memory.
- **Shapes come from an untrusted file.** A matrix whose row count is not
  a multiple of the interleave group, or whose bytes do not split into
  whole blocks, is left unprepared and runs on `portable`; preparation
  never panics and never reads past a tensor.
- **Formats**: the four the reference models use, in order of time spent:
  `Q4_K` and `Q6_K` (TinyLlama), then `Q5_0` and `Q8_0` (Qwen2.5). A format
  the `neon` adapter has no kernel for falls back to the `portable` kernel
  for that matrix.
- **Selection**: `active()` picks `neon` wherever it is compiled in, else
  `portable`. *My call:* an environment variable
  (`RUSTICIDE_KERNELS=portable`) forces `portable`, so the gate and the
  reference tests can compare adapters on one machine and a user can rule
  the adapter out when triaging. Any other value is reported and ignored,
  never a panic.
- **Equivalence**: the integer sums of `neon` and `portable` are exact and
  must match; float accumulation order may differ, so cross-adapter tests
  compare against the dequantized reference within a stated tolerance,
  and reference parity is the end-to-end check.

## Findings that shape the design

Probes on the pinned toolchain (2026-11-01):

- NEON intrinsics are safe to call inside a function marked
  `#[target_feature(enable = "neon,dotprod")]`, and `bytemuck` (feature
  `aarch64_simd`) converts `[i8; 16]` to `int8x16_t` without `unsafe` in our
  code. So the kernels can be ordinary safe Rust compiled for those
  features, and no pointer-taking load is needed.
- Calling such a function from code without the attribute needs one
  `unsafe` block, even when the build enables the features globally.
  That call is the only `unsafe` the design needs, and it lives in
  `rusticide-simd`, whose safe entry points are the kernels themselves:
  `rusticide-simd` holds the NEON kernels (not just vector wrappers,
  which would not inline across the boundary), and the `neon` adapter in
  `rusticide-kernels` calls them safely.
- *My call:* the soundness argument is a build-time guarantee, not run-time
  detection: the module compiles only under
  `cfg(all(target_arch = "aarch64", target_feature = "dotprod"))`, which
  `aarch64-apple-darwin` satisfies by default and the Linux aarch64 targets
  do not. Linux aarch64 release binaries keep `portable` this milestone;
  run-time detection there is a later, separate decision.
- `vdotq_s32` and friends are stable only from Rust 1.98. The owner set
  the rule that the MSRV is the higher of the rust-skills target (1.96) and
  the lowest version a shipped feature needs, so this milestone raises the
  MSRV to 1.98 (recorded in AGENTS.md). No compiler-version probe or
  `build.rs`: every supported compiler builds the `neon` adapter.
- Miri cannot execute these intrinsics, and the `unsafe` block is a call,
  not a memory access. *My call:* the Miri job interprets `rusticide-simd`'s
  tests for `aarch64-apple-darwin` (so the module is compiled in) with the
  `cfg(miri)` scalar bodies, on a dated nightly pinned like the coverage
  task, and becomes a gate in `cicd.yml`, as `docs/UNSAFE.md` requires of
  the first `unsafe` change (today Miri runs only in `nightly.yml`).
- CI gains an `arm` job on GitHub's Linux arm64 runner
  (`ubuntu-24.04-arm`, free for this public repository; Neoverse-class
  cores with dotprod) that runs `mise run ci:quick`, coverage gate
  included, with `RUSTFLAGS="-D warnings -C target-feature=+dotprod"` so
  the `neon` module is compiled in. `RUSTFLAGS` replaces
  `.cargo/config.toml`'s `-D warnings` rather than merging, so the job
  repeats it. The flag lives only in that test job, never in committed
  build or release config. The x86_64 `gates` job keeps covering
  `portable`; the M1's `mise run ci:quick` covers both locally. The new
  job joins the `main` ruleset's required checks.
- The M1 Pro is 6 performance plus 2 efficiency cores, and the pool runs
  8 threads; llama.cpp is fastest at 6. *My call:* the thread count is
  measured again after part 1, not changed blindly.

## The gate

`bench:gate` measures the active adapter. Each part is expected to improve
the e2e rows beyond tolerance; after each, the M1 baseline is re-recorded
(a reviewed change) so the gain is locked in. The kernel metrics keep
their keys and measure the active adapter's kernels.
`RUSTICIDE_KERNELS=portable mise run bench:gate` must still pass against
the baseline recorded after part 2, proving the fallback did not regress.

## Constraints

Everything in AGENTS.md holds (strict TDD, 100% coverage, files under 300
lines, ASCII, SPDX headers, `-D warnings`, MSRV 1.98 per the rule
above). Coverage and Miri: tarpaulin measures the `neon` build on the M1
and on the `arm` CI job; Miri runs the `cfg(miri)` bodies in CI; x86_64
runners compile the adapter out and cover `portable`.

## Left to the builder

- The exact interleave (rows per group, bytes per lane), tile shape and
  activation layout, following llama.cpp's repacked kernels where they
  fit.
- The set of wrapper operations in `rusticide-simd` and their names.
- The preparation port's signature and where packed data lives.
- The cross-adapter tolerance values.
- Whether `Q5_0`/`Q8_0` reach the same structure or a simpler tiled kernel,
  as long as Qwen2.5 meets goal 1.
- How part 1 splits the serial work across the pool (per token, fused
  norm and quantization, or folded into the next parallel region).

## Results

Part 1, Apple M1 Pro (8 cores), tokens/s, `PARALLEL_MIN_TOKENS` = 2
(unchanged). Before is the previous baseline; after is the re-recorded
baseline (median of three sessions).

- Qwen2.5 prefill: 348.48 before, 426.03 after (+22%).
- Qwen2.5 decode: 131.01 before, 132.60 after (within tolerance).
- TinyLlama prefill: 148.00 before, 166.82 after (+13%).
- TinyLlama decode: 96.85 before, 96.36 after (within tolerance).
- Thread count numbers come from separate `mise run bench` runs and are not
  comparable to the baseline rows above.
- Thread count, `RAYON_NUM_THREADS=6` (best): Qwen2.5 prefill 405.53,
  decode 117.98; TinyLlama prefill 154.23, decode 80.04.
- Thread count, `RAYON_NUM_THREADS=8` (best): Qwen2.5 prefill 422.72,
  decode 117.10; TinyLlama prefill 159.25, decode 89.63.
- Decision: keep the default (8 threads); 8 threads is faster or equal on
  every row except Qwen2.5 decode (117.98 vs 117.10, under 1%).

Part 2, Apple M1 Pro (8 cores), tokens/s. Sampling decode is `rusticide run`
on Qwen2.5 with `--temperature 0.8 --seed 1 --max-tokens 256`, three runs
alternating before (commit 6e56270) and after.

- Qwen2.5 sampling decode: 100.08, 105.03, 105.56 before; 117.25, 117.51,
  117.44 after (about +12%).
- Baseline re-recorded: Qwen2.5 prefill 426.03 -> 424.68, decode 132.60 ->
  130.38; TinyLlama prefill 166.82 -> 167.55, decode 96.36 -> 96.25. The
  greedy bench does not exercise sampling, so these are noise-level moves.

Part 3, Apple M1 Pro (8 cores), NEON adapter (dotprod, selected at build
time on aarch64-apple-darwin). Before is the part-2 baseline (portable
kernels); after is the re-recorded baseline. Kernel metrics are G/s: weights
(multiply-adds) per second on one thread, as printed by the gate.

- Q4_K kernel: row 29.43 -> 35.10, rows8 29.94 -> 90.25.
- Q6_K kernel: row 25.94 -> 28.43, rows8 26.81 -> 79.99.
- Q5_0 kernel: row 20.28 -> 29.24, rows8 24.37 -> 81.48.
- Q8_0 kernel: row 35.04 -> 59.53, rows8 35.43 -> 97.18.
- Qwen2.5 prefill: 424.68 before, 1016.21 after (+139%); decode 130.38
  before, 146.21 after (+12%).
- TinyLlama prefill: 167.55 before, 468.82 after (+180%); decode 96.25
  before, 102.82 after (+7%).
- Portable fallback (`RUSTICIDE_KERNELS=portable`) passes the gate against
  the part-2 baseline: no regression.
- llama.cpp (`llama-bench -ngl 0 -t 6,8 -p 128,512 -n 0 -r 5`, build
  8642541b2, BLAS+Metal backends). The reference is `-t 6`: `-t 8` is much
  slower and noisier on this 6P+2E M1 Pro because two threads land on
  efficiency cores. TinyLlama pp128 318.20 +- 4.62 (t8 273.26), pp512
  281.36 (t6). Qwen2.5-0.5B pp128 599.26 +- 25.91 (t8 340.41), pp512 722.82
  +- 17.44 (t8 475.82). An earlier single-run measurement (286.71 and
  640.46) was noisy and is superseded.
- Our bench prefill prompt is a fixed paragraph of roughly 150 tokens,
  between pp128 and pp512, so both are given. TinyLlama: 468.82 / 318.20 =
  147% of pp128 (pp512 is below pp128, so 147% is the conservative figure).
  Qwen2.5: 1016.21 / 599.26 = 170% of pp128, 1016.21 / 722.82 = 141% of
  pp512. Headline (conservative): 141% for Qwen2.5 and 147% for TinyLlama.
  Goal 1 (at least 65%) is met.
- Follow-up: an N=1 decode specialization for Q4_K.
- Follow-up: Linux aarch64 release builds still run portable (run-time
  dotprod detection is a later decision).
- Follow-up: attention and RoPE tables are the next milestone.
