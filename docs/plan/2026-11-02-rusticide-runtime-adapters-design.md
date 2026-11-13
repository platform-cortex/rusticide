<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Runtime-selected kernel adapters (Design)

Status: done, 2026-11-12. Follows the
[NEON adapter](2026-11-01-rusticide-neon-adapter-design.md) (release
0.8.0). Branch `feat/runtime-adapters`.

## Intent

Release 0.8.0 chooses its kernel adapter when it is compiled. That is
right for `aarch64-apple-darwin`, whose target guarantees `dotprod`, and
wrong everywhere else:

- The Linux aarch64 release binaries build without `+dotprod`, so they
  run the portable kernels, about 3x slower than NEON, on the Graviton
  and Ampere servers that all have it.
- No x86_64 binary has SIMD kernels at all. The owner's server, an iMac
  Pro with a Xeon W-2150B (Skylake-SP, 10 cores / 20 threads, AVX2 and
  AVX-512 F/BW/VL/DQ, no VNNI), runs the portable kernels.

A release binary must run on every CPU of its target (AGENTS.md: never
`target-cpu=native`). So the binary itself has to find out, once per
process, what the CPU offers, and pick the fastest adapter it can run.

The owner chose (2026-11-02): runtime detection for every SIMD adapter,
the `unsafe` soundness argument moving from "the build target guarantees
the feature" to "detection proved the feature before the kernel could be
reached"; an AVX2 adapter for x86_64; AVX-512 only if it earns its place
by measurement (see [Sequencing](#sequencing)).

## Sequencing

1. **Runtime selection.** NEON moves from build-time to run-time
   selection. Nothing gets faster on the M1; the Linux aarch64 release
   binary gains NEON. Proven by the `arm` CI job dropping its
   `+dotprod` flag and still selecting `neon`.
2. **AVX2 adapter.** Kernels for `Q4_K`, `Q6_K`, `Q5_0` and `Q8_0`, one
   format per task, as NEON did; measured on the iMac Pro.
3. **AVX-512, conditional.** A spike first. It becomes an adapter only if
   both hold:
   - prefill on the W-2150B beats AVX2 by at least 10% (twice the gate
     tolerance) on both reference models;
   - CI can run its tests on every run, not only when the runner happens
     to have AVX-512.

   Otherwise the spike's numbers go into [Results](#results) and AVX-512
   becomes a recorded follow-up. The exception: if AVX2 misses a floor in
   [Goals](#goals) and AVX-512 is what closes the gap, it is no longer
   optional (both CI conditions still apply).
4. **Release 0.9.0** after the merge, as for 0.8.0.

## Goals

- One binary per target selects, on first kernel use, the best adapter
  the CPU supports: `neon` on aarch64 with `dotprod`, `avx2` on x86_64
  with AVX2 (and FMA only if a kernel uses it; see Constraints), else
  `portable`.
- `RUSTICIDE_KERNELS=portable|neon|avx2` still forces a choice. Asking
  for an adapter the CPU cannot run warns and falls back to the best one
  it can, never runs it.
- The chosen adapter is observable: `rusticide` reports it (the builder
  picks where: `--version` output, a `-v` line, or `rusticide info`), and
  the bench results record it.
- AVX2 kernels are bit-identical to the portable ones, as NEON's are,
  for every input, including byte values a well-formed file never holds.
- On the iMac Pro, each AVX2 kernel is at least 2x its portable kernel
  (rows8, one thread), and prefill reaches at least 65% of llama.cpp
  CPU (`llama-bench`, already installed there) on both reference models.
  The 65% floor is the NEON milestone's; matching or beating llama.cpp is
  the aim, not the gate. The reference is llama.cpp's CPU backend alone:
  the Homebrew build there also loads a BLAS backend, which takes over
  batched products in prefill; that number is recorded beside it, not
  compared against. The builder pins how BLAS is kept out and records the
  `llama-bench` command line.
- **Both floors block** (owner, 2026-11-02). A miss means more
  optimisation inside the milestone, AVX-512 included if that is what
  closes it. After two optimisation rounds that still miss, the work
  stops and the numbers go to the owner before anything else is
  decided.

## Non-goals

- `f32` kernels (dot, softmax, RoPE, norms): attention and RoPE are
  Milestone B. Every adapter keeps the portable `f32` set.
- `Q4_0` SIMD kernels (no reference model uses it).
- AVX-512 VNNI, AMX, SVE: the owner's hardware has none of them.
- Run-time detection on any OS other than Linux, macOS and Windows that
  std supports; std's detection macros decide this, not us.
- New targets in `publish.yml`.

## Anti-goals

- A binary that executes an instruction its CPU lacks: SIGILL is a crash
  for the user, and undefined behaviour in Rust's terms. This is the one
  outcome that makes the milestone a failure even if every number is
  good.
- Different results on different machines. Bit-identity to portable is
  what keeps greedy parity, seeded sampling and tests the same everywhere.
- Per-call detection cost in the hot loop.
- `target-cpu=native` or a per-machine build anywhere in committed config.

## Architecture

**Proof tokens (`rusticide-simd`).** Each instruction set gets a
zero-sized `Copy` type, constructible only by detection:
`Dotprod::detect() -> Option<Dotprod>` (aarch64), `Avx2::detect() ->
Option<Avx2>` (x86_64). Every public kernel entry point takes the token
by value as its first argument. Holding a token is the proof; the single
`unsafe` call into the `#[target_feature]` body cites it. The public API
stays safe, and no safe caller can reach a kernel without detection
having succeeded. Detection uses std's `is_aarch64_feature_detected!` /
`is_x86_feature_detected!`, which cache their answer.

The modules now compile per architecture (`neon` on every `aarch64`,
`avx2` on every `x86_64`), no longer per target feature.

**Selection (`rusticide-kernels`).** `compiled_default()` becomes a
detection step that yields the ordered list of adapters this CPU can run
(best first, `portable` last). `choose(requested, available)` keeps its
pure, injectable shape: the tests pass any list, so every branch is
reachable on any machine. `active()` still resolves once through the
`OnceLock`.

**How the adapter reaches the token.** Each SIMD adapter's `QuantKernel`
functions are plain `fn` pointers, so they cannot capture a token. The
builder chooses how they get one, within these rules: no detection per
kernel call beyond a cached load, and no path by which a kernel runs
before `Adapter` selection proved the feature. One acceptable shape: the
token is stored in a `OnceLock` in the adapter module, set by the
detection step, and the wrappers read it (a missing token is a bug path
that returns the portable result, never an `unsafe` call).

**AVX2 kernels.** Same structure as NEON: a row-group layout prepared
once at load, a group kernel per format, one-token decode paths where the
NEON milestone found them worth it. The layout is AVX2's own (the NEON
one is shaped for `sdot` lanes); `Layout`/`prepare_matrix` already allow
that.

## Findings that shape the design

- **`vpmaddubsw` saturates.** It multiplies unsigned by signed bytes and
  adds pairs into a saturating `i16`. For `Q4_K` (0..15), `Q5_0` and
  `Q6_K` (offset-coded, at most 63) against `Q8` activations the pair sum
  stays inside `i16`. For `Q8_0` the usual sign trick (`|w|` unsigned,
  `a` carrying `w`'s sign) is exact while the activation is never `-128`:
  `2 x 128 x 127 = 32512`. Weights come from the file and may hold `-128`,
  which this order tolerates; the activations come from our quantizer,
  which maps the largest magnitude to `+-127`. My call (2026-11-02): the
  invariant holds at `rusticide-simd`'s public API, the crate with the
  `unsafe`, not only at `rusticide-kernels`. Its activation views
  (`Q8Blocks::new` and its AVX2 counterpart) refuse quants holding
  `-128`, one scan per activation matrix, so every value the safe API
  accepts gets an exact result and the `Q8_0` kernel needs no per-block
  branch. Bit-identity is required for every accepted input. Each kernel's tests include all-`-128` weight blocks and
  extreme activations.
- **AVX-512 on Skylake-SP** has no VNNI: the int8 dot is the same
  `maddubs` + `madd` pair at twice the width, and heavy 512-bit work
  lowers the clock of every core. Its gain over AVX2 is unknown, which is
  why it is conditional.
- **Miri.** NEON's pattern holds: `cfg(miri)` scalar stand-ins with the
  same names, the Miri job interpreting the crate for each architecture
  (add an x86_64 target). Under Miri, detection must yield a token (the
  stand-ins need no hardware); the builder decides how.
- **Coverage.** Each module now compiles on every runner of its
  architecture, so the `test` task's exclusion moves from "no `dotprod`"
  to "not this architecture": `neon/` off aarch64, `avx2/` off x86_64.
  GitHub's x86_64 runners all have AVX2, so the x86 `Gates` job measures
  the AVX2 kernels at 100%. Detection's "absent" branch is reached through
  the injectable `choose`.
- **The iMac Pro** (`ssh elio@192.168.18.146`, passwordless sudo) has
  rustup, mise and `llama-bench`, and only the TinyLlama model; Qwen2.5
  must be copied there. Its baseline is a new file
  `crates/rusticide-bench/baselines/<machine>.json` from
  `mise run bench:baseline` on that machine.

## The gate

The floors in [Goals](#goals) block, as stated there.

Per task, as before: tests and 100% coverage, Miri, `bench:gate` on the
M1 (no regression from the selection change), and on the iMac Pro for
the AVX2 tasks once its baseline exists. The final task re-records both
baselines, runs `test:reference` on both machines (NEON on the M1, AVX2
on the iMac Pro, and `RUSTICIDE_KERNELS=portable` on each), and records
the llama.cpp ratios.

## Constraints

- MSRV 1.98, toolchain pin unchanged. The AVX2 and AVX-512 intrinsics and
  `#[target_feature]` on safe functions are stable well before 1.98.
- `unsafe` stays inside `rusticide-simd`, one block per kernel entry
  point, each with a `// SAFETY:` comment naming the token; `docs/UNSAFE.md`
  updates its `rusticide-simd` entry to the run-time argument.
- FMA changes rounding. A kernel that must stay bit-identical to portable
  may not use it where portable does a separate multiply and add.
- The usual rules: ASCII, files under 300 lines, SPDX headers, no
  `unwrap`/`expect` in production code, bilingual docs, conventional
  signed commits.

## Left to the builder

- Where the adapter name is reported.
- How the adapter wrappers reach their token (within the rules above).
- The AVX2 layout, group width and token chunking per format.
- How detection behaves under Miri.
- Whether `Gates arm64` keeps any `RUSTFLAGS` (it must at least drop
  `+dotprod` and assert the selected adapter is `neon`).
- How the AVX-512 spike tests in CI, if it gets that far.

## Results

Measured 2026-11-12 at the final tree, both machines on AC. Kernel
figures are G/s (weights, that is multiply-adds, per second on one
thread); prefill and decode are tokens/s with each machine's default
thread pool. The SIMD column is the re-recorded baseline (median of three
sessions' bests); the portable column is a same-day run with
`RUSTICIDE_KERNELS=portable`.

### Apple M1 Pro (8 cores), `neon`

Baseline `apple-m1-pro-8.json`, recorded at 1-minute load 2.45; the
portable run at 2.47.

- Q4_K: row 47.71 vs 29.69 (1.6x), rows8 91.86 vs 30.29 (3.0x).
- Q6_K: row 28.74 vs 26.21 (1.1x), rows8 79.45 vs 27.21 (2.9x).
- Q5_0: row 29.70 vs 20.46 (1.5x), rows8 80.73 vs 24.50 (3.3x).
- Q8_0: row 59.60 vs 35.41 (1.7x), rows8 96.93 vs 35.71 (2.7x).
- Qwen2.5 prefill 962.4 vs 416.4 (2.3x); decode 135.9 vs 124.9.
- TinyLlama prefill 459.9 vs 166.1 (2.8x); decode 102.4 vs 90.7.
- Against the previous baseline (recorded at `44ff283`) every kernel row
  is within 0.5%, and the e2e rows are 2-7% lower: Qwen2.5 decode 145.9
  -> 135.9. A same-session A/B of `44ff283` against the final tree,
  three alternating runs each, gives the same decode (best 140.1 vs
  139.6), so the drop is this desktop's background load that day, not
  the code. The runtime selection costs the M1 nothing measurable, as
  [Sequencing](#sequencing) intended.
- llama.cpp was not re-measured on the M1; the NEON adapter's reference
  stands (`llama-bench -ngl 0 -t 6 -p 128,512 -n 0 -r 5`, BLAS and Metal
  loaded): TinyLlama pp128 318.2, Qwen2.5 pp128 599.3 and pp512 722.8.
  Today's prefill is 145% of TinyLlama's pp128 and 133% of Qwen2.5's
  pp512.

### iMac Pro, Xeon W-2150B (10 cores / 20 threads), `avx2`

Baseline `intel-r-xeon-r-w-2150b-cpu-3-00ghz-20.json`, recorded at
1-minute load 1.98; the portable run at 1.73.

- Q4_K: row 39.53 vs 4.31 (9.2x), rows8 63.31 vs 9.41 (6.7x).
- Q6_K: row 16.97 vs 3.54 (4.8x), rows8 43.41 vs 9.57 (4.5x).
- Q5_0: row 21.75 vs 3.36 (6.5x), rows8 41.93 vs 8.20 (5.1x).
- Q8_0: row 27.61 vs 6.95 (4.0x), rows8 45.57 vs 6.99 (6.5x).
- Every rows8 kernel clears the 2x floor in [Goals](#goals).
- Qwen2.5 prefill 561.0 vs 176.0 (3.2x); decode 56.9 vs 32.7 (1.7x).
- TinyLlama prefill 367.7 vs 65.4 (5.6x); decode 43.0 vs 22.3 (1.9x).
- Against the previous baseline (recorded at `f97c8c2`, before the
  `f32` FMA dots and the loop alignment of Task 8) prefill is +27%
  (Qwen2.5) and +64% (TinyLlama), Qwen2.5 decode +10%, and every kernel
  row is within 5% except two that rose: Q4_K row (+6%) and Q8_0 rows8
  (+9%).
- This machine is noisy end to end: back-to-back sessions lose up to 15%
  of prefill and of the portable `Q4_0` kernel to heat, and llama.cpp
  shows two speed modes (below). The baseline attempts that day gave
  prefill medians of 557-632 (Qwen2.5) and 349-380 (TinyLlama). The
  committed one was recorded with the `bench:baseline` task's own
  commands run directly (three `rusticide-bench run` sessions, then
  `rusticide-bench baseline`); the attempts through
  `mise run bench:baseline` measured the same e2e rows but a `Q4_0`
  portable row 10-16% under the previous baseline, while single runs of
  the same binary measured 6.1-6.9 G/s, above it. No cause was found;
  `Q4_0` has no SIMD kernel and no reference model uses it.

llama.cpp on the iMac Pro, build `8642541b2` (llama.cpp 0.8.0, 20
threads, prompt lengths matching the bench prompt as `llama-tokenize`
counts them: 161 tokens for TinyLlama, 142 for Qwen2.5):

- CPU backend alone, the floor's reference: a CPU-only AVX2 build from
  the Homebrew source (`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON
  -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_METAL=OFF -DLLAMA_CURL=OFF`),
  measured with `llama-bench -m <gguf> -t 20 -p <161|142> -n 0 -r 10
  -o json`. Same day, load 1.81-1.91: TinyLlama best 465 (its samples
  split between about 220 and about 450), Qwen2.5 best 560 (average
  512).
- Prefill against it, baseline median over llama.cpp best: TinyLlama
  367.7 / 465 = **79%**, Qwen2.5 561.0 / 560 = **100%**. Best against
  best in one session (Task 8, after its last change): 414 / 466 =
  **89%** and 667 / 560 = **119%**. The 65% floor holds on both models
  either way.
- Decode against the same build (`-p 0 -n 64`, Task 8): TinyLlama 43.0
  vs 83.2 (52%), Qwen2.5 57.2 vs 86.0 (67%). Decode has no floor.
- The installed Homebrew build, for context only: its CPU backend is
  SSE3-only (no `ymm` instructions) and its BLAS (Accelerate) backend
  takes the batched products. `llama-bench -m <gguf> -t 20 -p <161|142>
  -n 64 -r 5`: TinyLlama pp161 167.3 and tg64 18.1, Qwen2.5 pp142 310.1
  and tg64 13.7. With `-ub 31`, which keeps every product on its SSE3
  CPU backend: 24.6 and 24.2.

### Reference parity

`mise run test:reference` passes 2/2 (TinyLlama and Qwen2.5 greedy
tokens against llama.cpp) on both machines with the default adapter
(`neon` on the M1, `avx2` on the iMac Pro, as `rusticide version`
reports) and with `RUSTICIDE_KERNELS=portable`. 100% coverage on the M1
(`mise run ci:quick`, 5276/5276 lines) and, at Task 8, on the iMac Pro
(5304/5304, the `avx2` modules included).

### AVX-512 (Task 9)

Ruled a follow-up; no adapter was built (Task 10 skipped). A spike added
an `avx512` adapter above `avx2` with a 512-bit `Q4_K` kernel (the other
formats delegated to `avx2`). On the W-2150B, best of four rotated
rounds, each started at 1-minute load under 2.5:

- `Q4_K` kernel: rows8 86.5 vs 65.5 G/s for `avx2` (1.32x), row 51.2 vs
  41.0 (1.25x).
- Prefill: Qwen2.5 612.5 vs 662.3 (-7.5%, `avx512` slower in 3 of 4
  rounds), TinyLlama 432.6 vs 404.0 (+7%). Decode: 58.3 vs 57.7 and
  42.0 vs 42.8.

The build condition in [Sequencing](#sequencing) needs at least 10%
prefill on both models; neither model reaches it, and AVX2 already holds
every floor, so the exception does not apply. The likely cause (not
proven): heavy 512-bit work lowers the Skylake-SP clock for the
surrounding AVX2 and `f32` work, and Qwen2.5 spends more of prefill
outside `Q4_K`. The CI answer the spike proposed: a Miri step built with
`-C target-feature=+avx512f,+avx512bw,+avx512vl` and a `cfg` switch so
Miri runs the real SIMD bodies, plus an isolated Intel SDE job (pinned
SHA-256, no permissions); GitHub-hosted runners guarantee no AVX-512,
QEMU has none, and a self-hosted runner is unsafe for a public repo.

**macOS detection finding.** On the iMac Pro, std's
`is_x86_feature_detected!` reports `avx512f`, `avx512bw` and `avx512vl`
as absent, while `sysctl hw.optional.avx512f` (and `bw`, `vl`) report 1.
Darwin enables AVX-512 register state lazily, on the first use, so XCR0
lacks the opmask and ZMM bits that std's detection requires. An AVX-512
adapter detected through std alone would never arm on macOS; the spike
needed a `sysctlbyname` fallback for macOS.

### Follow-ups

- AVX-512: reopen only with a macOS `sysctl` detection fallback, the CI
  answer above, and a measurement on Ice Lake or later (no heavy
  downclock). The spike is kept outside the tree as a patch.
- The Linux aarch64 release binary gaining `neon` is proven by the `arm`
  CI job only; run the branch's workflow once before the merge (with the
  x86_64 `Gates` job, which proves the `avx2` assertion on GitHub's
  runners).
- The iMac Pro's end-to-end noise (heat between sessions, two speed
  modes) makes its baseline a loose guard for prefill; a longer rest
  between sessions in `bench:baseline` is worth trying there.
- The MSRV CI job sets `RUSTFLAGS` without the x86_64 loop-alignment
  flag, so 1.98's acceptance of `-align-loops=32` is proven only by
  local builds.
- Carried from the task reviews: no test sends a `-128` activation
  through the single-input and on-stack paths; the odd-address checks
  never offset the activations; `Q5_0` rows8 on NEON stays about 1%
  under 0.8.0.
- Attention and RoPE (`f32`) remain the next performance milestone, as
  in [Non-goals](#non-goals).
