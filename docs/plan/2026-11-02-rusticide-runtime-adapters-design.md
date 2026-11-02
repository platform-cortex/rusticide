<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Runtime-selected kernel adapters (Design)

Status: draft, 2026-11-02. Follows the
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

(Filled in by the final task.)
