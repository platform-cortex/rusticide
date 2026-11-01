<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Kernel facade and performance guard (Design)

Status: done, 2026-11-01. Follows the inference performance milestone
([notes](2026-10-31-rusticide-inference-performance.md), release 0.6.0)
and pays the debts it left, most important first. Branch
`chore/kernel-architecture`.

## Intent

Release 0.6.0 made `rusticide run` 3.6 to 4.3 times faster, but it did so
against two written decisions and left the compute code scattered:

- [The run design](2026-10-22-rusticide-run-design.md) lists quantized
  matmul kernels as a non-goal and says reducing memory is explicitly not
  a goal. The code now does both, and the documents say the opposite.
- Nothing in the repository measures speed. Every figure in 0.6.0 came
  from throwaway tools outside it, so a harmless-looking change can undo
  the milestone and every gate stays green.
- The kernels live in three crates: f32 slice kernels in
  `rusticide-kernels`, the integer row kernels (`int_dot`) and the
  activation quantizer (`Q8Input`) in `rusticide-quantize`, and the
  parallel scheduler (`matmul_batch`) in `rusticide-model`. The format
  contract (`BlockFormat`) grew a kernel and an activation kind, so a new
  format can no longer be "a decoder and a table entry".

The owner's direction: one kernel facade with pluggable kernel modules,
in the spirit of hexagonal architecture (ports and adapters), stopping
where PyTorch stops for CPU kernels, and no `unsafe` in this milestone.

## Goals, in order

1. **The documents tell the truth.** The run design and the framework
   design describe what 0.6.0 does and record the architecture below as a
   decision. This comes first because every later step argues from them.
2. **A local performance gate with a baseline.** It fails when speed
   drops; CI only shows the numbers. This comes before the refactor
   because the refactor moves exactly the code it guards.
3. **The kernel facade.** `rusticide-kernels` is the one place that
   computes. Callers use its ports; one adapter, `portable` (today's safe
   auto-vectorized code), implements them, and the seam is real enough
   that a second adapter is a module plus registration, not an edit to
   any model crate.
4. **`rusticide-quantize` is data again.** A block format is its name,
   type id, block size, block bytes and dequantizer, as the run design
   specified. Adding a format is a decoder in `rusticide-quantize` plus a
   kernel registered in `rusticide-kernels`.

## Non-goals

- `unsafe`, `rusticide-simd`, AVX2, NEON or dotprod adapters, runtime CPU
  feature detection beyond the selection seam itself. Next milestone.
- Measuring or tuning the Intel server.
- Tiled prefill kernels, parallelizing the serial per-layer work, new
  formats (`i2_s`).
- Routing the dispatcher's `matmul` operator through the facade. That is
  SP2's job; this milestone only makes it possible.
- Backfilling Portuguese translations of documents this milestone does
  not create (see "Documents").

## Anti-goals

- **Slower than 0.6.0.** Any measured decode or prefill figure on the
  M1 Pro below the committed baseline by more than the gate's tolerance
  fails the milestone, however clean the code looks.
- **A per-block indirect call.** 0.6.0 measured that an indirect call per
  32-element block hid the loops from the vectorizer. Ports are called
  per row run or per batch, never per block.
- **A second dispatcher.** The facade is not a registry of boxed
  operators. Selection happens once, and afterwards a call through a port
  is a plain function call.
- **Weaker parity or coverage.** `test:reference` passes at today's
  NEAR_TIE (0.25), and coverage stays at 100%.

## Architecture: two levels, as in PyTorch

PyTorch selects implementations at two levels, and rusticide copies both:

- **Operator level: the dispatcher** (`rusticide-dispatch`, already
  built, modelled on `c10` dispatch keys). It picks a backend per
  operator call and pays for a boxed argument stack each time. Right for
  "this tensor is on the CPU", wrong for an inner loop. Unchanged here.
- **Kernel level: the facade** (`rusticide-kernels`), modelled on
  PyTorch's `DispatchStub` (`DECLARE_DISPATCH` / `REGISTER_DISPATCH` with
  `CPU_CAPABILITY`). Each port is a slot of plain function pointers;
  each adapter fills every slot; one adapter is chosen once per process
  and cached. A later dispatched `matmul` kernel calls the facade, so the
  two paths converge instead of diverging.

In hexagonal terms:

- **Ports** -- one per kernel family: dot product, element-wise
  (add, multiply, SiLU), normalization (RMSNorm, softmax), rotary
  embedding (both variants), activation quantization (the ggml `Q8_0` and
  `Q8_K` schemes), and quantized matrix products (a row of a block format
  against one or several quantized inputs).
- **Adapters** -- `portable`, the current code, is the only production
  adapter.
- **Selection** -- a function that returns the process's adapter, chosen
  once. With one production adapter it always returns `portable`; the
  seam exists so the next milestone adds a CPU check, not a new API.
- **Scheduling** -- splitting a matrix product across the thread pool
  (today's `matmul_batch` with its fused regions and `TASKS_PER_THREAD`)
  is part of the facade, above the adapter: every adapter gets the same
  scheduler and only supplies row kernels. *Builder's call, settled
  here:* only matrix-product scheduling moves. The head-parallel attention
  loop and the `rayon::scope` around the forward pass are the model's own
  structure and stay in `rusticide-llama`, calling facade ports inside;
  `rusticide-model` drops rayon if nothing else there uses it.

Dependency direction: `rusticide-kernels` depends on `rusticide-quantize`
(block layouts, f16 reads) and on rayon; `rusticide-quantize` depends on
neither. Model crates depend on `rusticide-kernels` for all compute, and
on `rusticide-quantize` only to load and decode weights.

What moves, settled here so the direction holds:

- `BlockFormat` loses `dot_q8`, `dot_rows` and `activation`; it keeps
  name, type id, block size, block bytes and the dequantizer.
- `int_dot`, `Q8Input` and `Q8Kind` move to `rusticide-kernels`, their
  tests with them. The adapter maps a format (by type id) to its row
  kernels and to the activation scheme they pair with; a format the
  adapter has no kernel for is an error at weight load, not a panic.
- The quantized port takes the format's identity and raw row bytes, so no
  type in `rusticide-quantize` names a kernel.

## The performance gate

Two layers of measurement:

- **Kernel benchmarks** need no model: each quantized format's row kernel
  (single thread, single input and batched) and activation quantization,
  on synthetic blocks, reported as elements per second. Runs anywhere,
  including CI.
- **End-to-end benchmarks** need `RUSTICIDE_MODELS`: decode and prefill
  tokens per second for both reference models, with the prompts and
  lengths the 0.6.0 notes used.

The benchmarks are built against the 0.6.0 code, before the refactor
touches anything, and the first baseline is captured from that code on
the M1 Pro: the gate then measures the refactor against what shipped, not
against figures from throwaway tools. *My call:* the tolerance is set
from measured noise (the run-to-run spread over the runs the gate makes,
with a floor of 5%), and is recorded in the baseline file next to the
numbers, so tightening it is a reviewed change too. Decode and prefill
use the same rule. The gate is run by hand. The baseline format is keyed
by machine identity so other machines can add theirs; this milestone
commits only the M1 Pro's.

A committed baseline per machine holds the numbers. The gate task
compares a fresh run against the baseline for the current machine and
fails on a drop beyond the tolerance; a second task rewrites the
baseline, and rewriting it is a reviewed change like any other. A
machine with no baseline gets a clear message, not a pass. CI runs the
kernel benchmarks and prints the table in the job summary without ever
failing on the numbers. The gate is not part of `ci:quick`; AGENTS.md
requires it before merging any change to kernels, scheduling or the
forward pass.

## Documents

- `2026-10-22-rusticide-run-design.md`: an amendment block (the
  document's existing style) that withdraws the two non-goals, states the
  format contract this milestone restores, and points here.
- `2026-08-21-rusticide-framework-design.md`: the two-level selection as
  an architecture decision, and `rusticide-simd` as the home of the
  future native adapters.
- AGENTS.md: current plan, benchmark commands, the gate rule.
- This document and the 0.6.0 performance notes get `.pt-BR.md`
  translations, kept in sync.

## Constraints

Everything in AGENTS.md holds: strict TDD, 100% coverage, no
`unwrap`/`expect` outside tests, files under 300 lines, ASCII, SPDX
headers, `-D warnings`, MSRV 1.96, rebase-only integration.

## Left to the builder

- The exact port signatures, and whether a port is a struct of function
  pointers or a trait with a static adapter, under the anti-goals above.
- Whether the matrix storage type moves into `rusticide-kernels` or stays
  in `rusticide-model`, within "What moves" above.
- The benchmark harness and its output format, as long as `cargo deny`
  passes and the kernel benchmarks finish in CI in a few minutes.
- How the end-to-end benchmark times prefill separately from loading.
- How a machine is identified, run counts, warm-up and the statistic
  compared (one rule for decode and prefill).

## Results

Status: done. The gate (`mise run bench:gate`, M1 Pro) passes with every
metric within tolerance. Tokens/s for `e2e/*`, G elements/s for `kernel/*`
(the q8 rows are quantization throughput):

| metric | baseline | measured |
| --- | --- | --- |
| e2e/qwen2.5/decode | 131.0 | 132.6 |
| e2e/qwen2.5/prefill | 348.5 | 351.1 |
| e2e/tinyllama/decode | 96.8 | 96.0 |
| e2e/tinyllama/prefill | 148.0 | 148.2 |
| kernel/Q4_0/row | 31.72 | 31.99 |
| kernel/Q4_0/rows8 | 31.85 | 32.22 |
| kernel/Q4_K/row | 29.43 | 29.45 |
| kernel/Q4_K/rows8 | 29.78 | 29.93 |
| kernel/Q5_0/row | 20.23 | 20.27 |
| kernel/Q5_0/rows8 | 24.37 | 24.36 |
| kernel/Q6_K/row | 25.96 | 25.95 |
| kernel/Q6_K/rows8 | 26.77 | 26.82 |
| kernel/Q8_0/row | 35.03 | 35.06 |
| kernel/Q8_0/rows8 | 35.48 | 35.44 |
| kernel/q8/block | 1.14 | 1.14 |
| kernel/q8/superblock | 1.10 | 1.09 |

Lessons:

- The gate compares each metric's best of N runs, not the median, against a
  baseline that is the median of three capture sessions' bests. The owner's
  desktop is noisy, and noise only ever slows a run down, so the fastest run
  is the stable statistic.
- Ports must be slice-level. A per-element `silu` port regressed decode by
  14% (a dispatch per element); it was replaced by `silu_assign`, and
  attention got a `dot_rows` port for the same reason.
- `dot` now bounds both slices to the shorter one, so a length mismatch
  never reads out of range or panics.
- Kernels moved without changing arithmetic: the reference parity test and
  the baseline both held after every step.
