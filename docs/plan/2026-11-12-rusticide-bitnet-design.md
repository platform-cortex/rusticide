<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# BitNet b1.58: milestone 3 of `rusticide run` (Design)

Status: draft, 2026-11-12. Refines milestone 3 of the
[`rusticide run` design](2026-10-22-rusticide-run-design.md), which still
holds for what this milestone delivers; this document records what has
changed since and the calls made for it. Branch `feat/bitnet`.

## Intent

`rusticide run ggml-model-i2_s.gguf --prompt ...` runs Microsoft's BitNet
b1.58 2B 4T and produces the same greedy tokens as bitnet.cpp. It is the
last of the three `rusticide run` milestones the framework spec puts
before SP2-SP7, and the owner chose it ahead of the attention/RoPE
performance work (2026-11-12).

The run design was written on 2026-10-22. Since then the weights stay
quantized in memory, every format's products run through
`rusticide-kernels`, and kernel adapters are chosen at run time. And the
attention BitNet must share with `llama` lives `pub(crate)` inside
`rusticide-llama`. This milestone therefore first moves the shared
transformer blocks into their own crate, then adds BitNet on top.

## Sequencing

1. **Reference fixtures first** (design author's call): bitnet.cpp is
   built on the iMac Pro and the fixtures captured before any code moves.
   They are the milestone's go/no-go: if neither the legacy build nor the
   fallback (see below) yields fixtures, or the two disagree, the
   milestone stops and goes back to the owner; no other oracle is used.
2. **`rusticide-transformer`** (owner's call, 2026-11-12): attention with
   GQA, the RoPE styles, the KV cache mixing, the gated feed-forward with
   a pluggable activation, per-block RMSNorm and the token-parallel batch
   helpers move out of `rusticide-llama`. A pure move: see "Refactor
   proof" below.
3. **`i2_s`**: the format in `rusticide-quantize`, its activation
   quantization and a portable kernel in `rusticide-kernels`, then a
   decode-speed checkpoint on the M1 (see Goals).
4. **`rusticide-bitnet`**: the `bitnet-b1.58` family, registered next to
   `llama` and `qwen2`; `test:reference` gains BitNet.
5. Bench rows, results, docs, release 0.10.0.

## Refactor proof

"Bit-identical" means identical logits, not only identical tokens
(design author's call): `test:reference` checks tokens and tolerates
near-ties, so it cannot prove a move. Before the first move, a task
records, per reference prompt and position, a hash of the f32 bits of
the logits for TinyLlama and Qwen2.5 on the machine at hand (kept local,
never committed: it is machine-specific and needs the model files). Every
refactor task ends by matching it. So the move may not change summation
order, fuse loops or reorder batches; anything that would is a separate,
later change. `test:reference` and `bench:gate` still run at the end of
the refactor.

## Goals

- The run design's success criterion, unchanged: for at least three
  prompts (one non-ASCII), the prompt tokenizes to bitnet.cpp's ids and
  the first 32 greedy tokens match, under the near-tie rule (a divergence
  passes only when rusticide's top-1/top-2 margin is below 0.1).
- `rusticide-llama` keeps only what is `llama`/`qwen2`-specific: config,
  weight layout, variant parameters. `rusticide-bitnet` keeps only what is
  BitNet-specific: config, `i2_s` weights, sub-norms, squared ReLU wiring.
- `i2_s` is a module plus table entries in `rusticide-quantize` and
  `rusticide-kernels`, like every other format.
- Decode at least 2 tokens/s on the M1 Pro (the run design's target).
  A target, not a hard floor (design author's call): it is measured right
  after the portable kernel lands; a miss is reported to the owner then,
  who decides whether a NEON `i2_s` kernel joins this milestone, and is
  otherwise recorded in Results for the performance milestone.

## Non-goals

- SIMD `i2_s` kernels (NEON, AVX2). Portable first; a measured
  performance milestone follows, as it did for `llama`. (Design author's
  call: correctness and the shared-crate refactor are the risk here.)
- The TL1/TL2 lookup-table formats (`BITNET_ARM_TL1`, `BITNET_X86_TL2`)
  and the ARM 64-weight `i2_s` layout. Only the layout of the published
  file is read.
- Chat templates. `run` sends the raw prompt, as for the other families.
- Matching bitnet.cpp's logits bit for bit (f16 KV cache, f16 lm_head
  activations, thread-order float sums): tokens, not logits, are the bar.

## Anti-goals

- Any copy of attention, RoPE, RMSNorm, softmax or the feed-forward into
  `rusticide-bitnet`. The refactor exists to prevent it.
- `if arch == "bitnet-b1.58"` outside the family crate and its one
  registry line.
- A `llama`/`qwen2` regression hidden by the refactor: a changed token in
  `test:reference` or a `bench:gate` miss blocks the milestone.
- Relaxing AGENTS.md rules, or any panic on a hostile GGUF file (the new
  `i2_s` sizing and scale paths get fuzz coverage).

## Reference: which bitnet.cpp

Recon of `microsoft/BitNet` (2026-11-12) found that its current HEAD
(`0b341e5`, llama.cpp submodule moved to `isHuangXin/llama.cpp@390c307`
on 2026-07-15) builds the BitNet FFN with SiLU instead of squared ReLU and
has no NEON `i2_s` path, its scalar fallback reading another layout. The
legacy submodule (`Eddie-Wang1120/llama.cpp` at `1f86f058d`, BitNet
`1876a3e`, 2026-01-27; the next BitNet commit, `69bf64d`, moves the
submodule to a SiLU build) implements the model as published. Decision
(design author): fixtures come from the legacy fork, built on the iMac
Pro, loading the published file unchanged (built and captured
2026-11-12: Apple clang 17, BLAS and Metal off, one type-only `const`
patch that clang needs; see the fixture's `reference` block). The fixture
records the BitNet and submodule commits. If that build cannot be made to
work, BitNet `765741d` (2025-05) on the M1 is the fallback; its NEON path
reads the same 128 layout (inferred from source, untested).

Fixture capture: `llama-cli -m <file> -p "<prompt>" -n 32 --temp 0 -t <n>
-c 2048 -ngl 0` (conversation mode is off by default in the legacy fork,
BOS prepended); prompt and generated token ids taken from the tool (the
builder finds the flag or endpoint that prints them).

## Facts from recon

Citations are into the legacy fork unless noted; see the task's ledger for
file:line references.

- **File.** `microsoft/bitnet-b1.58-2B-4T-gguf`, `ggml-model-i2_s.gguf`,
  1,187,801,280 bytes, SHA-256
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
  332 tensors: 210 `i2_s` (type id 36), 121 F32, 1 F16
  (`token_embd`, tied: there is no `output.weight`).
- **`i2_s` layout (x86, the published file).** 2-bit codes, 0 -> -1,
  1 -> 0, 2 -> +1, 3 unused. Blocks of 128 weights in 32 bytes: byte `k`
  of a block holds weights `k` (bits 7-6), `k+32` (5-4), `k+64` (3-2) and
  `k+96` (1-0). Rows are `ne0/4` bytes, contiguous. One f32 scale per
  tensor right after the packed data, then 28 unused bytes: a tensor is
  `ne0*ne1/4 + 32` bytes. Verified on the file's bytes (layout correlation
  test; all 210 scales in [0.744, 4.596]).
- **Activation quantization.** Per token row: `max = max(|x|)` computed
  in f64 starting from `1e-5`; `s = (f32)(127 / max)`;
  `q = nearest_int(x * s)` (round half to even) clamped to [-128, 127];
  also `sum(q)`. The product is
  `(sum(code * q) - sum(q)) / s * w_scale` -- equivalently the integer
  dot of `code - 1` with `q`, exact in i32.
- **Forward, per block.** `attn_norm` (RMS) -> separate q, k, v `i2_s`
  products -> RoPE NEOX on q, k (dim 128, base 500000) -> attention, GQA
  20/5 heads, head dim 128, scale `1/sqrt(128)` -> `attn_sub_norm` (RMS)
  -> `attn_output` -> residual -> `ffn_norm` (RMS) -> gate and up ->
  `relu(gate)^2 * up` -> `ffn_sub_norm` (RMS, width 6912) -> `ffn_down`
  -> residual. Then `output_norm` and logits from the F16 `token_embd`.
  RMS eps 1e-5. No optional scale tensors in this file.
- **Tokenizer.** `gpt2` model, Llama 3 vocabulary (128256 tokens, 280147
  merges), no `tokenizer.ggml.pre`: llama.cpp's `DEFAULT` pre-tokenizer,
  which rusticide already selects when the key is absent. BOS 128000
  prepended; EOS 128001; `<|eot_id|>` 128009 also ends generation.

## Architecture

- **`rusticide-transformer`** depends on `rusticide-model` (the
  `KvCache`) and `rusticide-kernels`. It holds the decoder building
  blocks, parameterized by small config structs, and nothing that names a
  family. `rusticide-llama` and `rusticide-bitnet` depend on it;
  `rusticide-model` never does.
- **Products behind one seam.** A block calls "project these activations
  through this weight". For `llama` the weight is a GGUF-block format; for
  BitNet it is `i2_s`, whose product first quantizes the activations to
  int8. The shared blocks must not know which; the builder decides
  whether that seam is a trait, an enum of weight kinds or the existing
  kernel registry, as long as BitNet's activation quantization stays out
  of `rusticide-transformer`.
- **Sub-norms.** BitNet's `attn_sub_norm` and `ffn_sub_norm` are optional
  hooks in the shared attention output and feed-forward paths (absent for
  `llama`), not a forked block.
- **`i2_s` format contract.** Its size is per tensor (`numel/4 + 32`), not
  per block, and its scale is per tensor. `rusticide-quantize`'s format
  contract gains what that needs (builder's call how), keeping "a format
  is a module plus one table entry".
- **lm_head.** The tied F16 `token_embd` goes through the existing F16
  path.

## Constraints

- AGENTS.md in full: strict TDD, 100% coverage, no `unwrap`/`expect` in
  production code, `unsafe` only in `rusticide-alloc`/`rusticide-simd`,
  files under 300 lines, ASCII only, SPDX headers, MSRV 1.98.
- The model file is never committed; `RUSTICIDE_MODELS` gains the BitNet
  file on both machines, and `test:reference` names it.
- Fixtures commit token ids and provenance only (tool, commits, flags,
  file SHA-256), never model bytes.
- Release binaries stay portable: no `target-cpu=native`.

## Left to the builder

- The exact shape of `rusticide-transformer`'s API and of the product
  seam, within the constraints above.
- How the `i2_s` kernel is laid out in `rusticide-kernels` (row and
  batch entry points like the other formats, or fewer if BitNet's forward
  needs fewer), and its bench rows.
- How fixtures are captured from the legacy binaries, as long as each
  records tool, commits and flags.

## Results

Filled in by the milestone's last task.
