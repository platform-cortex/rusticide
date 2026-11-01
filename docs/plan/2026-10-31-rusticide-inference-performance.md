<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Inference performance milestone (notes and plan)

Status: done on branch `perf/inference`, 2026-11-01. Scope is
performance only: tokens per second of `rusticide run` on the CPU, decode and
prompt prefill, measured against llama.cpp's CPU backend on the two reference
models, with greedy output that stays on llama.cpp's path. No `unsafe` code
was added: every kernel is safe Rust the compiler vectorizes.

## Benchmark

Apple M1 Pro (6 performance and 2 efficiency cores, 32 GiB, macOS arm64),
`cargo build --profile profiling --features cli -p rusticide`, greedy
(`--temperature 0`). Models: Qwen2.5 0.5B Instruct `Q4_K_M` (mostly `Q5_0`,
with `Q8_0` embeddings and output) and TinyLlama 1.1B Chat `Q4_K_M` (mostly
`Q4_K`, with `Q6_K` output).

- Decode: `--prompt "The history of the Roman Empire" -n 64`, the
  `tokens/s` line, three runs.
- Prefill: a 140-token (Qwen) or 169-token (TinyLlama) prompt with `-n 1`,
  wall time minus the same run with a one-word prompt (model load).
- llama.cpp 0.8.0 (Homebrew build): `llama-bench -ngl 0`, `tg64` and `pp128`.
  Its best CPU thread count here is 4 (8 threads: 95 and 68 tokens/s).

## Results

Decode, tokens/s:

- Qwen2.5 0.5B: 32.7 at the start, 117 now; llama.cpp CPU 144 (81%).
- TinyLlama 1.1B: 19.1 at the start, 82 now; llama.cpp CPU 97 (85%).
- Single thread: Qwen 13.8 -> 42 (llama.cpp 53), TinyLlama 10.3 -> 22.9
  (llama.cpp 29.6).

Resident memory: Qwen 3.6 GB -> 1.05 GB, TinyLlama 5.4 GB -> 1.36 GB.

Prefill, tokens/s (llama.cpp `pp128` in parentheses):

- Qwen2.5 0.5B, 140 tokens: about 114 at the start, 350 now (652, 54%);
  a 600-token prompt runs at about 340.
- TinyLlama 1.1B, 169 tokens: about 81 at the start, 152 now (238, 64%);
  a 721-token prompt runs at about 136.

## Profiles and fixes

The profiler is `sample` on the running process, folded with
`inferno-collapse-sample`, as in the tokenizer milestone.

1. Weights were widened to f32 at load: the start profile was 34% f32 `dot`
   and 40% idle rayon threads. `Matrix` now keeps GGUF blocks as read
   (`ModelSource::quantized_tensor`, `GgufFile::quantized_tensor`) and rows
   are scheduled in a few runs per thread instead of one task each.
   Memory dropped 3.5x; speed did not move until the kernels changed.
2. Integer kernels. The activation is quantized once per matrix-vector
   product exactly as llama.cpp's CPU backend does it (ggml `Q8_0` for the
   32-element formats, `Q8_K` for K-quants: `Q8Input`, `Q8Kind`), and each
   block multiplies in `i8 x i8 -> i32`.
3. Monomorphized row kernels (`int_dot.rs`). An indirect call per 32-element
   block and slice lengths unknown at compile time hid the loops from the
   vectorizer. One function per format over fixed-size arrays lets LLVM emit
   `sdot` (the `aarch64-apple-darwin` default CPU has dotprod). Details that
   mattered:
   - nibbles are offset to signed values (`q - 8`, corrected through the
     block sums), because LLVM picks `sdot` only when both operands are
     sign-extended;
   - K-quants accumulate the whole block in `i32` and convert to float once,
     as ggml does, which also moved greedy output closer to llama.cpp;
   - `Q4_K` scales and mins unpack four at a time in `u32` words with ggml's
     masks: the scale extraction and the min/offset terms cost more than the
     dot products (ablation: 19.6 -> 26.3 and -> 31.2 G elements/s);
   - `Q5_0` fifth bits come from ggml's 256-entry expansion table.

   Single-core throughput, G elements/s: `Q4_K` 12 -> 29, `Q6_K` 14 -> 26,
   `Q5_0` 6 -> 21, `Q8_0` 13 -> 37, `Q4_0` 12 -> 33.
4. Fewer, cheaper parallel regions. A rayon region costs about 18 us to wake
   and join from outside the pool and 10 us from a pool thread, paid 169
   times a token on Qwen. `matvec_batch` runs Q, K and V in one region and
   gate and up in another (7 -> 4 regions a layer), and `Llama::forward`
   runs inside `rayon::scope`. Activation quantization was rewritten into
   preallocated buffers (2.3 -> 0.7 ns an element).
5. Batched prefill (`matmul_batch`, `Model::forward_batch`): the prompt runs
   through the model in chunks of 64 tokens, each weight row multiplied
   against every token of the chunk while it is in L1. `Q5_0`, `Q4_K` and
   `Q6_K` unpack a run of weight blocks once and reuse it for every token
   (`int_dot/rows.rs`); a single token keeps the fused kernels. Logits are
   bit-identical to running the tokens one at a time, and decode is the
   one-token case of the same path.
6. Attention runs over (token, head) in parallel: the batch stores every
   key and value first, then token `t` attends to cache rows `0..=start+t`.
   This also helps decode at longer contexts (TinyLlama at 400 tokens: 73
   -> 86 tokens/s).

Tried and reverted: four independent partial sums per row (the `index % 4`
store broke vectorization of the 32-element formats), a fully indexed
`Q4_K` loop (the compiler vectorized the outer loop with lane loads).

## Reference parity

The greedy fixtures were captured with `llama-server` on the Metal backend.
llama.cpp's own CPU and Metal backends already pick different Qwen2.5 tokens
within three steps on two of the four prompts, and rusticide now follows the
CPU arithmetic, so the fixtures were recaptured with `-ngl 0` (the settings
field says so). The near-tie margin widened from 0.1 to 0.25 logits: both
engines round activations to 8 bits. TinyLlama matches all 32 tokens of
every prompt; Qwen matches until a near tie in each.

## Coverage

The 100% gate was relaxed for this milestone, but nothing needed it: every
kernel is checked against the dequantized reference on random blocks with
every quant bit varying (a first version kept bit 6 clear and missed the
high bits; mutation checks on `Q6_K` and `Q5_0` confirm the new one), and
the batched path is checked bit for bit against the sequential one.

## Not done

- Tiled matrix kernels for prefill. Unpacking once removed only about 8% of
  the `Q4_K` time: the per-token dot and its reductions dominate. Reaching
  llama.cpp's prefill needs register tiling across rows and tokens, as its
  repacked kernels do. `Q4_0` and `Q8_0` have no batched kernel yet.
- Serial per-layer work: activation quantization, norms, rope and the
  key/value stores still run on one thread.
- Other targets. Release binaries are built for the baseline CPU of each
  target: Linux aarch64 has no dotprod and x86_64 has no AVX2, so the
  kernels fall back to slower instructions there. The Intel iMac server gets
  the full speed only from a local build with
  `RUSTFLAGS="-C target-cpu=native"`; runtime dispatch would need `unsafe`.
- A spinning thread pool: the remaining per-region cost is rayon's wake-up.
  A safe one needs every buffer shared as atomics; an `unsafe` one belongs
  under `docs/UNSAFE.md`, a policy decision.
