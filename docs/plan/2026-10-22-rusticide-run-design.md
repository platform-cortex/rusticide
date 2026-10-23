<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide run -- End-to-End Text Generation (Design)

Status: approved for planning, 2026-10-22. Supersedes the order of the
sub-projects in
[the framework design](2026-08-21-rusticide-framework-design.md) (see
"Roadmap change" below); it does not change that document's architecture.

## Intent

After SP1 (v0.2.0) rusticide can parse a GGUF file, dequantize Q4_K and
Q6_K weights and run individual ops, but it cannot run a model. A hands-on
test on 2026-10-22 against real files showed why that matters: pieces that
are 100% covered by unit tests still failed on real data (the GPT-2
tokenizer path, and the loader rejecting a whole file over one unsupported
tensor type).

The next goal is therefore one command that generates text from a real
model on CPU:

```sh
rusticide run model.gguf --prompt "Ola"
```

It must work for more than one model family, BitNet included, and adding a
family or a quantization format must mean adding a crate or a module, not
editing the core or the CLI.

## Goals

- `rusticide run <file.gguf> --prompt <text>` streams generated text to
  stdout.
- The model family is detected from the file (`general.architecture`); the
  user never names it.
- Three milestones, each ending with a working `rusticide run`:
  1. **TinyLlama 1.1B Chat, Q4_K_M** (architecture `llama`, SentencePiece
     tokenizer).
  2. **Qwen2.5 0.5B Instruct** (architecture `qwen2`, GPT-2 byte-level BPE
     tokenizer).
  3. **BitNet b1.58 2B 4T, `i2_s`** (architecture `bitnet-b1.58`).
- Families and quantization formats plug in behind shared contracts.

## Success criterion

For a fixed prompt, with greedy decoding (temperature 0), rusticide must
produce the same tokens as the reference implementation of that model:
`llama.cpp` for milestones 1 and 2, Microsoft's `bitnet.cpp` for
milestone 3.

Concretely, per model and per prompt:

- the prompt tokenizes to exactly the reference token ids;
- the first **32** generated tokens are identical to the reference;
- at every one of those positions, rusticide's top-1 token equals the
  reference's.

The 32-token window exists because the references compute with quantized
activations while rusticide computes in f32 (except where noted for
BitNet); tiny numeric differences may legitimately flip a token.

**Near-tie rule.** A divergence inside the window passes only when, at the
diverging position, rusticide's own top-1 and top-2 logits differ by less
than **0.1**; the comparison stops there and the test passes. A divergence
with a larger margin is a bug. Reproducing llama.cpp's activation
quantization to avoid near-ties is out of scope (see Non-goals). This rule
applies to all three milestones; for BitNet, reproducing bitnet.cpp's
activation quantization (see "Left to the builder") is expected to make
near-ties rare.

Reference outputs are generated once with the reference tool and committed
as fixtures: prompt text, prompt token ids, generated token ids, and the
reference tool and version used. At least three prompts per model,
including one non-ASCII prompt (e.g. Portuguese with accents).

Speed is secondary: the target is at least **2 tokens/s** for TinyLlama on
the development machine (Apple Silicon, many cores, 128 GiB). It is a
target, not a release blocker and not a CI gate: each milestone measures
and records tokens/s in its release notes. Parallelizing the matrix
products with rayon (already a workspace dependency) is part of milestone 1
regardless, because it is cheap.

> **Amended 2026-10-23.** The development machine is not Apple Silicon:
> it is an Intel Xeon W-2150B (10 cores / 20 threads, 128 GiB, macOS
> x86_64). The 2 tokens/s target stands unchanged.

## Non-goals

- GPU, training, autograd, the ~100-operator SP2 surface.
- Chat templates and multi-turn chat. `--prompt` is raw completion text.
- Matching reference output beyond the 32-token window, or matching
  non-greedy sampling bit for bit.
- mmap-based loading (needs `unsafe` outside `rusticide-alloc`).
- Quantized matmul kernels (computing on quantized blocks directly). Weights
  are dequantized at load time; see "Memory".
- Formats other than GGUF.
- Downloading models from the CLI.

## Anti-goals

These would make the work a failure even if it ran:

- A family wired in with `if arch == "..."` branches spread through the
  CLI or the generation loop instead of behind the model contract.
- Kernels copied per family. Llama and BitNet share RMSNorm, RoPE, softmax
  and attention; one implementation each.
- Relaxing the project rules (AGENTS.md) to get there faster: strict TDD,
  the 100% coverage gate, no `unwrap`/`expect` in production code, no
  `unsafe` outside `rusticide-alloc`, files under 300 lines, ASCII only.
- Any panic on a hostile or truncated GGUF file. The existing fuzz targets
  must keep passing, and the new loading paths get fuzz coverage.

## Architecture

### Crates

New crates, all under `crates/` (picked up by the workspace glob):

- **`rusticide-model`** -- the family-independent runtime:
  - the `Model` contract: build from a loaded GGUF file, then
    "given these token ids and this cache, return logits for the next
    token";
  - a registry mapping `general.architecture` strings to constructors;
  - the KV cache;
  - the generation loop (prompt prefill, decode step, sampling via
    `rusticide-sampling`, stop on EOS or `--max-tokens`).
- **`rusticide-llama`** -- the `llama` family, also covering `qwen2`
  (differences: Q/K/V bias, RoPE variant, tokenizer; parameterized, not
  forked).
- **`rusticide-bitnet`** -- the `bitnet-b1.58` family.

The CLI's `run` subcommand only parses flags, loads the file, asks the
registry for a model and drives the generation loop. Registering a family
is one line in one place.

Dependency direction: family crates depend on `rusticide-model`;
`rusticide-model` never depends on a family. The facade gains a feature
(e.g. `models`) that pulls the families in; `cli` enables it.

### Quantization formats

Today `rusticide-io` maps raw GGML type ids through a closed enum and
rejects the whole file on an unknown id. That changes:

- **Lazy rejection.** Parsing a file records every tensor's raw type id.
  An unknown or unsupported type is an error only when that tensor is
  materialized. Metadata and the tokenizer stay readable for any valid
  GGUF file.
- **Formats behind one contract** in `rusticide-quantize`: type id, block
  size, block byte length, dequantize-to-f32. The loader looks formats up
  in one table instead of matching on an enum. Adding a format is a module
  plus one table entry.
- Formats added by milestone: none for 1 (Q4_K, Q6_K, F32, F16 exist);
  Q8_0 and Q5_0 for 2 (Qwen2.5 0.5B's 896-wide tensors are not k-quant
  compatible, so llama.cpp falls back to them); `i2_s` (GGML type id 36 in
  bitnet.cpp's fork) for 3.

### Tokenizer

- Milestone 1 uses the existing SentencePiece path; it must match
  llama.cpp's tokenization of the fixture prompts.
- Milestone 2 rewrites the GPT-2 byte-level BPE path, which is broken
  today: it assumes byte `b` is token id `b`, drops whitespace in
  pre-tokenization, and decode does not reverse the GPT-2 byte-to-unicode
  mapping. The fix needs the byte-to-unicode table, the pre-tokenizer
  regex selected by `tokenizer.ggml.pre` (with llama.cpp's default when
  the key is absent), and merge ranks.
- BitNet's file uses the GPT-2 tokenizer with the Llama 3 vocabulary
  (128256 tokens, 280147 merges, no `tokenizer.ggml.pre` key), so
  milestone 3 depends on milestone 2's tokenizer.

### Kernels

The forward pass needs, in f32: matrix-vector and matrix-matrix products,
RMSNorm, RoPE (both the adjacent-pair `llama` variant and the half-split
`neox` variant `qwen2` uses), softmax, SiLU, element-wise multiply/add,
embedding row lookup, and for BitNet squared ReLU and its sub-norms.

Most exist inside `rusticide-ggml`'s graph ops. Builder's choice where the
shared implementations live, under one constraint: one implementation per
kernel, reused by the graph and the model crates (move, don't copy).

### Memory

Weights are dequantized to f32 once at load. TinyLlama needs about 4.4 GiB
and BitNet 2B about 10 GiB; acceptable on the development machine and
explicitly not a goal to reduce here. The file itself is still read whole
(`fs::read`), as today.

## CLI

```text
rusticide run <MODEL> --prompt <TEXT>
    [-n, --max-tokens <N>]     default 128
    [--temperature <T>]        default 0.8; 0 means greedy
    [--top-k <K>]              default 40
    [--top-p <P>]              default 0.95
    [--seed <S>]               default: random
```

Tokens stream to stdout as they are decoded; diagnostics (load time,
tokens/s) go to stderr. An unknown architecture, an unsupported tensor type
or a malformed file is a clear one-line error and a non-zero exit, never a
panic. Defaults follow llama.cpp's.

## Testing

- **Unit tests** under the existing rules (strict TDD, 100% tarpaulin
  coverage). Model crates are tested against tiny synthetic GGUF files
  built in the tests (a few hundred parameters, random weights), so the
  coverage gate never needs a real model.
- **Reference tests** against real models live outside the coverage run,
  are skipped unless an environment variable (e.g. `RUSTICIDE_MODELS`)
  points at a directory holding the model files, and compare against the
  committed fixtures as defined in "Success criterion". Model files are
  never committed. CI does not run them initially.
- **Fuzzing**: the lazy-rejection loader and every new format's
  dequantizer get fuzz targets or extended existing ones.

## Milestones and releases

Each milestone is its own implementation plan and ends with a minor
release through the existing pipeline (`mise run release:prepare`):

| Milestone | Model | Release |
|---|---|---|
| 1 | TinyLlama 1.1B Chat Q4_K_M | 0.3.0 |
| 2 | Qwen2.5 0.5B Instruct | 0.4.0 |
| 3 | BitNet b1.58 2B 4T `i2_s` | 0.5.0 |

Milestone 1 builds the shared parts (`rusticide-model`, the contract, the
registry, KV cache, generation loop, `run`) and the `llama` family. 2 adds
the tokenizer rewrite, lazy rejection, Q8_0/Q5_0 and `qwen2`. 3 adds
`i2_s` and `rusticide-bitnet`.

Canonical model files (design author's call). Each milestone's first task
downloads its file and records the SHA-256 in the fixtures, so fixtures
stay reproducible:

- Milestone 1: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`,
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`. (The TinyLlama file on disk today
  is Q2_K, unsupported until a later format addition.)
- Milestone 2: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`,
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`. The formats milestone 2 adds are
  whatever this file actually contains beyond the supported set; Q8_0 and
  Q5_0 are the expectation, the file is the authority.
- Milestone 3: `microsoft/bitnet-b1.58-2B-4T-gguf`,
  `ggml-model-i2_s.gguf`, already at
  `~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/`.
  rusticide implements the `i2_s` layout this file uses. bitnet.cpp's
  `i2_s` packing may depend on the CPU (block size differing between x86
  and ARM builds; unverified). Milestone 3's first task verifies this; if
  so, fixtures come from a bitnet.cpp build that reads this file correctly
  (an x86 machine or an x86 container), not from a re-converted file.

## Roadmap change

The framework design's SP2-SP7 (CPU operator surface and oracle, autograd,
GPU, nn/optim, operator expansion, distributed) move after these three
milestones. Its "Sub-project decomposition" section gets an amendment
pointing here. The items listed under its "Deferred from SP1" stay
deferred unless a milestone needs one.

## Facts the builder needs

- `llama.cpp` is installed (`/usr/local/bin/llama-cli`, `llama-simple`;
  Homebrew `llama.cpp` 0.5.0) and is the reference for milestones 1-2.
- `bitnet.cpp` is not installed; milestone 3 builds it (from
  `github.com/microsoft/BitNet`) to produce fixtures, outside the repo.
- The BitNet GGUF declares: architecture `bitnet-b1.58`, 30 blocks,
  embedding 2560, FFN 6912, 20 heads / 5 KV heads, RoPE dimension 128,
  freq base 500000, RMS eps 1e-5, context 4096. Tensor types: 210 x
  `i2_s` (id 36), 121 x F32, 1 x F16 (`token_embd`). Each block has
  `attn_sub_norm` and `ffn_sub_norm` in addition to the usual norms.

## Left to the builder

- Where shared kernels live, and the exact shape of the `Model` contract
  and KV cache, within the constraints above.
- Whether the generation loop uses `rusticide-ggml`'s graph or calls
  kernels directly. The graph recomputes everything on each forward and
  has no notion of a cache, so direct calls are the expected answer.
- The `i2_s` byte layout and BitNet's activation quantization: read them
  from bitnet.cpp's source. If matching bitnet.cpp's tokens requires
  reproducing its int8 activation quantization, reproduce it.
- How fixtures are captured from the reference tools (which binary,
  flags), as long as the fixture records the tool and version.
- The regex engine for GPT-2 pre-tokenization (the standard pattern uses
  look-ahead, which the `regex` crate lacks).
