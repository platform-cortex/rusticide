<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide

A pure-Rust deep learning framework modelled on PyTorch's design: a tensor
core, torchgen-style operator schemas, and a dispatch table that routes
operators to kernels. No C, C++, Python, BLAS or vendored bindings.

[![MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
![Rust Stable](https://img.shields.io/badge/rust-stable-purple)
![Unsafe Denied](https://img.shields.io/badge/unsafe-denied-red)
![Status: run milestone 3](https://img.shields.io/badge/status-run%20milestone%203-yellow)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/platform-cortex/rusticide/badge)](https://securityscorecards.dev/viewer/?uri=github.com/platform-cortex/rusticide)

## Status

rusticide has finished **SP1, the foundation sub-project**, and **milestone 3
of `rusticide run`**: end-to-end text generation from a GGUF model, for
TinyLlama, Qwen2.5 0.5B Instruct and BitNet b1.58 2B 4T. What exists:

- the tensor core (dtypes, devices, shapes, strides, storage) and the dispatch
  key algebra;
- the operator dispatcher and its boxed calling convention;
- a port of torchgen's operator schema model, checked against PyTorch's own
  schema corpus;
- a GGUF file reader and a tokenizer built from GGUF metadata;
- shared f32 kernels, a model contract with an architecture registry, a KV
  cache and a streaming generation loop;
- the `llama`, `qwen2` and `bitnet-b1.58` model families (TinyLlama,
  Qwen2.5 and BitNet b1.58 2B 4T are tested), running on the CPU;
- the facade crate with an embedded CLI, including `rusticide run`.

Operators are not generated, there is no autograd and no GPU backend, and the
legacy eager tensor API is slated for removal in SP2. The next step is
attention and RoPE, Milestone B of the
[runtime adapters design](docs/plan/2026-11-02-rusticide-runtime-adapters-design.md).
See the [run design](docs/plan/2026-10-22-rusticide-run-design.md), the
[BitNet design](docs/plan/2026-11-12-rusticide-bitnet-design.md) and the
[framework design](docs/plan/2026-08-21-rusticide-framework-design.md) for the
roadmap, and [ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the crates fit
together.

## Crates

Everything lives under `crates/`:

- `rusticide` -- the facade: re-exports the public API and embeds the CLI.
- `rusticide-alloc` -- aligned buffer allocation; permitted `unsafe`.
- `rusticide-core` -- tensor value types and the dispatch key algebra.
- `rusticide-dispatch` -- the operator dispatcher and boxed calling convention.
- `rusticide-schema` -- port of torchgen's operator schema data model.
- `rusticide-io` -- GGUF model file reader.
- `rusticide-quantize` -- quantization formats (Q4_0, Q4_K, Q5_0, Q6_K,
  Q8_0 and BitNet's ternary I2_S).
- `rusticide-ggml` -- legacy GGML-style compute graph over core tensors.
- `rusticide-sampling` -- token sampling (greedy, top-k, top-p, temperature).
- `rusticide-tokenizer` -- BPE and SentencePiece tokenizer from GGUF metadata.
- `rusticide-kernels` -- shared f32 kernels (matrix products, norms, softmax).
- `rusticide-simd` -- NEON kernels behind safe entry points; permitted
  `unsafe`.
- `rusticide-model` -- the `Model` contract, architecture registry, KV cache
  and streaming generation loop.
- `rusticide-transformer` -- the decoder-only transformer shared by the model
  families: the `Decoder` forward pass, attention, feed-forward, norm and RoPE.
- `rusticide-llama` -- the llama model family: configuration and weights by
  tensor name, loaded into a `rusticide-transformer` `Decoder`.
- `rusticide-bitnet` -- the BitNet b1.58 family (`bitnet-b1.58`): squared
  ReLU, sub-norms and ternary `i2_s` weights on the same `Decoder`.
- `rusticide-fuzz` -- libFuzzer targets for the GGUF reader, the tokenizer and
  loading a llama, qwen2 or bitnet-b1.58 model (`llama_load`); its own
  workspace, not published.

## Getting started

[mise](https://mise.jdx.dev) provisions the Rust toolchain (pinned in
`rust-toolchain.toml`) and every dev tool.

```sh
git clone https://github.com/platform-cortex/rusticide.git
cd rusticide
mise trust
mise run setup
```

Build and test:

```sh
mise run build
mise run test           # cargo-tarpaulin with the 100% coverage gate
```

### The CLI

The binary is built only with the `cli` feature, so library users never pull
in `clap`:

```sh
cargo run -p rusticide --features cli -- schema \
  'add.Tensor(Tensor self, Tensor other) -> Tensor'
cargo run -p rusticide --features cli -- version
```

### Run a model

`rusticide run` completes a raw prompt with a GGUF model, streaming tokens
as they are generated. It is part of the `cli` feature (facade feature
`models`):

```sh
cargo run --release -p rusticide --features cli -- run \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --prompt "Once upon a time," -n 64 --temperature 0
```

Options: `-n/--max-tokens` (default 128), `--temperature` (default 0.8; 0
means greedy), `--top-k` (default 40), `--top-p` (default 0.95) and `--seed`
(random when absent). The file is read whole. Quantized weights stay
quantized in memory and are decoded a block at a time inside the matrix
products; F32 and F16 tensors are held as f32.

- **Supported families:** `llama`, `qwen2` (a variant of the llama
  implementation: half-split RoPE and Q/K/V biases) and `bitnet-b1.58`
  (squared ReLU, sub-norms and ternary `i2_s` weights on the same decoder).
  Tested models: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (`Q4_K_M`),
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)
  and `microsoft/bitnet-b1.58-2B-4T-gguf` (`ggml-model-i2_s.gguf`).
- **Parity:** all 32 greedy tokens match the reference engine (llama.cpp,
  or bitnet.cpp for BitNet) on the fixture prompts, including a Portuguese
  one. TinyLlama: llama-server 0.5.0, build 11146, 3 prompts. Qwen2.5:
  llama-server 0.8.0, build 11960 (commit 8642541b2), 4 prompts, one of
  them a newline and digits prompt. BitNet b1.58 2B 4T: bitnet.cpp
  `1876a3e`, 3 prompts x 32 tokens. Run it with
  `mise run test:reference` and `RUSTICIDE_MODELS` set.
- **Speed**, with the command above (the design's target is 2 tokens/s; it
  is a target, not a release blocker):
  - Apple M1 Pro (8 cores, 32 GiB, macOS arm64): about 18.7 tokens/s,
    4.5 s in all for the 64 tokens, load average about 3.
  - Intel Xeon W-2150B (10 cores / 20 threads, 128 GiB, macOS x86_64):
    1.58 tokens/s, measured while the host was busy with other work (load
    average about 17). Load time is about 10 s.
  - Qwen2.5 0.5B `Q4_K_M` on the Apple M1 Pro (8 cores, 32 GiB, macOS
    arm64): about 34 tokens/s (33.94, 34.75 and 33.50 over three runs of
    `--prompt "Once upon a time," -n 64 --temperature 0`).
  - BitNet b1.58 2B 4T `i2_s`, from the bench baselines (64 decoded
    tokens; `i2_s` runs a portable kernel, no SIMD yet): Apple M1 Pro
    decode 8.65 tokens/s, prefill 10.92; Intel Xeon W-2150B decode 13.58,
    prefill 26.50 (see the Results of the
    [BitNet design](docs/plan/2026-11-12-rusticide-bitnet-design.md)). Run
    it as above:

    ```sh
    cargo run --release -p rusticide --features cli -- run \
      ggml-model-i2_s.gguf \
      --prompt "The capital of France is" -n 32 --temperature 0
    ```

### Reading a GGUF file

`rusticide-io` parses GGUF version 3 files from bytes, with every offset and
size validated, and exposes the metadata and tensor table:

```rust,ignore
use rusticide_io::GgufFile;

let file = GgufFile::from_bytes(&bytes)?;
```

### Quality gate

```sh
mise run fmt            # format (writes)
mise run lint           # fmt check + clippy, warnings are errors
mise run ci:quick       # the gate CI runs on every push
mise run ci             # the full pipeline, adds header and security checks
```

`AGENTS.md` lists every task and the rules the gate enforces.

## Design decisions

- **`unsafe_code = "deny"`** workspace-wide. Only `rusticide-alloc` and
  `rusticide-simd` opt out, under the policy in [docs/UNSAFE.md](docs/UNSAFE.md).
- **Strict TDD** with a 100% coverage gate that includes doctests.
- **Schema first.** Operators are described by torchgen-style schemas and
  routed through a dispatch table, not hard-coded.
- **Fallible allocation.** Sizes that come from untrusted input are checked and
  allocated fallibly, because a panic is a denial of service.

## License

Dual-licensed under **MIT OR Apache-2.0**. See `LICENSES/MIT.txt` and
`LICENSES/Apache-2.0.txt`.

Source files carry SPDX headers:

```text
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
SPDX-License-Identifier: MIT OR Apache-2.0
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and `AGENTS.md` for the workflow and
the rules agents follow.
