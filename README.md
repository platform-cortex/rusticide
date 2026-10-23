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
![Status: run milestone 1](https://img.shields.io/badge/status-run%20milestone%201-yellow)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/platform-cortex/rusticide/badge)](https://securityscorecards.dev/viewer/?uri=github.com/platform-cortex/rusticide)

## Status

rusticide has finished **SP1, the foundation sub-project**, and **milestone 1
of `rusticide run`**: end-to-end text generation from a GGUF model. What
exists:

- the tensor core (dtypes, devices, shapes, strides, storage) and the dispatch
  key algebra;
- the operator dispatcher and its boxed calling convention;
- a port of torchgen's operator schema model, checked against PyTorch's own
  schema corpus;
- a GGUF file reader and a tokenizer built from GGUF metadata;
- shared f32 kernels, a model contract with an architecture registry, a KV
  cache and a streaming generation loop;
- the `llama` model family (TinyLlama is tested), running on the CPU;
- the facade crate with an embedded CLI, including `rusticide run`.

Operators are not generated, there is no autograd and no GPU backend, and the
legacy eager tensor API is slated for removal in SP2. The next milestones of
`rusticide run` add Qwen and BitNet. See the
[run design](docs/plan/2026-10-22-rusticide-run-design.md) and the
[framework design](docs/plan/2026-08-21-rusticide-framework-design.md) for the
roadmap, and [ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the crates fit
together.

## Crates

Everything lives under `crates/`:

- `rusticide` -- the facade: re-exports the public API and embeds the CLI.
- `rusticide-alloc` -- aligned buffer allocation; the only crate with `unsafe`.
- `rusticide-core` -- tensor value types and the dispatch key algebra.
- `rusticide-dispatch` -- the operator dispatcher and boxed calling convention.
- `rusticide-schema` -- port of torchgen's operator schema data model.
- `rusticide-io` -- GGUF model file reader.
- `rusticide-quantize` -- quantization formats (Q4_0, Q4_K, Q6_K).
- `rusticide-ggml` -- legacy GGML-style compute graph over core tensors.
- `rusticide-sampling` -- token sampling (greedy, top-k, top-p, temperature).
- `rusticide-tokenizer` -- BPE and SentencePiece tokenizer from GGUF metadata.
- `rusticide-kernels` -- shared f32 kernels (matrix products, norms, softmax).
- `rusticide-model` -- the `Model` contract, architecture registry, KV cache
  and streaming generation loop.
- `rusticide-llama` -- the llama model family: configuration, weights and the
  forward pass.
- `rusticide-fuzz` -- libFuzzer targets for the GGUF reader, the tokenizer and
  loading a llama model (`llama_load`); its own workspace, not published.

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
(random when absent). Weights are dequantized to f32 once at load and the
file is read whole.

- **Supported family:** `llama`. `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
  (`Q4_K_M`) is the tested model.
- **Parity:** all 32 greedy tokens match llama.cpp (llama-server 0.5.0, build
  11146) on the 3 fixture prompts, including a Portuguese one. Run it with
  `mise run test:reference` and `RUSTICIDE_MODELS` set.
- **Speed:** 1.58 tokens/s with the command above on an Intel Xeon W-2150B
  (10 cores / 20 threads, 128 GiB, macOS x86_64), measured while the host
  was busy with other work (load average about 17). Load time is about 10 s.
  The design's target is 2 tokens/s; it is a target, not a release blocker.

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

- **`unsafe_code = "deny"`** workspace-wide. Only `rusticide-alloc` opts out,
  under the policy in [docs/UNSAFE.md](docs/UNSAFE.md).
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
