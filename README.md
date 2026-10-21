<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide

CPU-first LLM inference engine in pure Rust -- a port of [llama.cpp](https://github.com/ggml-org/llama.cpp).

[![MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
![Rust Stable](https://img.shields.io/badge/rust-stable-purple)
![Unsafe Forbidden](https://img.shields.io/badge/unsafe-forbidden-red)
![Status: Prototype](https://img.shields.io/badge/status-prototype-yellow)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/elioseverojunior/rusticide/badge)](https://securityscorecards.dev/viewer/?uri=github.com/elioseverojunior/rusticide)

## Overview

rusticide reimplements llama.cpp's architecture entirely in Rust with zero unsafe code. It runs large language models on CPU, with optional GPU (CUDA/HIP) support planned behind feature flags.

### Why rusticide?

- **Zero unsafe code** -- `unsafe_code = "deny"` workspace-wide. Memory safety by construction, ruling out whole classes of CVEs common in C/C++ inference engines.
- **CPU-first** -- optimized for CPU inference before GPU backends. Uses Rayon for thread-level parallelism.
- **BitNet b1.58** -- pure integer add/sub matmul kernels (no floating-point multiplications) for ternary-bit models.
- **Modern stable Rust** -- Edition 2024 on a pinned stable toolchain (channel 1.97, MSRV 1.92), no nightly required.
- **llama.cpp compatible** -- reads GGUF model files, follows the GGML compute graph API.

## Project Status

**Phase 2a -- Correctness-First Real-Model Inference (in progress).** The workspace compiles; the CLI `download` and `run` subcommands work; GGUF models can be fetched from HuggingFace Hub and cached locally. The single-precision CPU inference pipeline (tensor ops, compute graph, GGUF reader, tokenizer, sampling, KV-cache attention) has landed. Phase 2a correctness fixes (causal masking on the prompt, real-GGUF `bos_token_id`/`eos_token_id` keys, EOS-terminated generation) have also landed, but real-model parity against llama.cpp has NOT yet been verified: the golden reference token ids are not yet captured.

| Phase | Focus | Timeline | Status |
|-------|-------|----------|--------|
| 0 | Workspace, CI/toolchain, project setup | 2026 Q3 | Done |
| 1a | CLI skeleton, HuggingFace download, GGUF cache | 2026 Q3 | Done |
| 1b | Tensor ops, compute graph, GGUF reader, tokenizer | 2026 Q4 | Done |
| 2 | KV cache, attention, sampling, CLI inference, quantization | 2027 Q1 | In progress |
| 2a | Correctness fixes for real-model inference; llama.cpp parity not yet verified | 2027 Q1 | In progress |
| 3 | SIMD (AVX2, NEON), multithreaded CPU, memory optimization | 2027 Q2 | Pending |
| 3b | BitNet b1.58 ternary kernels | 2027 Q2-Q3 | Pending |
| 4 | GPU backends: CUDA, HIP (AMD) | 2027 Q3-Q4 | Pending |
| 5 | LoRA, vLLM batching, training, HTTP server | 2027 Q4+ | Pending |

## Architecture

```text
rusticide-cli (binary)
    |
rusticide (umbrella re-export lib)
    |
+-------+-------+-------+-------+--------+--------+---------+
model  tokenizer sampling context quantize bitnet  train
    \      |         |        |        |       |      /
     \     |         |        |        |       |     /
      rusticide-ggml (compute graph -- Rayon parallel)
           |
      rusticide-tensor (foundation type -- half, fp16)
```

Eleven crates under `crates/`:

| Crate | Role |
|-------|------|
| `rusticide` | Umbrella library re-exporting all sub-crates |
| `rusticide-cli` | CLI binary for inference |
| `rusticide-tensor` | N-dimensional tensor with fp16 support (foundation) |
| `rusticide-ggml` | Compute graph matching llama.cpp GGML API |
| `rusticide-model` | Model loading, GGUF reader, weight management |
| `rusticide-tokenizer` | BPE and SentencePiece tokenizer |
| `rusticide-sampling` | Sampling strategies: greedy, top-k, top-p, temperature |
| `rusticide-context` | Inference context with KV cache for transformers |
| `rusticide-quantize` | Quantization formats: Q4_0, Q5_1, Q8_0, etc. |
| `rusticide-bitnet` | 1-bit ternary kernels for BitNet b1.58 |
| `rusticide-train` | Training/fine-tuning scaffold (later phase) |

## Getting Started

### Prerequisites

- [mise](https://mise.jdx.dev) -- provisions the Rust toolchain and all dev tools
- Rust stable (installed automatically by mise via `rust-toolchain.toml`)
- Node 24+ (for Markdown lint toolchain)

### Setup

```sh
git clone https://github.com/elioetibr/rusticide.git
cd rusticide
mise trust
mise run setup
```

This installs the Rust toolchain, all cargo tools, git hooks, and npm dependencies.

### Build

```sh
mise run build          # or: cargo build --workspace
```

### Test

```sh
mise run test           # cargo nextest run (primary test runner)
mise run test:doc       # cargo test --doc (doctests)
```

### Run

```sh
# Download a GGUF model from HuggingFace Hub (cached to ~/.cache/huggingface/hub/)
cargo run -p rusticide-cli -- download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

# Run inference on a downloaded model
cargo run -p rusticide-cli -- run --model ~/.cache/huggingface/hub/tinyllama-1.1b-chat-v1.0.Q2_K.gguf --prompt "Hello"
```

### Full Quality Gate

```sh
mise run ci:quick       # fmt:check + clippy + tests + doctests (~1 min)
```

Or run every check manually in the prescribed order:

```sh
cargo sort --workspace --check \
  && taplo format --check Cargo.toml crates/*/Cargo.toml \
  && mise run markdownlint \
  && cargo fmt --check \
  && cargo clippy -- -D warnings \
  && cargo audit --deny warnings \
  && cargo deny check \
  && cargo vet \
  && cargo machete \
  && cargo nextest run
```

## Tooling

All dev tools are provisioned by mise and pinned in `mise.toml` for reproducible builds across machines.

| Tool | Purpose |
|------|---------|
| `cargo nextest` | Test runner (faster than `cargo test`) |
| `cargo clippy` | Lint with `-D warnings` |
| `cargo fmt` | Rust formatting |
| `cargo sort` | Cargo.toml dependency sorting |
| `taplo` | TOML formatting |
| `cargo audit` | Security advisory scanning |
| `cargo deny` | License, ban, and source policy enforcement |
| `cargo vet` | Supply-chain audit |
| `cargo machete` | Unused dependency detection |
| `cargo tarpaulin` | Code coverage |
| `cargo criterion` | Benchmarking |
| `cargo flamegraph` | Performance profiling |
| `cargo pgo` | Profile-guided optimization |
| `cargo fuzz` | Fuzz testing |
| `cargo mutants` | Mutation testing |
| `cargo semver-checks` | API semver verification |
| `gitleaks` | Secret scanning |
| `hk` | Git hook manager (Rust) |
| `git-cliff` | Changelog generation |
| `rumdl` | Markdown lint |
| `yamllint-rs` | YAML lint (Rust) |
| `actionlint` | GitHub Actions workflow lint |
| `reuse` | SPDX/REUSE compliance |

## Key Design Decisions

- **No `unsafe`** -- forbidden workspace-wide. Safety without unsafe blocks.
- **GGUF format** -- model file format compatible with the llama.cpp ecosystem.
- **CPU-first**, GPU behind feature flags -- keeps the core simple and portable.
- **Rayon** for thread-level parallelism in the compute graph.
- **Virtual workspace** -- each crate independently versioned, faster compilation, clean dependency isolation.
- **TDD** -- all code is written test-first.

## License

Dual-licensed under **MIT OR Apache-2.0**. See `LICENSES/MIT.txt` and `LICENSES/Apache-2.0.txt`.

All source files carry SPDX headers:

```text
SPDX-FileCopyrightText: 2026 XLLM contributors
SPDX-License-Identifier: MIT OR Apache-2.0
```

## Contributing

See `docs/guidelines/contribution.md` and `AGENTS.md` for AI agent rules and workflow.
