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
![Status: SP1 foundation](https://img.shields.io/badge/status-SP1%20foundation-yellow)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/platform-cortex/rusticide/badge)](https://securityscorecards.dev/viewer/?uri=github.com/platform-cortex/rusticide)

## Status

rusticide is at the end of **SP1, the foundation sub-project**. What exists:

- the tensor core (dtypes, devices, shapes, strides, storage) and the dispatch
  key algebra;
- the operator dispatcher and its boxed calling convention;
- a port of torchgen's operator schema model, checked against PyTorch's own
  schema corpus;
- a GGUF file reader and a tokenizer built from GGUF metadata;
- the facade crate with a small embedded CLI.

There is **no model inference yet**. Operators are not generated, there is no
autograd and no GPU backend, and the legacy eager tensor API is slated for
removal in SP2. See the [design](docs/plan/2026-08-21-rusticide-framework-design.md)
for the roadmap and [ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the crates
fit together.

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
- `rusticide-fuzz` -- libFuzzer targets for the GGUF reader and tokenizer; its
  own workspace, not published.

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
