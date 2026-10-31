<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Architecture

This is the map of the workspace as it stands at the end of milestone 2 of
`rusticide run`. The reasoning behind it -- why the framework is shaped like
PyTorch, what the oracle is, how the sub-projects are cut -- lives in the
[design document](plan/2026-08-21-rusticide-framework-design.md), and the
crate list there ("Crate graph") describes the target state. This page
describes what is on disk.

## Crate graph

Edges are the `[dependencies]` of each `Cargo.toml`; an arrow reads "depends
on".

```text
rusticide (facade)
  |-- rusticide-core                      always
  |-- rusticide-dispatch -> core          feature "dispatch"
  |-- rusticide-io -> core, quantize      feature "io"
  |-- rusticide-schema                    feature "schema"
  |-- rusticide-tokenizer                 feature "models"
  |-- rusticide-sampling                  feature "models"
  |-- rusticide-model, rusticide-llama    feature "models"
  `-- clap, anyhow                        feature "cli"

rusticide-core -> rusticide-alloc
rusticide-dispatch -> rusticide-core
rusticide-schema   -> (no workspace crates)

rusticide-quantize -> rusticide-core
rusticide-io       -> rusticide-core, rusticide-quantize
rusticide-sampling -> rusticide-core
rusticide-tokenizer -> rusticide-io

rusticide-kernels  -> (no workspace crates)
rusticide-model    -> core, io, kernels, sampling, tokenizer
rusticide-llama    -> core, io, kernels, model
rusticide-ggml     -> core, kernels

rusticide-fuzz (own workspace) -> io, llama, model, tokenizer
```

The `cli` feature enables `schema`, `dispatch` and `models`; `models` enables
`io` and pulls in the model crates, so `rusticide-io` is part of `cli`. The
facade's `io` feature also exposes it without the CLI.

## Layers

From the bottom up:

1. **Capability.** `rusticide-alloc` converts an unsafe capability (aligned,
   uninitialized buffers) into a safe API. It is the only crate that opts out
   of `unsafe_code = "deny"`; see [UNSAFE.md](UNSAFE.md).
2. **Framework core.** `rusticide-core` holds `DType`, `Device`, `Shape`,
   `Strides`, `Storage`, `Tensor`, the version counter and the dispatch key
   algebra. `Tensor` lives here, not in its own crate, because the dispatcher
   needs `Tensor` and `Tensor` needs `DispatchKeySet`.
3. **Framework services.** `rusticide-dispatch` routes an operator to a kernel
   through a dispatch table. `rusticide-schema` is independent of the tensor
   types: it parses and prints operator schemas and nothing else.
4. **Formats.** `rusticide-quantize` defines the block quantization formats;
   `rusticide-io` reads GGUF files and dequantizes through it.
5. **Models.** `rusticide-kernels` holds the f32 kernels, each with exactly
   one implementation; `rusticide-ggml` delegates to it. `rusticide-model`
   defines the `Model` contract, the architecture registry, the KV cache and
   the streaming `generate` loop. `rusticide-llama` implements the llama
   family against that contract. It does not register itself: the CLI registers
   it in `crates/rusticide/src/cli/registry.rs`, one line in one place.
6. **Facade.** `rusticide` re-exports the public API of the layers below
   behind feature gates and embeds the CLI. Only `rusticide-core` is a
   non-optional dependency, so a library consumer never builds `clap`.

`rusticide-ggml`, `rusticide-sampling` and `rusticide-tokenizer` sit beside
the layers rather than in them. They are consumers of the framework: nothing
in the framework layers depends on them. The model crates build on
`rusticide-sampling` and `rusticide-tokenizer` rather than the other way
round.

## Formats, lazy loading and the tokenizer

Milestone 2 added Qwen2.5 (`qwen2`, registered beside `llama` in
`registry.rs`; it is a variant of `rusticide-llama` with half-split (NEOX)
RoPE and Q/K/V biases). Three mechanisms carry it.

- **Block-format table.** `rusticide-quantize` describes each format as a
  `BlockFormat` row. `block_format(type_id)` looks a GGUF tensor type up and
  `formats()` lists every row: Q4_0, Q5_0, Q8_0, Q4_K and Q6_K. Adding a
  format is one module plus one row in the table.
- **Lazy rejection.** `rusticide-io` reads the tensor table without judging
  types: `TensorInfo.type_id` keeps the raw id. An unsupported type fails
  with `UnsupportedTensorType { name, type_id }` only when that tensor is
  materialized, so a file may carry types the model never reads.
- **GPT-2 tokenizer path.** `rusticide-tokenizer` adds a byte-to-unicode
  table and picks the pre-tokenizer from `tokenizer.ggml.pre`: `gpt2` or
  `gpt-2`, `qwen2`, or llama.cpp's default when the key is absent. The
  patterns are llama.cpp's, verbatim, run through `fancy-regex` (they need
  look-ahead). The end-of-generation set is the eos id, the eot id and the
  token texts llama.cpp treats as end of generation.

## SP1 foundation versus rewired legacy

The crates fall in two groups.

**Written for the framework in SP1:**

- `rusticide-alloc`
- `rusticide-core`
- `rusticide-dispatch`
- `rusticide-schema`
- `rusticide` (the facade and its CLI)

**Written for `rusticide run` (milestones 1 and 2):**

- `rusticide-kernels`
- `rusticide-model`
- `rusticide-llama`

**Legacy crates, rewired onto the new core:**

- `rusticide-ggml` -- the GGML-style compute graph, rebound to `Tensor` from
  `rusticide-core` and kept as is until the framework replaces it.
- `rusticide-sampling` -- sampling strategies over logits.
- `rusticide-tokenizer` -- reads vocabulary, scores and special tokens from
  GGUF metadata through `rusticide-io`.
- `rusticide-quantize` -- block formats, now built on core types.
- `rusticide-io` -- the GGUF reader, which began as the model crate's reader
  and now lives in its own crate.

None of the legacy group is public API the framework commits to. SP2 is
expected to move or replace them as the operator surface appears.

## The EagerOps removal seam

The legacy eager operators (`add`, `matmul`, `reshape`, `contiguous`)
compute directly instead of going through the dispatch table, so they
contradict the dispatcher design. Exactly these four are quarantined behind
one trait, `EagerOps`, defined in `rusticide-core` under `src/tensor/eager`.

- `rusticide-core` implements `EagerOps` for `Tensor`.
- `rusticide-ggml` is the only other crate that imports it.
- The facade does not re-export it, and a `compile_fail` doctest keeps it that
  way. Callers must name `rusticide_core::EagerOps` explicitly.

SP2 removes the seam by deleting the four operators and the `ggml` call sites
together, then re-expressing those callers as dispatched operators. Because
the facade never exposed the trait, that removal is not a breaking change to
the facade's API.

The data-access methods (`from_slice`, `get`, `set`, `to_vec`, `ndim`,
`deep_copy`) are not behind the seam: they are inherent `Tensor` methods, the
core's construction and inspection API (the analogue of `torch.tensor`,
`.item` and `.tolist`). They are reachable through the facade, used by
`rusticide-io`, `rusticide-quantize`, `rusticide-sampling` and
`rusticide-ggml`, and SP2 keeps them. The binding rulings are recorded in the "Rulings that bind
later work" section of the
[foundation plan](plan/2026-08-21-rusticide-foundation-plan.md).

## Workspace mechanics

- `members = ["crates/*"]` picks up every crate under `crates/`;
  `rusticide-fuzz` is listed in `exclude` because it is its own workspace
  (libFuzzer needs nightly and cannot satisfy `unsafe_code = "deny"`).
- Every crate inherits the workspace lints, so `-D warnings` and the clippy
  groups apply uniformly.
- `rusticide-schema` carries a corpus test over PyTorch's own schema list
  (`crates/rusticide-schema/tests`); it is the workspace's first oracle-style
  check.
