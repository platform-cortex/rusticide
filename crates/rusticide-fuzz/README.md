<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-fuzz

libFuzzer targets for [rusticide](https://github.com/platform-cortex/rusticide), driven
by [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Why this crate is special

It is **its own workspace** (note the bare `[workspace]` table in its
`Cargo.toml`) and is listed in the root `Cargo.toml` `exclude`. Two reasons:

- `libfuzzer-sys` requires a nightly toolchain and sanitizer `-Z` flags, while
  the project builds on stable.
- The `fuzz_target!` macro generates an `extern "C"` entry point that cannot
  satisfy the workspace-wide `unsafe_code = "deny"` lint.

Because it sits outside the workspace, `cargo build --workspace`,
`cargo clippy --all-targets` and `cargo nextest run` never see it, and it needs
the explicit `--fuzz-dir .` flag shown below.

## Targets

| Target | Entry point | Looks for |
| ------ | ----------- | --------- |
| `dequantize` | `BlockFormat::dequantize` for every table row | Panics from a block format reading past a block, or from an unaligned or oversized shape |
| `gguf_load` | `GgufFile::from_bytes` | Panics, out-of-bounds reads, and unbounded allocations from a malformed GGUF header, then every tensor via `tensor_by_index` |
| `gguf_tokenizer` | `Tokenizer::from_gguf` + `encode`/`decode` | Panics from vocab/score/special-token metadata that disagrees with itself, including out-of-range ids |
| `llama_load` | `rusticide_llama::load` + one `forward` | Panics from llama hyperparameters or tensor shapes that are zero, overflow, or disagree with each other |

The GGUF targets feed the input straight to `GgufFile::from_bytes`; inputs
under 24 bytes (the GGUF header size) are skipped. `dequantize` uses the first
byte to pick a format and the rest as block data.

## Running

Requires nightly and cargo-fuzz: `rustup toolchain install nightly` and
`cargo install cargo-fuzz`. Run from this directory, since the crate is its own
workspace:

```sh
cd crates/rusticide-fuzz
cargo +nightly fuzz list --fuzz-dir .
cargo +nightly fuzz run --fuzz-dir . dequantize -- -max_total_time=60 -rss_limit_mb=2048
cargo +nightly fuzz run --fuzz-dir . gguf_load -- -max_total_time=60 -rss_limit_mb=2048
cargo +nightly fuzz run --fuzz-dir . gguf_tokenizer -- -max_total_time=60 -rss_limit_mb=2048
```

A random start finds little because almost every input fails the magic check.
Pass a directory holding a small valid GGUF file as a positional argument
(before `--`) to seed the corpus.

Replay a single saved input (this is how a reported crash is reproduced):

```sh
cargo +nightly fuzz run --fuzz-dir . gguf_load artifacts/gguf_load/crash-<hash>
```

## What a finding means

Both entry points parse untrusted files. Per `CLAUDE.md` a panic is a denial of
service, so **any** panic, overflow, or unbounded allocation reachable from
these targets is a bug in rusticide, not in the fuzzer -- fix it with a `Result`,
a `checked_*` arithmetic operation, or a bound on the allocation.

`corpus/` and `artifacts/` are local working data and are git-ignored; only
`Cargo.toml` and `fuzz_targets/` are tracked.

## License

MIT OR Apache-2.0
