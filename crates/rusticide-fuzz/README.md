# rusticide-fuzz

libFuzzer targets for [rusticide](https://github.com/elioseverojunior/rusticide), driven
by [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Why this crate is special

It is **its own workspace** (note the bare `[workspace]` table in its
`Cargo.toml`) and is listed in the root `Cargo.toml` `exclude`. Two reasons:

- `libfuzzer-sys` requires a nightly toolchain and sanitizer `-Z` flags, while
  the project pins stable 1.97.
- The `fuzz_target!` macro generates an `extern "C"` entry point that cannot
  satisfy the workspace-wide `unsafe_code = "deny"` lint.

Because it sits outside the workspace, `cargo build --workspace`,
`cargo clippy --all-targets` and `cargo nextest run` never see it, and it needs
the explicit `--fuzz-dir crates/rusticide-fuzz` flag shown below.

## Targets

| Target | Entry point | Looks for |
| ------ | ----------- | --------- |
| `gguf_load` | `Model::load` | Panics, out-of-bounds reads, and unbounded allocations from a malformed GGUF header |
| `gguf_tokenizer` | `Tokenizer::from_gguf` + `encode`/`decode` | Panics from vocab/score/special-token metadata that disagrees with itself |

Both write their input to a scratch file first, because `Model::load` takes a
path rather than a byte slice.

## Running

Requires nightly: `rustup toolchain install nightly`.

```sh
mise run fuzz list                  # show targets
mise run fuzz run gguf_load         # fuzz until you stop it
mise run fuzz run gguf_load -- -max_total_time=60
```

Or call cargo-fuzz directly:

```sh
cargo +nightly fuzz run --fuzz-dir crates/rusticide-fuzz gguf_load
```

Replay a single saved input (this is how a reported crash is reproduced):

```sh
cargo +nightly fuzz run --fuzz-dir crates/rusticide-fuzz gguf_load artifacts/gguf_load/crash-<hash>
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
