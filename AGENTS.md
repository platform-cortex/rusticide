<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide -- AGENTS.md

Canonical rules for every coding agent in this repo. `CLAUDE.md` imports this
file; edit here, never there.

## Project identity

- Pure-Rust deep learning framework modelled on PyTorch's design (tensor core,
  torchgen-style operator schemas, a dispatch table routing ops to kernels).
  No C, C++, Python, BLAS or bindgen of vendored C.
- Spec: `docs/plan/2026-08-21-rusticide-framework-design.md`. Current plan:
  `docs/plan/2026-08-21-rusticide-foundation-plan.md`. Consult the spec for
  architecture decisions and update it when they change.
- Plans are executed task-by-task with `superpowers:subagent-driven-development`;
  progress lives in `.superpowers/sdd/<plan>/progress.md`.

## Workspace

- Virtual workspace (root `Cargo.toml` is `[workspace]` only, resolver 3).
  `members` is an **explicit list**: `rusticide-alloc`, `rusticide-core`,
  `rusticide-schema`. A new crate must be added to it by hand.
- `crates/rusticide-fuzz` is `exclude`d and is its own workspace: libfuzzer
  needs nightly and its generated entry point cannot satisfy
  `unsafe_code = "deny"`. Do not turn `members` back into a `crates/*` glob.
- `crates/rusticide`, `-ggml`, `-tokenizer`, `-sampling`, `-quantize` are
  pre-reset leftovers: on disk, not members, never built. They get rewired
  once the core types they import exist.

## Toolchain

- Stable channel pinned in `rust-toolchain.toml` (read it; do not restate the
  number elsewhere). Edition 2024.
- MSRV **1.88** (`Cargo.toml` `rust-version`, `.clippy.toml` `msrv`). Do not
  use APIs newer than 1.88. README and plan saying 1.92 are stale.
- Nightly only where a task pins it: fuzz, miri, and the coverage task
  (tarpaulin doctests need `nightly-2026-08-03`).

## Commands

`mise.toml` holds every task (`.mise.toml` only carries `commit-config`).
Prefer these over bare cargo.

```sh
mise run setup          # first run in a fresh clone: tools + git hooks
mise run build          # b
mise run fmt            # WRITES: rustfmt -> cargo-sort -> taplo, in that order
mise run check          # ck  cargo check --all-targets --all-features
mise run cargo:clippy   # cl  pedantic + nursery + cargo, -D warnings
mise run lint           # l   fmt:check + clippy
mise run test           # t   cargo-tarpaulin, 100% coverage gate, includes doctests
mise run ci:quick       # ciq the same gate CI runs (slow: tarpaulin)
mise run pr:ready       # format, then the full check
mise run security       # sec cargo audit + cargo deny
mise run markdownlint   # md  rumdl check (mdf = rumdl fix); never markdownlint-cli
mise run commit:config  # cc  regenerate committed.toml after editing commit-types.toml
```

- One test: `cargo test -p <crate> <test_name_substring>`.
- Full lint picture:
  `cargo clippy --keep-going --workspace --all-targets --all-features -- -D warnings`;
  without `--keep-going` the first failing crate hides the rest.
- When the coverage gate fails, `mise run test` prints an `Uncovered Lines`
  list per file; start there.
- Every task runs `cargo fmt` before committing; `cargo fmt --check` is part
  of verification.

## Code rules

- **Strict TDD**: failing test -> see it red -> minimal code -> green ->
  refactor. Never write production code before its test.
- **`unsafe_code = "deny"`** workspace-wide. Only `rusticide-alloc` opts out,
  under `docs/UNSAFE.md`: `// SAFETY:` on every block
  (`undocumented_unsafe_blocks = "deny"`), safe public API, Miri in CI.
- **No `unwrap()`/`expect()` in production code** (a panic is a DoS); allowed
  in `#[cfg(test)]` only.
- Warnings are errors (`-D warnings` in `.cargo/config.toml` and
  `[workspace.lints]`).
- `redundant_pub_crate` is allowed on purpose: it contradicts rustc's
  `unreachable_pub`. Do not "fix" one by violating the other.
- **Files under 300 lines**, tests included. Split by responsibility and
  re-export from the crate root.
- **ASCII only** in source and docs: no em dash, arrow, multiplication sign
  or ellipsis characters; write `--`, `->`, `x`, `...` instead.
- Every new `.rs` file starts with this exact three-line SPDX header:

  ```text
  // SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
  //
  // SPDX-License-Identifier: MIT OR Apache-2.0
  ```

- Errors: `thiserror` in libraries, `anyhow` in binaries.
- No comments unless explaining a non-obvious invariant.
- Principles: KISS, DRY, YAGNI, SOLID.

## Coverage (100% gate)

- Tests for code that already exists pass on the first run, so prove each one:
  break the line it covers, watch the test fail, restore the line.
- A line whose removal no test can detect is dead: delete it, don't test it.
- `#[allow(dead_code)] const fn assert_*` exhaustiveness guards still count as
  lines; cover them with a test that calls the guard for every variant.
- A branch the public API cannot reach is reached from the in-module
  `#[cfg(test)]` block, where private fields are visible.
- Assert the exact `ParseError` variant, never just `is_err()`: the schema
  parsers' round-trip check rejects most bad input too, so it hides which
  check fired.
- New schema parsers return through `error::ensure_round_trip` rather than
  re-implementing the check.

## Git

- **Never commit, push, or open a PR without explicit human approval.** Never
  write PR descriptions, commit messages, or reviewer responses on your own.
  `.claude/settings.json` blocks `git push` and `gh pr create`; publishing
  stays a human action.
- **No attribution trailers of any kind**: not `Co-authored-by:`, not
  `Assisted-by:`.
- Commits are GPG-signed (`git commit -S`) and follow conventional commits,
  enforced by `committed` (`committed.toml`, generated from
  `commit-types.toml`).
- Feature work goes in a worktree under `worktrees/` (git-ignored):
  `git worktree add worktrees/<branch> <branch>`.
- Reintegrate by **rebase only** so `main` stays linear; never `--no-ff`:

  ```sh
  git rebase main <branch>
  git switch main && git merge --ff-only <branch>
  ```

## Licensing

- Code and docs carry `MIT OR Apache-2.0` SPDX headers (inline, or via the
  `REUSE.toml` aggregate annotation). `mise run reuse` enforces it in CI.
- Background and the July 2026 rationale:
  `docs/history/2026-07-agent-session-notes.md`.
