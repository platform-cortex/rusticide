---
name: quality-gate
description: Run the rusticide pre-PR quality gate (format, sort, TOML, version/lock drift, clippy, tarpaulin coverage with doctests). Use before requesting review, before any commit, or when the user asks to verify the workspace is clean.
---

<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Quality Gate

Run the pre-PR gate for this repo and report results honestly.

## Steps

1. Run the quick gate:

   ```bash
   mise run ci:quick
   ```

   This runs, in order: `cargo:fmt:check`, `cargo:sort:check`,
   `taplo:check`, `version:check`, `cargo:lock:check`, `cargo:clippy`
   (`-D warnings`), and `test` (cargo-tarpaulin with the 100% coverage gate,
   doctests included). It is the same gate CI runs. Tarpaulin instruments a
   full build, so expect several minutes.

2. For a fast inner loop before the full gate, use `mise run lint`
   (fmt:check + clippy) and `cargo test -p <crate>`.

3. If the user asked for the full gate, run `mise run ci` instead
   (`ci:quick` plus `comply:format:check`). For supply chain, add
   `mise run security`.

## Reporting

- State the actual pass/fail for each stage. Never claim success without seeing
  the passing output.
- On failure, show the relevant output and stop; do NOT auto-fix unless asked.
  For formatting failures, `mise run fmt` fixes them. For a coverage shortfall
  or clippy/test failures, follow strict TDD (add the missing test, keep tests
  green).
- Do not commit, push, or open a PR as part of this skill.
