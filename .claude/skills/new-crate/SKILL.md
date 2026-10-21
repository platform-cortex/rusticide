---
name: new-crate
description: Scaffold a new crate in the rusticide virtual workspace with the standard lints, metadata, SPDX headers, and a TDD test stub. Use when adding a new library or sub-crate under crates/.
disable-model-invocation: true
---

# New Crate

Scaffold a new crate under `crates/<name>` that inherits the workspace lints and
metadata. Crate names in this repo are prefixed `rusticide-` (e.g.
`rusticide-dispatch`).

Usage: `/new-crate rusticide-<name>` (pass the full crate name as `$ARGUMENTS`).

## Steps

1. Confirm the crate name from `$ARGUMENTS`. It must start with `rusticide-`
   and must not already exist under `crates/` (some pre-reset leftovers such
   as `rusticide-ggml` are on disk but not workspace members; ask the user
   before reusing one). If the name is missing, ask the user.

2. Inspect a sibling first: read `crates/rusticide-schema/Cargo.toml` (a
   small, current member) to copy the `[package]` / `[lints]` shape. Do not
   invent fields; mirror what siblings use (`*.workspace = true`
   inheritance).

3. Create `crates/<name>/Cargo.toml`:

   ```toml
   # SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
   #
   # SPDX-License-Identifier: MIT OR Apache-2.0

   [package]
   authors.workspace = true
   description = "<one line>"
   edition.workspace = true
   license.workspace = true
   name = "<name>"
   readme = "README.md"
   repository.workspace = true
   rust-version.workspace = true
   version.workspace = true

   [dependencies]
   # add via `cargo add -p <name> <dep>`; prefer `<dep>.workspace = true`

   [lints]
   workspace = true
   ```

4. Create `crates/<name>/README.md` with the same SPDX header (as an HTML
   comment) and a one-paragraph purpose.

5. Create `crates/<name>/src/lib.rs` with the SPDX header and a **failing
   test first** (strict TDD):

   ```rust
   // SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
   //
   // SPDX-License-Identifier: MIT OR Apache-2.0

   //! <Crate purpose in one line.>

   #[cfg(test)]
   mod tests {
       // Write the first failing test for the crate's real behavior here,
       // then implement the minimal code above to make it pass.
   }
   ```

6. Add `"crates/<name>"` to the explicit `members` list in the root
   `Cargo.toml` (keep it sorted). Never replace the list with a `crates/*`
   glob: `rusticide-fuzz` must stay excluded.

7. Verify it builds, the lints apply, and formatting/ordering are clean:

   ```bash
   cargo build -p <name>
   cargo clippy -p <name> --all-targets -- -D warnings
   mise run fmt
   ```

Do not commit the new crate; leave that to the user.
