<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Crates permitted to use `unsafe`

`unsafe_code = "deny"` is the workspace default. Every crate listed here
opts out with its own `[lints.rust]` table, and every one must satisfy
the policy in `docs/plan/2026-08-21-rusticide-framework-design.md`:
isolation, a safe public API, a `// SAFETY:` comment per block, a README
justification, an entry in this file, and Miri in CI.

Miri in CI is a prerequisite for landing an `unsafe` block here, not
infrastructure that already exists: it is unavailable on the toolchain
this workspace currently pins, and no CI task runs it yet. Wiring it up
is part of the work of the change that introduces the first `unsafe`
block, not a follow-up.

If this list stops being short, the policy has failed.

## `rusticide-alloc`

**Capability:** allocating tensor buffers.

**Why safe Rust is not enough:** zero-filling every allocation costs real
time on a framework that immediately overwrites most of them. Avoiding
that needs `MaybeUninit` and the raw allocator API. Over-aligned
allocation for SIMD loads has the same requirement.

**Current state:** the implementation is entirely safe today
(`Vec::try_reserve_exact` plus `resize`). The crate and its lint table
exist so the boundary is in place before the uninitialized path lands.

**How it could be deleted:** if a future Rust release offers safe
uninitialized byte buffers with an allocation-failure result, this crate
folds back into `rusticide-core`.
