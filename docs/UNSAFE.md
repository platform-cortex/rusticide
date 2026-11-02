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

Miri in CI is a prerequisite for landing an `unsafe` block here. It is
unavailable on the stable toolchain this workspace pins, so `cicd.yml`'s
`miri` job runs it on a dated nightly (`nightly-2026-08-03`, as the
coverage task does) and gates every change. It was wired up with the
first `unsafe` block, in `rusticide-simd`.

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

## `rusticide-simd`

**Capability:** NEON kernels (`neon,dotprod`) behind safe entry points.

**Why safe Rust is not enough:** the kernels are safe Rust compiled with
`#[target_feature(enable = "neon,dotprod")]`, but calling such a function
from code without that attribute needs an `unsafe` block, even when the
build enables the features globally. Each kernel has exactly one: the
call from its safe wrapper. The vector wrappers alone would not do, as
they do not inline across that boundary, so the kernels live here whole.

**Soundness argument:** a token exists only after detection proved the
feature; every entry point takes one. The `neon` module compiles on
every `aarch64` target, and each public kernel takes a
`rusticide_simd::Dotprod` by value as its first argument. `Dotprod` is a
zero-sized type with a private field, so the only way to obtain one is
`Dotprod::detect()`, which returns `Some` exactly when std's
`is_aarch64_feature_detected!("dotprod")` reports the feature (NEON is
baseline on `aarch64`). Each `// SAFETY:` comment cites the token.

**Miri:** Miri cannot execute the intrinsics, so under `cfg(miri)` each
wrapper calls a scalar body with the same integer result, and the
`miri` CI job interprets the crate's tests for `aarch64-apple-darwin`.
Under `cfg(miri)` `Dotprod::detect()` returns `Some`, since the scalar
bodies need no hardware. The Miri gate therefore checks the scalar bodies
and the casts, not the intrinsic path; the soundness of each intrinsic
call rests on the detection argument above.
The `arm` CI job runs the full gate on Linux arm64 with no extra target
features: the module compiles in on every `aarch64` target, the kernel
adapter is selected at run time from `Dotprod::detect()`, and the job
fails unless the release binary reports `kernels: neon`.

**How it could be deleted:** if a future Rust release lets safe code
call a `#[target_feature]` function once a run-time check has proved the
features (a token-style proof in the language or std), the wrappers
become safe and the kernels can move into `rusticide-kernels`.
