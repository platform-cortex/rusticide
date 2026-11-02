<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-simd

NEON kernels behind safe entry points for Rusticide.

This is the second crate in the workspace permitted to use `unsafe`; the
workspace otherwise sets `unsafe_code = "deny"`. The kernels themselves
are safe Rust: inside a function marked
`#[target_feature(enable = "neon,dotprod")]` the NEON intrinsics are safe
to call, and `bytemuck` (feature `aarch64_simd`) converts byte arrays to
vector types without `unsafe`. What safe Rust cannot express is the call
from ordinary code into such a function: the compiler requires an
`unsafe` block for it even when the build enables the features globally.
That call, one per kernel, is the only `unsafe` here.

Soundness is a build-time guarantee, not run-time detection: the `neon`
module compiles only under
`cfg(all(target_arch = "aarch64", target_feature = "dotprod"))`, so every
CPU the binary targets has the features. `aarch64-apple-darwin` enables
`dotprod` by default; the Linux aarch64 targets do not, and there the
module (and `AVAILABLE`) is absent. Every public entry point is a safe
function over fixed-size arrays.

Every kernel follows the template of `neon::dot::dot_i8x32`: a safe
`#[inline]` wrapper whose single `unsafe` block calls a private
`#[target_feature(enable = "neon,dotprod")]` function, with a
`// SAFETY:` comment citing the build-time `cfg`. Under `cfg(miri)` the
wrapper calls a scalar body with the same integer result instead, because
Miri cannot execute the intrinsics; CI interprets the tests under Miri
for `aarch64-apple-darwin`.

See [`docs/UNSAFE.md`](../../docs/UNSAFE.md) for the workspace's full
`unsafe` policy and the index of every crate permitted to use it.
