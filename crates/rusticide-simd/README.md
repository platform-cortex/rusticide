<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-simd

NEON and AVX2 kernels behind safe entry points for Rusticide.

This is the second crate in the workspace permitted to use `unsafe`; the
workspace otherwise sets `unsafe_code = "deny"`. The kernels themselves
are safe Rust: inside a function marked
`#[target_feature(enable = "neon,dotprod")]` the NEON intrinsics are safe
to call, and `bytemuck` (feature `aarch64_simd`) converts byte arrays to
vector types without `unsafe`. What safe Rust cannot express is the call
from ordinary code into such a function: the compiler requires an
`unsafe` block for it even when the build enables the features globally.
That call, one per kernel, is the only `unsafe` here.

Soundness rests on run-time detection, carried by a proof token: the
`neon` module compiles on every `aarch64` target, and each public entry
point takes a `Dotprod` by value as its first argument. `Dotprod` is a
zero-sized type that only `Dotprod::detect()` makes, and only when std's
`is_aarch64_feature_detected!("dotprod")` reports the feature (NEON is
baseline on `aarch64`; std caches the answer). Holding a token is the
proof, so no safe caller reaches a kernel on a CPU without the features.
Every public entry point is a safe function over fixed-size arrays.

Every kernel follows the template of `neon::dot::dot_i8x32`: a safe
`#[inline]` wrapper taking the token, whose single `unsafe` block calls a
private `#[target_feature(enable = "neon,dotprod")]` function, with a
`// SAFETY:` comment citing the token. Under `cfg(miri)` the wrapper
calls a scalar body with the same integer result instead, because Miri
cannot execute the intrinsics, and `Dotprod::detect()` returns `Some`;
CI interprets the tests under Miri for `aarch64-apple-darwin`.

On `x86_64` the `Avx2` token follows the same rules: only `Avx2::detect()`
makes one, when `is_x86_feature_detected!` reports both `avx2` and `fma`,
and it returns `Some` under `cfg(miri)`. FMA is part of the proof because
the portable `Q4_K` kernel fuses a multiply and add, and the AVX2 one must
round it the same way. The `avx2` module's kernels (`Q4_K`, `Q5_0`, `Q6_K`,
`Q8_0` and the `f32` dot products of `float`) enable `avx2,fma`. The
`float` ones compile the portable bodies of the crate's `float` module, so
`mul_add` is one instruction rather than a call to `fmaf`, with the same
bits for every non-NaN input. CI also interprets the tests under Miri for
`x86_64-unknown-linux-gnu`.

See [`docs/UNSAFE.md`](../../docs/UNSAFE.md) for the workspace's full
`unsafe` policy and the index of every crate permitted to use it.
