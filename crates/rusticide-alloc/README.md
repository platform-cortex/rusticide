<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-alloc

Aligned buffer allocation for Rusticide tensors.

This is the first crate in the workspace permitted to use `unsafe`; the
workspace otherwise sets `unsafe_code = "deny"`. Today's implementation is
entirely safe (`Vec::try_reserve_exact` plus `resize` gives a zeroed,
correctly aligned allocation with a fallible path), but safe Rust will not
stay sufficient: zero-filling every allocation costs real time on a
framework that immediately overwrites most of them, and over-aligned
allocation for SIMD loads needs the same `MaybeUninit` plus raw-allocator
approach. This crate exists so that boundary, its lint table, and its
`docs/UNSAFE.md` entry are in place before that path lands.

See [`docs/UNSAFE.md`](../../docs/UNSAFE.md) for the workspace's full
`unsafe` policy and the index of every crate permitted to use it.
