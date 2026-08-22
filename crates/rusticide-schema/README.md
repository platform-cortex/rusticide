<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-schema

A Rust port of [torchgen](https://github.com/pytorch/pytorch/tree/main/torchgen)'s
operator schema data model: the types that parse a PyTorch-style function
schema string (as found in `native_functions.yaml`) into a structured
representation, and print it back out again.

Every parse is round-trip asserted: `parse(s).to_string()` must equal `s`
exactly, byte for byte. A representation that cannot be printed back exactly
is a representation that lost information -- the assertion catches that at
parse time instead of letting it surface later as subtly wrong codegen.

This crate has no dependency on `rusticide-core`; it knows nothing about
tensors. It parses text into a data model, nothing more.
