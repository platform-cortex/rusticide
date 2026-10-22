<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-dispatch

The operator dispatcher for rusticide, modelled on PyTorch's dispatcher.

This crate holds the boxed calling convention: `IValue`, a dynamically typed
value, and `Stack`, the argument and return stack that kernels read from and
write to. Kernel registration and dispatch are built on top of these.
