<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-ggml

Compute graph over `rusticide-core` tensors, matching the llama.cpp GGML API.

A `GraphBuilder` declares input placeholders and operator nodes, inferring and
validating every node's shape and dtype as it is added. The built `Graph`
executes its nodes in id order, which is a topological order because a node
can only reference earlier nodes. This crate currently provides the structural
operators (copy, reshape, permute, cont, repeat, get_rows) and the add and
matmul operators.

## License

MIT OR Apache-2.0
