<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-kernels

Slice-level `f32` kernels shared by the rusticide crates: dot product and
parallel matrix-vector product, RMS norm, softmax, rotary embeddings (adjacent
pairs), SiLU and element-wise multiply/add.

Kernels never panic on mismatched slice lengths: they iterate with `zip`, so
the shorter slice bounds the work. Callers validate lengths.

## License

MIT OR Apache-2.0
