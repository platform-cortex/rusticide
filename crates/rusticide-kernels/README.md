<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-kernels

The kernel facade shared by the rusticide crates. Slice-level `f32` kernels:
dot product, RMS norm, softmax, rotary embeddings, SiLU and element-wise
multiply/add. Quantized kernels: activation quantization (`Q8Input`, the ggml
`Q8_0` and `Q8_K` schemes) and, per block format, the integer row kernels
(`QuantKernel`, looked up by GGML type id). Every call routes through one
`Adapter`, chosen once per process.

Kernels never panic on mismatched slice lengths: they iterate with `zip`, so
the shorter slice bounds the work. Callers validate lengths.

## License

MIT OR Apache-2.0
