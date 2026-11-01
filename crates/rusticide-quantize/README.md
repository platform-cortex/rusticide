<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-quantize

Model quantization formats as data.

Block layouts and dequantizers for Q4_0, Q5_0, Q8_0, Q4_K and Q6_K, plus the
`f16` reader and synthetic blocks for tests. The integer kernels that compute
on these blocks live in `rusticide-kernels`.

## License

MIT OR Apache-2.0
