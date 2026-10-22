<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-quantize

Model quantization formats.

Implements the Q4_0, Q4_K and Q6_K block formats on top of `rusticide-core`
tensors. Q4_0 quantizes and dequantizes; Q4_K and Q6_K dequantize, for
reduced memory usage and faster CPU inference.

## License

MIT OR Apache-2.0
