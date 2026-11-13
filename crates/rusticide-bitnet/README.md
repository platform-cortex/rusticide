<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-bitnet

The BitNet b1.58 architecture (`bitnet-b1.58`) for rusticide: half-split
(NEOX) RoPE, a squared-ReLU gated feed-forward and the sub-norms before
`attn_output` and `ffn_down`, over ternary (`i2_s`) weights and a token
embedding tied to the output. Configuration and weights are read by the
`rusticide-transformer` loader into its `Decoder`.

## License

MIT OR Apache-2.0
