<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-transformer

Decoder-only transformer building blocks shared by the rusticide model
families: the `Decoder` forward pass over a KV cache, the decoder
configuration and weights, per-token batch helpers, the layout transpose,
`RoPE` styles, RMS normalization, grouped-query attention and the gated
feed-forward. A family describes itself as a `Family`, from which
`DecoderConfig::from_source` reads its hyperparameters and
`DecoderWeights::from_source` its weights by tensor name.

## License

MIT OR Apache-2.0
