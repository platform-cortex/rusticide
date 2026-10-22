<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-sampling

Token sampling strategies for LLM text generation.

Implements greedy, top-k, top-p (nucleus), temperature and repetition-penalty
sampling over `f32` logits held in a `rusticide-core` tensor of shape `[V]` or
`[1, V]`. A `Sampler` owns a seeded RNG, so the same seed reproduces the same
token sequence.

## License

MIT OR Apache-2.0
