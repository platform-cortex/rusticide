<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-io

GGUF model file reader.

Parses untrusted GGUF input through a bounds-checked little-endian byte
cursor. Every read is checked, untrusted lengths saturate instead of
overflowing, and nested metadata arrays are depth-limited.

`GgufFile::from_bytes` parses the header, metadata and tensor infos of a GGUF
v3 file. It rejects an invalid `general.alignment`, duplicate tensor names,
more than four dimensions and element counts that overflow. Metadata arrays
expand to roughly 24-32 bytes per element in memory.

`GgufFile::tensor` materializes tensors, dequantizing Q4_0, Q4_K and Q6_K to
F32. Only GGUF version 3 is accepted (llama.cpp also reads v2).

## License

MIT OR Apache-2.0
