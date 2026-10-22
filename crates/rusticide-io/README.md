<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide-io

GGUF model file reader.

Parses untrusted GGUF input through a bounds-checked little-endian byte
cursor. Every read is checked, untrusted lengths saturate instead of
overflowing, and nested metadata arrays are depth-limited.

## License

MIT OR Apache-2.0
