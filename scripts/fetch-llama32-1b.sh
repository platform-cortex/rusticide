#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Fetch the Llama-3.2-1B-Instruct Q4_K_M GGUF for the parity test.
# Output: models/Llama-3.2-1B-Instruct-Q4_K_M.gguf (git-ignored).
# Override REPO / FILE to use a different mirror.
set -euo pipefail

OUT_DIR="${OUT_DIR:-models}"
FILE="${FILE:-Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
REPO="${REPO:-bartowski/Llama-3.2-1B-Instruct-GGUF}"
OUT="${OUT_DIR}/${FILE}"

mkdir -p "${OUT_DIR}"
if [[ -f "${OUT}" ]]; then
  echo "already present: ${OUT}"
  exit 0
fi

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
echo "downloading ${URL} ..."
if command -v curl >/dev/null 2>&1; then
  curl -fSL "${URL}" -o "${OUT}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUT}" "${URL}"
else
  echo "need curl or wget to download the model" >&2
  exit 1
fi
echo "wrote ${OUT}"
