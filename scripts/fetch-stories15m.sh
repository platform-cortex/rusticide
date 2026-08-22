#!/usr/bin/env bash
# Fetch Karpathy's stories15M and convert it to a single-file F32 GGUF.
# Output: models/stories15M-f32.gguf (git-ignored). Requires llama.cpp's
# convert tooling on PATH or at $LLAMA_CPP (env var pointing at a checkout).
set -euo pipefail

OUT_DIR="${OUT_DIR:-models}"
OUT="${OUT_DIR}/stories15M-f32.gguf"
mkdir -p "${OUT_DIR}"

if [[ -f "${OUT}" ]]; then
  echo "already present: ${OUT}"
  exit 0
fi

# stories15M is distributed as a HuggingFace repo with safetensors + tokenizer.
# Adjust REPO if you mirror it elsewhere.
REPO="${STORIES15M_REPO:-Xenova/llama2.c-stories15M}"
SRC_DIR="${OUT_DIR}/stories15M-src"
mkdir -p "${SRC_DIR}"

echo "downloading ${REPO} ..."
# Use the huggingface CLI if available; otherwise document manual download.
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${REPO}" --local-dir "${SRC_DIR}"
else
  echo "huggingface-cli not found. Manually place the model files in ${SRC_DIR}" >&2
  exit 1
fi

CONVERT="${LLAMA_CPP:-}/convert_hf_to_gguf.py"
if [[ ! -f "${CONVERT}" ]]; then
  echo "Set LLAMA_CPP to a llama.cpp checkout containing convert_hf_to_gguf.py" >&2
  exit 1
fi

echo "converting to F32 GGUF ..."
python3 "${CONVERT}" "${SRC_DIR}" --outtype f32 --outfile "${OUT}"
echo "wrote ${OUT}"
