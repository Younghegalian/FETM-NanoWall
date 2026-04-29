#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/.venv/bin/activate"

export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export HF_HOME="${ROOT_DIR}/models/huggingface"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}"

echo "Activated FETM environment at ${ROOT_DIR}/.venv"
echo "Use: sem-to-domain --config data/configs/<sample>.json --image data/raw/<sample>/sem.tif --out runs/<sample>"
