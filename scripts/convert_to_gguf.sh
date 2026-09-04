#!/bin/bash
# Export a fine-tuned Hugging Face checkpoint to GGUF for the llama.cpp-based
# cognitive module used at inference time.
#
#   bash scripts/convert_to_gguf.sh <checkpoint-dir> [outtype]
#
# `checkpoint-dir` is one of the best_loss_* / best_f1_* directories a training
# run writes. `outtype` is f16 (default), bf16 or q8_0. The result is written to
# <checkpoint-dir>_gguf_<outtype>/ and is what demo/app_local.py expects.
#
# Requires a llama.cpp checkout; set LLAMA_CPP_PATH in configs/paths.yaml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint-dir> [f16|bf16|q8_0]"
    exit 1
fi

INPUT_DIR=$1
OUTTYPE=${2:-f16}
OUTPUT_DIR="${INPUT_DIR%/}_gguf_${OUTTYPE}"

LLAMA_CPP=$(python -c "from def_fsm.paths import require; print(require('LLAMA_CPP_PATH'))")

mkdir -p "$OUTPUT_DIR"
echo "Converting $INPUT_DIR -> $OUTPUT_DIR ($OUTTYPE)"
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    --outtype "$OUTTYPE" \
    --outfile "$OUTPUT_DIR" \
    "$INPUT_DIR"
