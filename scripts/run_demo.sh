#!/bin/bash
# Run the full-duplex FSM interactively against the local microphone and speakers.
#
#   bash scripts/run_demo.sh <gguf-checkpoint> [extra args for demo/app_local.py]
#
# Extra arguments are passed straight through, e.g.
#   bash scripts/run_demo.sh model.gguf --asr-worker simul --devices cuda:0 cuda:1 cuda:2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <gguf-checkpoint> [extra args]"
    exit 1
fi

GGUF=$1
shift

cd "$REPO_ROOT"
python demo/app_local.py --llm-model-path "$GGUF" "$@"
