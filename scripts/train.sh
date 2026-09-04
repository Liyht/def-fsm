#!/bin/bash
# Fine-tune a single DEF-FSM model.
#
#   bash scripts/train.sh <config> [override.yaml ...] [-- extra --key=value ...]
#
# Examples:
#   bash scripts/train.sh configs/base.yaml
#   bash scripts/train.sh configs/base.yaml configs/experiments/wo_sac.yaml
#   bash scripts/train.sh configs/base.yaml -- --training.max_train_steps=2
#
# GPU count, starting GPU and the rendezvous port come from the environment:
#   NUM_GPU=4 START_GPU=0 PORT=29502 bash scripts/train.sh configs/base.yaml
#
# The reference runs use 4 GPUs, per_device_train_batch_size 4 and
# gradient_accumulation_steps 16, for a total batch size of 256.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/_gpu.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [override.yaml ...] [-- extra args]"
    exit 1
fi

CONFIG=$1
shift

OVERRIDES=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    OVERRIDES+=("$1")
    shift
done
[ "${1:-}" = "--" ] && shift

NUM_GPU=${NUM_GPU:-4}
START_GPU=${START_GPU:-0}
PORT=${PORT:-29502}
CUDA_LIST=$(build_cuda_visible_devices "$NUM_GPU" "$START_GPU")

echo "========================================================"
echo "Config              : $CONFIG"
echo "Overrides           : ${OVERRIDES[*]:-none}"
echo "NUM_GPU             : $NUM_GPU"
echo "CUDA_VISIBLE_DEVICES: $CUDA_LIST"
echo "========================================================"

OVERRIDE_ARGS=()
[ ${#OVERRIDES[@]} -gt 0 ] && OVERRIDE_ARGS=(--override_path "${OVERRIDES[@]}")

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=$CUDA_LIST TOKENIZERS_PARALLELISM=false accelerate launch \
    --main_process_port "$PORT" \
    --num_processes="$NUM_GPU" \
    training/train.py \
    --config_path "$CONFIG" \
    "${OVERRIDE_ARGS[@]}" \
    "$@"
