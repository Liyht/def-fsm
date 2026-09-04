#!/bin/bash
# Report the per-type F1 over state transition tokens on the validation and test
# splits, together with the confusion matrices over token types. This is the
# turn-taking metric; the benchmarks alongside it measure the other capabilities.
#
#   bash evaluation/state_transition_f1/state_transition_f1.sh <output_dir_of_a_training_run>
#
# The run's own saved config.yaml is reused, so the evaluation sees exactly the
# tokenizer, prompts and data paths it was trained with. All splits are set to
# their full size regardless of the training mixture ratios.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/_gpu.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run-output-dir> [extra args]"
    exit 1
fi

OUTPUT_DIR=$1
shift
SAVED_CONFIG="$OUTPUT_DIR/config.yaml"
if [ ! -f "$SAVED_CONFIG" ]; then
    echo "No config.yaml in $OUTPUT_DIR - is this a training output directory?"
    exit 1
fi

NUM_GPU=${NUM_GPU:-4}
START_GPU=${START_GPU:-0}
CUDA_LIST=$(build_cuda_visible_devices "$NUM_GPU" "$START_GPU")
# Evaluation has no optimizer state, so it takes twice the training batch size.
PER_DEVICE_BATCH_SIZE=$((32 / NUM_GPU))

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=$CUDA_LIST TOKENIZERS_PARALLELISM=false accelerate launch \
    --num_processes="$NUM_GPU" \
    training/eval.py \
    --config_path "$SAVED_CONFIG" \
    --skip_existing \
    --skip_wandb \
    --dataset.assistant.dataset_ratios.switchboard=1.0 \
    --dataset.assistant.dataset_ratios.fisher=1.0 \
    --dataset.assistant.dataset_ratios.shareGPT=1.0 \
    --dataset.assistant.dataset_ratios.synthetic=0.0 \
    --training.per_device_train_batch_size=$PER_DEVICE_BATCH_SIZE \
    "$@" \
    2>&1 | tee "$OUTPUT_DIR/run_test.log"
