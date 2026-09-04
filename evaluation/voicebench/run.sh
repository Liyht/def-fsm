#!/usr/bin/env bash
# VoiceBench evaluation for a DEF-FSM checkpoint.
#
#   bash evaluation/voicebench/run.sh <model-dir-or-hf-id> [fsm|chat]
#
# Stage 1 generates responses in this repository's environment (the perception
# module and vLLM); stage 2 scores them with VoiceBench's own interpreter, which
# is where its evaluators' dependencies live. Set VOICEBENCH_PY to that
# interpreter. Both stages resume, so an interrupted run can simply be rerun.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?usage: run.sh <model-dir-or-hf-id> [fsm|chat]}"
MODE="${2:-fsm}"

VOICEBENCH_PATH="$(python -c 'from def_fsm.paths import require; print(require("VOICEBENCH_PATH"))')"
export VOICEBENCH_PATH

# Interpreter holding VoiceBench's dependencies (qa_metrics in particular).
VOICEBENCH_PY="${VOICEBENCH_PY:-$VOICEBENCH_PATH/.venv/bin/python}"

# Perception module. Use "simul" for a checkpoint trained against SimulStreaming.
ASR_TYPE="${ASR_TYPE:-faster}"
# GPUs for the vLLM generation pass.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

MODEL_TAG="$(basename "${MODEL%/}")_${ASR_TYPE}_${MODE}"
RESULT_DIR="${VOICEBENCH_RESULTS_ROOT:-$REPO_ROOT/outputs/voicebench}/$MODEL_TAG"

echo "Model   : $MODEL"
echo "Mode    : $MODE (ASR: $ASR_TYPE)"
echo "Results : $RESULT_DIR"

# 1) Transcribe every clip, then answer all transcripts in one vLLM batch.
python evaluation/voicebench/inference.py \
    --model "$MODEL" --mode "$MODE" --asr-type "$ASR_TYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --result-dir "$RESULT_DIR"

# 2) Score, using VoiceBench's evaluators.
"$VOICEBENCH_PY" evaluation/voicebench/score.py --result-dir "$RESULT_DIR"

echo ""
echo "Done: $RESULT_DIR/score.json"
