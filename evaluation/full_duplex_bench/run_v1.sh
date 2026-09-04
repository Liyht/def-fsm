#!/usr/bin/env bash
# Full-Duplex-Bench v1.0 evaluation for a DEF-FSM checkpoint.
#
#   bash evaluation/full_duplex_bench/run_v1.sh <model.gguf> [subset]
#
# Stages, in order:
#   0  stage the read-only dataset into a per-model tree as symlinks
#   1  FSM inference           -> output.wav
#   2  time-aligned ASR        -> output.json
#   3  per-subset metric       -> printed by the benchmark's evaluate.py
#
# v1.0 has no clean/overlap pair and the response window lives inside input.wav
# (its trailing silence), so inference runs with --protocol v1: feed only
# input.wav, truncate the output to its duration. The v1.5 behavior, prosody and
# timing stages do not apply here.
#
# Two environments are involved. Stage 1 runs in this repository's environment
# (the FSM needs llama.cpp, Kokoro and Faster-Whisper); stages 2-3 run in the
# benchmark's own environment, which brings NeMo parakeet, silero-vad and utmosv2.
# See README.md in this directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LLM_PATH="${1:?usage: run_v1.sh <model.gguf> [subset]}"
TASK="${2:-all}"

# Resolved from configs/paths.yaml (or the environment).
FDB_PATH="$(python -c 'from def_fsm.paths import require; print(require("FDB_PATH"))')"
FDB_DATA="$(python -c 'from def_fsm.paths import require; print(require("FDB_DATA"))')"

BENCH="$FDB_PATH/v1_v1.5"
DATA_DIR="$FDB_DATA/v1.0"
V1_TASKS=(candor_pause_handling synthetic_pause_handling candor_turn_taking icc_backchannel synthetic_user_interruption)
[ "$TASK" = "all" ] && TASKS=("${V1_TASKS[@]}") || TASKS=("$TASK")

MODEL_TAG="$(basename "$(dirname "$LLM_PATH")")"
RESULTS_ROOT="${FDB_RESULTS_ROOT:-$REPO_ROOT/outputs/fdb_v1}/$MODEL_TAG"

# One device each for the LLM, the ASR and the TTS worker.
FSM_DEVICES="${FSM_DEVICES:-cuda:0 cuda:1 cuda:1}"
# Interpreter for the benchmark environment (stages 2-3).
BENCH_PY="${BENCH_PY:-python}"

# llama.cpp must find the CUDA runtime that ships with this environment's torch
# wheels, or a system libcudart loads first and torch aborts on an undefined symbol.
NV_LIBS="$(find "${VIRTUAL_ENV:-$REPO_ROOT/.venv}" -type d -path '*/nvidia/*/lib' 2>/dev/null | paste -sd: - || true)"
FSM_ENV=(env PYTHONUNBUFFERED=1 "LD_LIBRARY_PATH=${NV_LIBS}:${LD_LIBRARY_PATH:-}")

echo "Model   : $LLM_PATH"
echo "Results : $RESULTS_ROOT"

# 0) Stage the inputs. Only input-side files are linked, so the read-only dataset is
#    never written to and two models never clobber each other's outputs.
for t in "${TASKS[@]}"; do
    for src in "$DATA_DIR/$t"/*/; do
        [ -d "$src" ] || continue
        id="$(basename "$src")"
        [ "$id" = "__MACOSX" ] && continue
        [ -f "$src/input.wav" ] || continue
        mkdir -p "$RESULTS_ROOT/$t/$id"
        for name in input.wav pause.json turn_taking.json interrupt.json transcription.json context.wav interrupt.wav; do
            [ -f "$src$name" ] && ln -sf "$(realpath "$src$name")" "$RESULTS_ROOT/$t/$id/$name"
        done
    done
done

# 1) FSM inference, in this repository's environment.
"${FSM_ENV[@]}" python evaluation/full_duplex_bench/inference.py \
    --protocol v1 --base-dir "$RESULTS_ROOT" --task "$TASK" \
    --llm-model-path "$LLM_PATH" --devices $FSM_DEVICES

for t in "${TASKS[@]}"; do
    ROOT="$RESULTS_ROOT/$t"
    echo ""
    echo "################ $t ################"

    # 2) Time-aligned ASR of the response. The interruption subset crops the audio
    #    before the interruption (via interrupt.json) so the rated response excludes
    #    the preceding context.
    ASR_TASK="default"
    [ "$t" = "synthetic_user_interruption" ] && ASR_TASK="user_interruption"
    "$BENCH_PY" "$BENCH/get_transcript/asr.py" --root_dir "$ROOT" --task "$ASR_TASK" --audio_name output.wav

    # 3) Per-subset metric. evaluate.py must run from the benchmark's evaluation/
    #    directory: eval_backchannel.py reads ./icc_gt_distribution.json.
    case "$t" in
        candor_pause_handling|synthetic_pause_handling) EVAL_TASK=pause_handling ;;
        candor_turn_taking)                             EVAL_TASK=smooth_turn_taking ;;
        icc_backchannel)                                EVAL_TASK=backchannel ;;
        synthetic_user_interruption)                    EVAL_TASK=user_interruption ;;
    esac
    (cd "$BENCH/evaluation" && "$BENCH_PY" evaluate.py --task "$EVAL_TASK" --root_dir "$ROOT")
done

echo ""
echo "Done: $RESULTS_ROOT"
