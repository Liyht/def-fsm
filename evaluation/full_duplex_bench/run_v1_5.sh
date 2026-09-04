#!/usr/bin/env bash
# Full-Duplex-Bench v1.5 evaluation for a DEF-FSM checkpoint.
#
#   bash evaluation/full_duplex_bench/run_v1_5.sh <model.gguf> [task]
#
# Stages, in order:
#   0  stage the read-only dataset into a per-model tree as symlinks
#   1  FSM inference       -> output.wav, clean_output.wav
#   2  time-aligned ASR    -> output.json, clean_output.json
#   3  behavior judge, before/after comparison and timing
#
# v1.5 pairs an overlapping condition with a clean reference, so inference runs
# with the default --protocol v15: after the input drains, dithered silence keeps
# feeding until the FSM's response finishes, and the response is recorded on a
# timeline sharing t=0 with the input.
#
# Two environments are involved; see README.md in this directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LLM_PATH="${1:?usage: run_v1_5.sh <model.gguf> [task]}"
TASK="${2:-all}"

# Resolved from configs/paths.yaml (or the environment).
FDB_PATH="$(python -c 'from def_fsm.paths import require; print(require("FDB_PATH"))')"
FDB_DATA="$(python -c 'from def_fsm.paths import require; print(require("FDB_DATA"))')"

BENCH="$FDB_PATH/v1_v1.5"
DATA_DIR="$FDB_DATA/v1.5"
V15_TASKS=(user_interruption user_backchannel talking_to_other background_speech)
[ "$TASK" = "all" ] && TASKS=("${V15_TASKS[@]}") || TASKS=("$TASK")

MODEL_TAG="$(basename "$(dirname "$LLM_PATH")")"
RESULTS_ROOT="${FDB_RESULTS_ROOT:-$REPO_ROOT/outputs/fdb_v1_5}/$MODEL_TAG"

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
        mkdir -p "$RESULTS_ROOT/$t/$id"
        for name in input.wav clean_input.wav input.json clean_input.json metadata.json; do
            [ -f "$src$name" ] && ln -sf "$(realpath "$src$name")" "$RESULTS_ROOT/$t/$id/$name"
        done
    done
done

# 1) FSM inference, in this repository's environment.
"${FSM_ENV[@]}" python evaluation/full_duplex_bench/inference.py \
    --base-dir "$RESULTS_ROOT" --task "$TASK" \
    --llm-model-path "$LLM_PATH" --devices $FSM_DEVICES

for t in "${TASKS[@]}"; do
    ROOT="$RESULTS_ROOT/$t"
    echo ""
    echo "################ $t ################"

    # 2) Time-aligned ASR. input.json and clean_input.json come from the dataset, so
    #    only the two response wavs need transcribing.
    for wav in output.wav clean_output.wav; do
        "$BENCH_PY" "$BENCH/get_transcript/asr.py" --root_dir "$ROOT" --task default --audio_name "$wav"
    done

    # 3) Metrics. evaluate.py must run from the benchmark's evaluation/ directory:
    #    it reads ./instruction and writes its logs to the working directory.
    (
        cd "$BENCH/evaluation"
        "$BENCH_PY" evaluate.py --task behavior --root_dir "$ROOT"
        "$BENCH_PY" evaluate.py --task general_before_after --root_dir "$ROOT"
        "$BENCH_PY" get_timing.py --root_dir "$ROOT"
    )
done

echo ""
echo "Done: $RESULTS_ROOT"
