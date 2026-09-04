#!/bin/bash
# Build the training corpora: Human-Human tapes from spoken dialogue and
# Human-Agent tapes from text dialogue.
#
#   bash scripts/prepare_data.sh [stage ...]
#
# With no arguments every stage runs in order. Stages are independent enough to
# be re-run individually once their inputs exist:
#
#   hh-asr         transcribe the user channel of Switchboard and Fisher with
#                  the perception module the FSM deploys
#   hh-refine      refine the Fisher reference transcripts
#   hh-tape        turn-taking event classification and event-guided tape
#                  serialization
#   ha-clean       rule-based filtering of ShareGPT
#   ha-rewrite     LLM stylistic rewriting into a spoken register
#   ha-tape        serialize the rewritten dialogues into tapes
#   nfsm-generate  generate the synthetic NFSM baseline dialogues
#   nfsm-tape      serialize them, plus the MiU/UiM test sets
#   statistics     token-type ratios and split sizes
#
# ha-rewrite loads the rewriting model with vLLM in-process, so it needs GPUs but
# no server. nfsm-generate needs OPENAI_API_KEY in .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ALL_STAGES=(hh-asr hh-refine hh-tape ha-clean ha-rewrite ha-tape nfsm-generate nfsm-tape statistics)
STAGES=("${@:-}")
[ -z "${STAGES[0]:-}" ] && STAGES=("${ALL_STAGES[@]}")

run_stage() {
    echo ""
    echo "################ $1 ################"
    case "$1" in
        hh-asr)
            # One worker set per GPU; MPS keeps the many small Whisper jobs efficient.
            for dataset in switchboard fisher; do
                python data_transformation/asr_multiprocess_simulation.py \
                    --rank 0 --world-size 1 \
                    --dataset-type "$dataset" --asr-method faster \
                    --postfix _setting --workers-per-gpu 4
            done
            ;;
        hh-refine)
            python data_transformation/fisher_refiner.py
            ;;
        hh-tape)
            python data_transformation/audio_data_to_tape.py
            ;;
        ha-clean)
            python data_transformation/shareGPT_cleaning.py
            ;;
        ha-rewrite)
            python data_transformation/convert_to_spoken.py \
                --model Qwen/Qwen3-32B --world_size 1 --rank 0 --tensor_parallel_size 4 --continue_task
            ;;
        ha-tape)
            python data_transformation/text_dataset_to_tape.py --task shareGPT
            ;;
        nfsm-generate)
            python data_transformation/fsm_data_generation.py
            ;;
        nfsm-tape)
            python data_transformation/text_dataset_to_tape.py --task FSM_synthetic_train
            python data_transformation/text_dataset_to_tape.py --task FSM_synthetic_test
            ;;
        statistics)
            python data_transformation/statistics/statistics_dataset.py
            python data_transformation/statistics/statistics_token_ratio.py
            ;;
        *)
            echo "Unknown stage: $1"; echo "Known stages: ${ALL_STAGES[*]}"; exit 1
            ;;
    esac
}

for stage in "${STAGES[@]}"; do
    run_stage "$stage"
done

echo ""
echo "Data preparation finished."
