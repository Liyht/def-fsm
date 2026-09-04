#!/bin/bash
# Train every configuration in configs/experiments/, in sequence.
#
#   bash scripts/train_all.sh            # all runs
#   bash scripts/train_all.sh wo_sac     # one run by name
#
# Each run writes to the output_dir named in its config, under CHECKPOINT_ROOT.
#
#   ours_faster     SAC loss, token-matched mixture, Faster-Whisper perception
#   ours_simul      the same with the word-level SimulStreaming tapes
#   wo_sac          plain cross-entropy instead of SAC
#   hh_only         Human-Human tapes only
#   ha_only         Human-Agent tapes only
#   nfsm_synthetic  the synthetic NFSM baseline
#   scaled          proportionate upsampling mixture
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=(ours_faster ours_simul wo_sac hh_only ha_only nfsm_synthetic scaled)

if [ $# -gt 0 ]; then
    RUNS=("$@")
fi

for run in "${RUNS[@]}"; do
    echo ""
    echo "################ $run ################"
    if [ "$run" = "ours_faster" ]; then
        # The main setting is the base config itself, with no override file.
        bash "$REPO_ROOT/scripts/train.sh" "$REPO_ROOT/configs/base.yaml"
    else
        override="$REPO_ROOT/configs/experiments/${run}.yaml"
        if [ ! -f "$override" ]; then
            echo "No such experiment: $run ($override)"; exit 1
        fi
        bash "$REPO_ROOT/scripts/train.sh" "$REPO_ROOT/configs/base.yaml" "$override"
    fi
done
