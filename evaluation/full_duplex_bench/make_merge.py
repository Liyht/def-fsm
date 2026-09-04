#!/usr/bin/env python3
"""Merge user input and model output into one stereo wav for human inspection.

For every sample dir under <base-dir>/<task>/*, pair the streamed user audio with
the model's time-synchronous response and write a stereo file on one shared
timeline: left = user input, right = model output. Both start at t=0 (the FSM feeds
the input from the recording's start), so no offset is needed; the user audio is
resampled to the output rate before interleaving.

  input.wav       + output.wav       -> merge.wav
  clean_input.wav + clean_output.wav -> clean_merge.wav

Standalone and idempotent: reads only existing wavs (never re-runs inference), so
it backfills a partial results tree and can rerun freely. Skips a merge that already
exists unless --overwrite.

Usage:
  python make_merge.py --base-dir <RESULTS_ROOT> [--task all] [--overwrite]
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

V15_TASKS = ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]
V1_TASKS = [
    "candor_pause_handling",
    "synthetic_pause_handling",
    "candor_turn_taking",
    "icc_backchannel",
    "synthetic_user_interruption",
]
# (user input, model output) -> merged stereo name.
MERGE_PAIRS_V15 = [
    ("input.wav", "output.wav", "merge.wav"),
    ("clean_input.wav", "clean_output.wav", "clean_merge.wav"),
]
# v1 has no clean pair; the output is truncated to the input duration, so both
# share the same timeline and interleave with no offset.
MERGE_PAIRS_V1 = [
    ("input.wav", "output.wav", "merge.wav"),
]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def make_one(in_path: Path, out_path: Path, merge_path: Path) -> None:
    """Write a stereo merge (L=user, R=model) at the model output's sample rate."""
    user, user_sr = _load_mono(in_path)
    model, model_sr = _load_mono(out_path)
    if user_sr != model_sr:
        user = librosa.resample(user, orig_sr=user_sr, target_sr=model_sr)
    n = max(len(user), len(model))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[: len(user), 0] = user
    stereo[: len(model), 1] = model
    sf.write(str(merge_path), stereo, model_sr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True, help="Per-model results root (contains <task>/ subdirs).")
    ap.add_argument("--protocol", default="v15", choices=["v15", "v1"],
                    help="v15: input+clean pairs over v1.5 tasks; v1: input-only pair over v1.0 tasks.")
    ap.add_argument("--task", default="all", help="all | a single task name for the chosen protocol.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite merges that already exist.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    all_tasks = V1_TASKS if args.protocol == "v1" else V15_TASKS
    merge_pairs = MERGE_PAIRS_V1 if args.protocol == "v1" else MERGE_PAIRS_V15
    tasks = all_tasks if args.task == "all" else [args.task]

    made = skipped = missing = 0
    for task in tasks:
        for sample_dir in sorted(glob(str(base_dir / task / "*"))):
            sample_dir = Path(sample_dir)
            if not sample_dir.is_dir():
                continue
            for in_name, out_name, merge_name in merge_pairs:
                in_path, out_path = sample_dir / in_name, sample_dir / out_name
                merge_path = sample_dir / merge_name
                if not (in_path.exists() and out_path.exists()):
                    missing += 1
                    continue
                if merge_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                make_one(in_path, out_path, merge_path)
                made += 1
    print(f"[DONE] wrote {made} merges, skipped {skipped} existing, {missing} pairs incomplete.")


if __name__ == "__main__":
    main()
