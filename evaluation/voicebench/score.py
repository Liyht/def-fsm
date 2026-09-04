#!/usr/bin/env python3
"""Score VoiceBench generations with the benchmark's own evaluators.

Reads the per-dataset jsonl files written by inference.py and reports the four
subsets, plus their mean as the overall figure:

    mmsu       multiple-choice accuracy
    openbook   multiple-choice accuracy
    sdqa       open-ended answer match
    advbench   refusal rate

The evaluators come from the VoiceBench checkout and are used unmodified, with one
exception. Its QA evaluator averages two scores: PEDANT answer matching, and the
majority vote of a separate GPT judging pass. That pass writes a `score` field into
the jsonl and is run by the benchmark's own api_judge.py; without it the upstream
evaluator raises KeyError. We report the PEDANT score alone, so no judging pass and
no API key are needed. The subclass below is why the checkout itself needs no edit.

Run this with VoiceBench's own interpreter -- it needs that environment's
qa_metrics, and deliberately imports nothing from this repository:

    python evaluation/voicebench/score.py --result-dir <dir> --voicebench-path <checkout>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# The checkout has to be on the path before its evaluators can be imported, so this
# is resolved from the command line (or VOICEBENCH_PATH) ahead of the import below.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--voicebench-path", default=os.environ.get("VOICEBENCH_PATH"))
_known, _ = _ap.parse_known_args()
if not _known.voicebench_path:
    sys.exit("Set --voicebench-path or VOICEBENCH_PATH to the VoiceBench checkout.")
sys.path.insert(0, _known.voicebench_path)

from src.evaluator import evaluator_mapping  # noqa: E402
from src.evaluator.qa import QAEvaluator  # noqa: E402


class PedantOnlyQAEvaluator(QAEvaluator):
    """QA scoring without the GPT judging pass. See the module docstring."""

    def evaluate(self, data):
        scores = [
            self.pedant.evaluate([item["reference"].lower()], item["response"].lower(), item["prompt"].lower())
            for item in data
        ]
        return {"panda": float(np.mean(scores)) * 100}


# subset -> (jsonl stem, evaluator, the key its result dict reports)
TASKS = {
    "advbench": ("advbench-test", "harm", "refusal_rate"),
    "openbook": ("openbookqa-test", "mcq", "acc"),
    "mmsu": ("mmsu-all", "mcq", "acc"),
    "sdqa": ("sd-qa-usa", "qa", "panda"),
}


def build_evaluator(name: str):
    return PedantOnlyQAEvaluator() if name == "qa" else evaluator_mapping[name]()


def main() -> None:
    ap = argparse.ArgumentParser("Score VoiceBench generations")
    ap.add_argument("--result-dir", required=True, help="Directory holding the jsonl files from inference.py.")
    ap.add_argument("--voicebench-path", default=os.environ.get("VOICEBENCH_PATH"),
                    help="VoiceBench checkout, for its evaluators. Defaults to $VOICEBENCH_PATH.")
    ap.add_argument("--modality", default="audio", choices=["audio", "text"])
    ap.add_argument("--overwrite", action="store_true", help="Rescore even if score.json exists.")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_path = result_dir / "score.json"
    if out_path.exists() and not args.overwrite:
        logger.info(f"{out_path} exists: {json.load(open(out_path))}")
        return

    scores = {}
    for subset, (stem, evaluator_name, key) in TASKS.items():
        path = result_dir / f"{stem}-{args.modality}.jsonl"
        if not path.exists():
            logger.warning(f"Missing {path}, skipping {subset}")
            continue

        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        result = build_evaluator(evaluator_name).evaluate(data)
        logger.info(f"{subset}: {result}")

        # advbench reports a rate in [0, 1]; the others are already percentages.
        scores[subset] = result[key] * 100 if subset == "advbench" else result[key]

    missing = set(TASKS) - set(scores)
    if missing:
        logger.warning(f"Overall omits missing subsets: {sorted(missing)}")

    scores = {"overall": float(np.mean(list(scores.values()))), **scores}
    logger.info(scores)
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=4)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
