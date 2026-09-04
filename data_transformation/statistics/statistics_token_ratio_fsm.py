"""
Token-type ratios for FSM_tape_dataset: the five state transition tokens against the response tokens.

Reuses compute_class_breakdown / format_breakdown from statistics_token_ratio.py. FSM_tape_dataset
belongs to the assistant category, has no ASR variants, and only has train/val splits.
"""

import os
import json
import yaml
import sys

from transformers import AutoTokenizer

from def_fsm.paths import PROJECT_ROOT, expand

# training/ and data_transformation/ are script directories rather than packages,
# so make their flat imports resolvable from any working directory.
sys.path.append(str(PROJECT_ROOT / "training"))
sys.path.append(str(PROJECT_ROOT / "data_transformation"))

from utils import load_config  # training/utils.py
from dataset import TapeDataset

from def_fsm.utils import USER_PREFIX
from statistics_token_ratio import compute_class_breakdown, format_breakdown


def main(
    config_path=expand("${PROJECT_ROOT}/configs/base.yaml"),
    ds_dir=expand("${DATA_ROOT}/FSM_tape_dataset"),
    output_json=expand("${DATA_ROOT}/FSM_tape_dataset/token_ratio_statistics.json"),
    splits=("train", "val"),  # this dataset has no test split
):
    # load_config resolves the ${...} path placeholders in the config.
    cfg = load_config(config_path)

    model_dir = cfg["model_dir"]
    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    new_tokens = cfg.get("new_tokens", [])
    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} special tokens to tokenizer.")

    # FSM is in the assistant category, so the interlocutor is the user (as in train.py)
    interlocutor_prefix = USER_PREFIX
    max_length = cfg["training"]["max_length"]

    results = {}
    for split in splits:
        json_file = os.path.join(ds_dir, f"{split}.json")
        print("\n" + "=" * 80)
        print(f"SPLIT: {split}  ({json_file})")
        print("=" * 80)

        if not os.path.exists(json_file):
            print(f"[ERROR] file not found: {json_file}")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        ds = TapeDataset(
            raw_data=raw_data,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt_path=None,  # the ratios are computed at step 1, where the prompt plays no part
            verbose=False,
            interlocutor_prefix=interlocutor_prefix,
            cache_dir=None,
        )

        breakdown = compute_class_breakdown(ds)
        print(format_breakdown(breakdown))
        results[split] = breakdown

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        serializable = {}
        for tag, bd in results.items():
            total = sum(bd.values())
            serializable[tag] = {
                "counts": dict(bd),
                "ratios": {k: (v / total if total > 0 else 0.0) for k, v in bd.items()},
                "total": total,
            }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
