"""The statistics_dataset.py report for FSM_tape_dataset alone.

That dataset belongs to the assistant category, has no ASR variants and only train/val splits,
so it does not fit the loop over the config in statistics_dataset.py.
"""

import os
import json
import yaml
from transformers import AutoTokenizer
import sys

from def_fsm.paths import PROJECT_ROOT, expand

# training/ and data_transformation/ are script directories rather than packages,
# so make their flat imports resolvable from any working directory.
sys.path.append(str(PROJECT_ROOT / "training"))
sys.path.append(str(PROJECT_ROOT / "data_transformation"))

from utils import load_config  # training/utils.py
from dataset import TapeDataset


def main(
    config_path=expand("${PROJECT_ROOT}/configs/base.yaml"),
    ds_dir=expand("${DATA_ROOT}/FSM_tape_dataset"),
    # FSM_tape_dataset belongs to the "assistant" category
    system_prompt_path=None,
    output_json=expand("${DATA_ROOT}/FSM_tape_dataset/statistics.json"),
    splits=("train", "val"),  # this dataset has no test split
):
    # 1. Load config (for model_dir / new_tokens / max_length / system prompt)
    # load_config resolves the ${...} path placeholders in the config.
    cfg = load_config(config_path)

    model_dir = cfg["model_dir"]
    max_length = cfg["training"]["max_length"]
    if system_prompt_path is None:
        system_prompt_path = cfg["dataset"]["assistant"]["system_prompt_path"]

    # 2. Tokenizer + new tokens (same as the main statistics script)
    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    new_tokens = cfg.get("new_tokens")
    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} special tokens.")

    print(f"Max Length: {max_length} | System Prompt: {system_prompt_path}")

    # 3. Run statistics per split
    all_stats = {}
    for split in splits:
        json_file = os.path.join(ds_dir, f"{split}.json")
        print("\n" + "=" * 80)
        print(f">>> Split: [{split.upper()}]  ({json_file})")

        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        ds = TapeDataset(
            raw_data=raw_data,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt_path=system_prompt_path,
            verbose=True,
        )
        all_stats[split] = ds.stats

    # 4. Save
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)
    print(f"\nStatistics saved to {output_json}")


if __name__ == "__main__":
    main()
