"""Split sizes and token counts for every dataset named in a training config, per ASR setting.

Builds the datasets through TapeDataset so the numbers match exactly what training would see.
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


# Maps each ASR method to the tag identifying it inside a dataset path
ASR_METHOD_TAG = {
    "faster": "faster_setting",
    "simul": "simul_setting",
}


def rewrite_path_for_asr(ds_dir, asr_method):
    """Rewrite a dataset path from the config for the given ASR method.
    Paths containing no ASR tag (shareGPT, synthetic) are returned unchanged.
    """
    tag = ASR_METHOD_TAG[asr_method]
    for other_tag in ASR_METHOD_TAG.values():
        if other_tag in ds_dir:
            return ds_dir.replace(other_tag, tag)
    return ds_dir  # a text-only dataset, with no ASR variants


def run_statistics(cfg, tokenizer, asr_method):
    """Walk every dataset and split for one ASR method and return a stats dict."""
    all_stats = {}

    for category in cfg["dataset"]:
        print("=" * 80)
        print(f"[ASR={asr_method.upper()}] CATEGORY {category.upper()}")
        print("=" * 80)

        all_stats.setdefault(category, {})
        dataset_paths = cfg["dataset"][category]["dataset_paths"]
        max_length = cfg["training"]["max_length"]
        system_prompt_path = cfg["dataset"][category]["system_prompt_path"]

        if not dataset_paths:
            print("No dataset paths found in config['dataset']['dataset_paths'].")
            continue

        print(f"Max Length: {max_length} | System Prompt: {system_prompt_path}")
        print("=" * 80)

        splits = ["train", "val", "test"]

        for ds_name, ds_dir_raw in dataset_paths.items():
            ds_dir = rewrite_path_for_asr(ds_dir_raw, asr_method)

            print(f"\nDataset: {ds_name.upper()}  (ASR={asr_method})")
            print(f"Path:    {ds_dir}")

            all_stats[category].setdefault(ds_name, {})

            for split in splits:
                json_file = os.path.join(ds_dir, f"{split}.json")

                print(f"\n>>> Split: [{split.upper()}]")

                if not os.path.exists(json_file):
                    print(f"File not found: {json_file}")
                    continue

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)

                    ds = TapeDataset(
                        raw_data=raw_data,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        system_prompt_path=system_prompt_path,
                        verbose=True,
                    )
                    # stats["total_tokens"] counts tokens at the raw tape stage, i.e. excluding the system
                    # prompt (injected at step 2) and the padding (injected at step 3).
                    all_stats[category][ds_name][split] = ds.stats

                except Exception as e:
                    print(f"Error processing {ds_name} - {split}: {e}")

        print("\n" + "=" * 80)

    return all_stats


def main(
    config_path=expand("${PROJECT_ROOT}/configs/base.yaml"),
    output_json_template=expand("${DATA_ROOT}") + "/dataset_statistics_{asr}.json",
    asr_methods=("faster", "simul"),
):
    # 1. Read the config
    print(f"Loading configuration from {config_path}...")
    # load_config resolves the ${...} path placeholders in the config.
    cfg = load_config(config_path)

    # 2. Load the tokenizer
    model_dir = cfg["model_dir"]
    print(f"Loading Tokenizer from: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Add the new tokens once, shared across every ASR method
    new_tokens = cfg.get("new_tokens")
    if new_tokens:
        print(f"Adding {len(new_tokens)} special tokens to tokenizer: {new_tokens}")
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"   Added {num_added} tokens.")
    else:
        print("No new tokens found in config.")

    # 4. Measure and save each ASR method separately
    for asr_method in asr_methods:
        print("\n" + "#" * 80)
        print(f"# RUNNING STATISTICS FOR ASR METHOD: {asr_method.upper()}")
        print("#" * 80)

        stats = run_statistics(cfg, tokenizer, asr_method)

        output_json = output_json_template.format(asr=asr_method)
        print(f"Saving statistics for ASR={asr_method} to {output_json}...")
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"Statistics for ASR={asr_method} saved.")


if __name__ == "__main__":
    main()
