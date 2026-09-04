"""
Token-type ratios over the four datasets (Fisher and Switchboard, each under both ASR settings):
the five state transition tokens against the response tokens.

Reuses TapeDataset:
  - _compute_token_counts only counts tokens with mask >= 1, which naturally excludes the system
    prompt and the <interlocutor> spans;
  - each transition token is a single subword, located directly through maskid_to_tokenid_map;
  - the response count is the total mask >= 1 count minus the five transition counts.
"""

import os
import json
import yaml
import sys
from collections import OrderedDict

from transformers import AutoTokenizer

from def_fsm.paths import PROJECT_ROOT, expand

# training/ and data_transformation/ are script directories rather than packages,
# so make their flat imports resolvable from any working directory.
sys.path.append(str(PROJECT_ROOT / "training"))
sys.path.append(str(PROJECT_ROOT / "data_transformation"))

from utils import load_config  # training/utils.py
from dataset import TapeDataset

from def_fsm.utils import USER_PREFIX, ASSISTANT_PREFIX


ASR_METHOD_TAG = {
    "faster": "faster_setting",
    "simul": "simul_setting",
}

# The four datasets to measure: (Fisher, Switchboard) x (faster, simul)
DATASETS = ["fisher", "switchboard"]
ASR_METHODS = ["faster", "simul"]
# The assistant group is measured by default, since that is what is trained; change this for the user side
CATEGORY = "assistant"
SPLIT = "train"


def rewrite_path_for_asr(ds_dir, asr_method):
    tag = ASR_METHOD_TAG[asr_method]
    for other_tag in ASR_METHOD_TAG.values():
        if other_tag in ds_dir:
            return ds_dir.replace(other_tag, tag)
    return ds_dir


def compute_class_breakdown(ds):
    """
    Returns an OrderedDict {token_str: count} with the five transitions plus 'response',
    ordered by ascending mask_id with response last.
    """
    token_counts = ds.get_token_counts()                  # Counter[token_id] -> count
    maskid_to_tokenstr = ds.get_maskid_to_tokenstr_map()  # {mask_id: token_str}
    maskid_to_tokenid = ds.get_maskid_to_tokenid_map()    # {mask_id: token_id}

    breakdown = OrderedDict()
    transition_total = 0
    for mask_id in sorted(maskid_to_tokenstr.keys()):
        token_str = maskid_to_tokenstr[mask_id]
        token_id = maskid_to_tokenid[mask_id]
        cnt = token_counts.get(token_id, 0)
        breakdown[token_str] = cnt
        transition_total += cnt

    response_total = sum(token_counts.values()) - transition_total
    breakdown["response"] = response_total
    return breakdown


def format_breakdown(breakdown):
    total = sum(breakdown.values())
    lines = []
    lines.append(f"  {'token':<28}{'count':>14}{'ratio':>12}")
    lines.append(f"  {'-' * 28}{'-' * 14}{'-' * 12}")
    for k, v in breakdown.items():
        ratio = v / total if total > 0 else 0.0
        lines.append(f"  {k:<28}{v:>14d}{ratio:>11.4%}")
    lines.append(f"  {'TOTAL':<28}{total:>14d}{1.0:>11.4%}")
    return "\n".join(lines)


def main(
    config_path=expand("${PROJECT_ROOT}/configs/base.yaml"),
    output_json=expand("${DATA_ROOT}/token_ratio_statistics.json"),
    category=CATEGORY,
    split=SPLIT,
):
    print(f"Loading config from {config_path}")
    # load_config resolves the ${...} path placeholders in the config.
    cfg = load_config(config_path)

    model_dir = cfg["model_dir"]
    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    new_tokens = cfg.get("new_tokens", [])
    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} special tokens to tokenizer.")

    # category -> interlocutor prefix (kept consistent with train.py)
    interlocutor_prefix = {"user": ASSISTANT_PREFIX, "assistant": USER_PREFIX}[category]
    cat_cfg = cfg["dataset"][category]
    dataset_paths = cat_cfg["dataset_paths"]

    max_length = cfg["training"]["max_length"]

    results = {}

    for ds_name in DATASETS:
        if ds_name not in dataset_paths:
            print(f"[WARN] {ds_name} not in config['dataset'][{category}]['dataset_paths']")
            continue

        for asr_method in ASR_METHODS:
            ds_dir = rewrite_path_for_asr(dataset_paths[ds_name], asr_method)
            json_file = os.path.join(ds_dir, f"{split}.json")

            tag = f"{ds_name}/{asr_method}"
            print("\n" + "=" * 80)
            print(f"DATASET: {tag}  ({split})")
            print(f"PATH:    {json_file}")
            print("=" * 80)

            if not os.path.exists(json_file):
                print(f"[ERROR] file not found: {json_file}")
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # system_prompt_path=None: these statistics are computed at step 1, where the prompt plays no
            # part, which also saves the cost of reading it.
            ds = TapeDataset(
                raw_data=raw_data,
                tokenizer=tokenizer,
                max_length=max_length,
                system_prompt_path=None,
                verbose=False,
                interlocutor_prefix=interlocutor_prefix,
                cache_dir=None,
            )

            breakdown = compute_class_breakdown(ds)
            print(format_breakdown(breakdown))

            results[tag] = breakdown

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        # Turn the OrderedDict into a list of pairs to preserve order, adding the ratio
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
