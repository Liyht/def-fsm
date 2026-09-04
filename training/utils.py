"""Config loading, embedding resizing, logit-adjustment priors and evaluation plotting."""

import torch
import os
import numpy as np
import random
import yaml
import ast
import math
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl

from def_fsm.paths import expand, expand_config


def resize_and_initialize_embeddings(model, tokenizer, new_tokens_list, init_method="global_mean", verbose=True):
    """
    Resizes the tokenizer and model embeddings, initializing new tokens
    based on the specified strategy.

    Args:
        model: The transformer model.
        tokenizer: The tokenizer.
        new_tokens_list: List of strings to add.
        init_method: 'random', 'global_mean', or 'subtoken_mean'.
    """
    if not new_tokens_list:
        return model, tokenizer

    if verbose:
        print(f"Processing {len(new_tokens_list)} new tokens with strategy: {init_method}")

    # Step 0: Pre-calculation (for subtoken_mean)
    # We must calculate this BEFORE adding tokens, while the tokenizer
    # still tokenizes these strings into existing sub-words.
    if init_method == "subtoken_mean":
        subtoken_embeddings = {}
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        for token_str in new_tokens_list:
            # Tokenize the string using the ORIGINAL vocabulary
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if ids:
                ids_tensor = torch.tensor(ids, device=model.device)
                # Average the embeddings of the constituent sub-tokens
                in_avg = input_embeddings[ids_tensor].mean(dim=0)
                out_avg = output_embeddings[ids_tensor].mean(dim=0)
                subtoken_embeddings[token_str] = (in_avg, out_avg)
            else:
                subtoken_embeddings[token_str] = None

    # Step 1: Add Tokens & Resize
    num_added_tokens = tokenizer.add_tokens(new_tokens_list)

    # Resize to a multiple of 64 for NVIDIA Ampere Tensor Core performance
    current_vocab_size = len(tokenizer)
    pad_to_multiple_of = 128
    target_vocab_size = ((current_vocab_size + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    if verbose:
        print(f"Resizing embeddings from {model.get_input_embeddings().weight.shape[0]} to {target_vocab_size}...")

    model.resize_token_embeddings(target_vocab_size)

    # Get references to the resized weights
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # Calculate indices for the new tokens
    # Note: tokenizer adds new tokens at the end of the valid vocabulary
    start_index = current_vocab_size - num_added_tokens
    end_index = current_vocab_size

    # Step 2: Initialize

    # Strategy A: Random (Default behavior of resize_token_embeddings, usually N(0, 0.02))
    if init_method == "random":
        pass

    # Strategy B: Global Mean (Recommended for stability)
    elif init_method == "global_mean":
        # Calculate mean of valid existing tokens (excluding the newly added ones)
        # We use 'start_index' as the boundary for valid existing tokens
        valid_range = start_index

        in_avg = input_embeddings[:valid_range].mean(dim=0)
        out_avg = output_embeddings[:valid_range].mean(dim=0)

        input_embeddings[start_index:end_index] = in_avg
        output_embeddings[start_index:end_index] = out_avg

    # Strategy C: Sub-token Mean (Best for semantic terms)
    elif init_method == "subtoken_mean":
        # Fallback for tokens that couldn't be tokenized properly
        valid_range = start_index

        for i, token_str in enumerate(new_tokens_list):
            target_idx = start_index + i
            in_avg, out_avg = subtoken_embeddings.get(token_str)

            input_embeddings[target_idx] = in_avg
            output_embeddings[target_idx] = out_avg

    return model, tokenizer


def activate_reproducibility(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g


def deep_update(base, overrides):
    """Recursively merge `overrides` into `base` and return `base`.

    Nested dicts are merged key by key; every other value (including lists such
    as dataset_loss_weights) is replaced wholesale.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path, override_paths=()):
    """Load a yaml config, layer the override files on top, expand ${...} paths.

    The base config carries every key; an override file only needs the keys that
    differ from it (see configs/experiments/). Placeholders such as
    ${DATA_ROOT} are resolved from configs/paths.yaml.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for path in override_paths or ():
        with open(path, "r") as f:
            deep_update(cfg, yaml.safe_load(f) or {})

    return expand_config(cfg)


def update_and_save_config(cfg, kwargs, save_filename="config.yaml"):
    """
    1. Recursively update the config dict with command line arguments (kwargs)
    2. Create the output directory
    3. Save the final config to the output directory
    """
    # 1. Apply Overrides
    for key, value in kwargs.items():
        target = cfg
        keys = key.split(".")  # supports the training.learning_rate dot-notation form
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = expand(value)

    # 2. Create Dir & Save
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, save_filename)
    with open(save_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Config updated and saved to: {save_path}")
    return cfg


def parse_args_to_dict(args_list):
    """
    Parse the unknown_args (list) returned by argparse into a dict.
    Supported formats:
      --key value
      --key=value
      --key (bool flag, defaults to True)
    """
    overrides = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]

        # must start with --
        if not arg.startswith("--"):
            i += 1
            continue

        key = arg[2:]  # strip the leading --
        value = None

        # Case 1: --key=value
        if "=" in key:
            key, value_str = key.split("=", 1)
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str  # keep as string
            i += 1

        # Case 2: --key value (next element does not start with --)
        elif i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
            value_str = args_list[i + 1]
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str
            i += 2

        # Case 3: --key (Boolean flag)
        else:
            value = True
            i += 1

        overrides[key] = value

    return overrides


def compute_logit_adjustments_tensor(dataset, model_vocab_size, prob_denominator, tau=1.0):
    """
    Calculates tau * log(pi) for tokens.
     The logit adjustment requires class priors pi.
     prob_denominator is "all" or "transition".
    """
    print(f"Computing class priors for Logit Adjustment (tau={tau})...")

    total_counts = dataset.get_token_counts()
    maskid_to_tokenstr_map = dataset.get_maskid_to_tokenstr_map()
    maskid_to_tokenid_map = dataset.get_maskid_to_tokenid_map()

    num_total_tokens = sum(total_counts.values())

    # Collect all token IDs that are considered "transitions" (mask >= 2)
    # The map is {mask_id: token_str}. We need token_ids.
    transition_token_strs = []
    transition_token_ids = []
    for mask_id, token_str in maskid_to_tokenstr_map.items():
        transition_token_strs.append(token_str)
        transition_token_ids.append(maskid_to_tokenid_map[mask_id])

    indices = torch.tensor(transition_token_ids, dtype=torch.long)

    counts = torch.tensor([total_counts[idx] for idx in transition_token_ids], dtype=torch.float32)
    # probs should never be 0 here, but clamp before the log to be safe
    if prob_denominator == "all":
        probs = counts / num_total_tokens
    elif prob_denominator == "transition":
        total_transition_count = counts.sum()
        probs = counts / total_transition_count
    probs = probs.clamp(min=1e-9)
    adjustments_values = tau * torch.log(probs)
    print(f"{transition_token_strs=}")
    print(f"{indices=}")
    print(f"{probs=}")
    print(f"{adjustments_values=}")

    adjustments = torch.zeros(model_vocab_size)
    adjustments[indices] = adjustments_values

    return adjustments


def compute_all_logit_adjustments_tensor(dataset, model_vocab_size, prob_denominator, tau=1.0):
    """
    Calculates tau * log(pi) for ALL token positions (response + transition).
    - Transition tokens: each has its own prior pi_k (per-token probability).
    - Response tokens: all share a unified prior pi_resp (total response probability as one class).
    """
    print(f"Computing class priors for All Logit Adjustment (tau={tau})...")

    total_counts = dataset.get_token_counts()
    maskid_to_tokenstr_map = dataset.get_maskid_to_tokenstr_map()
    maskid_to_tokenid_map = dataset.get_maskid_to_tokenid_map()

    num_total_tokens = sum(total_counts.values())

    transition_token_strs = []
    transition_token_ids = []
    for mask_id, token_str in maskid_to_tokenstr_map.items():
        transition_token_strs.append(token_str)
        transition_token_ids.append(maskid_to_tokenid_map[mask_id])

    indices = torch.tensor(transition_token_ids, dtype=torch.long)
    counts = torch.tensor([total_counts[idx] for idx in transition_token_ids], dtype=torch.float32)

    transition_total = counts.sum().item()
    response_total = num_total_tokens - transition_total

    if prob_denominator == "all":
        trans_probs = counts / num_total_tokens
        resp_prob = response_total / num_total_tokens
    elif prob_denominator == "transition":
        trans_probs = counts / counts.sum()
        resp_prob = response_total / num_total_tokens

    trans_probs = trans_probs.clamp(min=1e-9)
    resp_prob = max(resp_prob, 1e-9)

    resp_adjustment = tau * math.log(resp_prob)
    trans_adjustments = tau * torch.log(trans_probs)

    print(f"{transition_token_strs=}")
    print(f"{indices=}")
    print(f"response_prob={resp_prob}, response_adjustment={resp_adjustment}")
    print(f"transition_probs={trans_probs}")
    print(f"transition_adjustments={trans_adjustments}")

    # Fill all positions with response adjustment, then override transition positions
    adjustments = torch.full((model_vocab_size,), resp_adjustment)
    adjustments[indices] = trans_adjustments

    return adjustments


def gen_maskid_to_tokenid_map(maskid_to_tokenstr_map, tokenizer):
    maskid_to_tokenid_map = {}
    for mask_id, token_str in maskid_to_tokenstr_map.items():
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        assert len(ids) == 1, f"{token_str} -> {ids}, more than one token!"
        maskid_to_tokenid_map[mask_id] = ids[0]
    return maskid_to_tokenid_map


def get_wandb_run_id(output_dir, project_name):
    """
    Recover the wandb run ID of an existing training run, so evaluation can log into it.
    Three sources are tried in order:
    1. wandb_run_id.txt, which the training run writes
    2. the name of the local wandb folder
    3. the wandb API
    """
    # 1. The ID the training run saved explicitly
    id_file_path = os.path.join(output_dir, "wandb_run_id.txt")
    if os.path.exists(id_file_path):
        with open(id_file_path, "r") as f:
            return f.read().strip()

    print("[INFO] wandb_run_id.txt not found, attempting to recover the run ID from the local wandb folder...")

    # 2. Try to parse from the local wandb folder name (usually in the form run-YYYYMMDD_HHMMSS-RUN_ID)
    wandb_local_dir = os.path.join(output_dir, "wandb")
    if os.path.exists(wandb_local_dir):
        # filter folders that start with 'run-', sort by modification time and take the latest
        run_folders = [f for f in os.listdir(wandb_local_dir) if f.startswith("run-") and os.path.isdir(os.path.join(wandb_local_dir, f))]
        if run_folders:
            run_folders.sort(key=lambda x: os.path.getmtime(os.path.join(wandb_local_dir, x)))
            latest_run_folder = run_folders[-1]
            # extract the last segment, i.e. the RUN_ID
            run_id = latest_run_folder.split("-")[-1]
            print(f"[OK] Extracted Run ID from local folder {latest_run_folder}: {run_id}")
            return run_id

    # 3. Try to look it up in the cloud via the Wandb API
    print("[INFO] No local wandb record directory found, querying the cloud via API...")
    try:
        api = wandb.Api()
        # run_name is the basename of output_dir
        target_run_name = os.path.basename(output_dir)

        # find runs in the current project whose name matches
        runs = api.runs(f"{project_name}", filters={"display_name": target_run_name})

        if len(runs) > 0:
            run_id = runs[0].id
            print(f"[OK] Matched Run ID from Wandb cloud: {run_id}")
            return run_id
        else:
            raise ValueError(f"Could not find a Run named {target_run_name} in project {project_name}.")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to retrieve Run ID, please check the network or wandb login status. Error: {e}")


def compute_batch_confusion_matrix(logits, labels, mask, tokenid_to_class, num_classes):
    """
    Compute the 1D-flattened confusion matrix for a single batch.
    """
    # 1. Shift (align Causal LM inputs and predictions)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # 2. Get predictions and flatten
    preds = shift_logits.argmax(dim=-1).view(-1)
    flat_labels = shift_labels.view(-1)
    valid_mask = shift_mask.view(-1) != 0

    # 3. Filter valid tokens and map to our defined classes (0~5)
    pred_classes = tokenid_to_class[preds[valid_mask]]
    true_classes = tokenid_to_class[flat_labels[valid_mask]]

    # 4. Convert 2D coordinates to 1D indices and count frequencies
    indices = true_classes * num_classes + pred_classes
    batch_conf_mat = torch.bincount(indices, minlength=num_classes**2)

    return batch_conf_mat


def plot_confusion_matrices(
    conf_mats_per_dataset, maskid_to_tokenstr_map, maskid_to_tokenid_map, save_dir, metric_prefix, accelerator, base_font_size=22
):
    """
    Receive a dict of pre-computed confusion matrices; the main process plots and saves them as images.
    """
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "conf_mats_per_dataset": conf_mats_per_dataset,
            "maskid_to_tokenstr_map": maskid_to_tokenstr_map,
            "maskid_to_tokenid_map": maskid_to_tokenid_map,
        },
        os.path.join(save_dir, f"conf_mat_inputs_{metric_prefix}.pt"),
    )

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "font.family": ["DejaVu Sans"],
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 4,
            "axes.labelsize": base_font_size + 4,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
        }
    )

    transition_mask_ids = sorted(list(maskid_to_tokenid_map.keys()))
    class_names = [maskid_to_tokenstr_map[m] for m in transition_mask_ids] + ["Response"]
    for i in range(len(class_names)):
        if class_names[i] == "S.LISTEN.INTERRUPT":
            class_names[i] = "S.LISTEN.I"
        elif class_names[i] == "S.LISTEN.NATURAL":
            class_names[i] = "S.LISTEN.N"

    wandb_logs = {}

    for ds_name, conf_mat in conf_mats_per_dataset.items():
        conf_mat_np = conf_mat.cpu().numpy()

        # row-wise normalization
        row_sums = conf_mat_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        conf_mat_norm = conf_mat_np / row_sums

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_mat_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": base_font_size * 0.9},
        )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"confusion_matrix_{metric_prefix}_{ds_name}.pdf")
        plt.savefig(save_path, dpi=300)
        plt.close()

        accelerator.print(f"Saved confusion matrix plot for '{ds_name}' to: {save_path}")

