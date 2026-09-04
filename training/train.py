"""Fine-tune a causal LM on FSM tapes. Entry point of scripts/train.sh.

Multi-GPU through accelerate + DeepSpeed. Validation tracks both loss and the state
transition F1, and a checkpoint is kept for the best of each.
"""

import os
import yaml
import torch
import shutil
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, ProjectConfiguration
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
import wandb
from icecream import ic
import fire
from tqdm import tqdm
from collections import defaultdict
import json
import random
import argparse
import gc
import math
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from def_fsm.utils import USER_PREFIX, ASSISTANT_PREFIX
from utils import (
    resize_and_initialize_embeddings,
    activate_reproducibility,
    load_config,
    update_and_save_config,
    parse_args_to_dict,
    plot_confusion_matrices,
    compute_batch_confusion_matrix,
)
from loss import compute_accuracy_counts, compute_accuracy_counts_per_sample, build_compute_loss_fn
from dataset import TapeDataset, CombinedTapeDataset, ResampledTapeDataset


def setup_accelerator(cfg):
    train_args = cfg["training"]
    project_config = ProjectConfiguration(project_dir=train_args["output_dir"], logging_dir=os.path.join(train_args["output_dir"], "logs"))
    ds_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"], offload_optimizer_device=train_args["offload_optimizer_device"]
    )
    accelerator = Accelerator(
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
        log_with="wandb",
        project_config=project_config,
    )

    print(f"Accelerator initialized. Device: {accelerator.device}, Num Processes: {accelerator.num_processes}")

    set_seed(cfg["seed"])

    # Init trackers (replaces wandb.init)
    if accelerator.is_main_process:
        # ensure the output directory exists
        os.makedirs(train_args["output_dir"], exist_ok=True)
        wandb_dir = train_args["output_dir"]
        run_name = os.path.basename(os.path.basename(train_args["output_dir"]))
        init_kwargs = {"wandb": {"dir": wandb_dir, "name": run_name}}
        accelerator.init_trackers(project_name=cfg["project_name"], config=cfg, init_kwargs=init_kwargs)

        # save the wandb Run ID for later resume
        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker is not None:
            run_id = wandb_tracker.run.id
            run_id_path = os.path.join(train_args["output_dir"], "wandb_run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(run_id)
            print(f"Saved wandb run ID ({run_id}) to {run_id_path}")
    return accelerator


def load_model_and_tokenizer(cfg, accelerator):
    accelerator.print(f"Loading Model: {cfg['model_dir']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_dir"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if cfg.get("new_tokens"):
        model, tokenizer = resize_and_initialize_embeddings(
            model,
            tokenizer,
            cfg["new_tokens"],
            init_method=cfg["token_init_method"],
            verbose=accelerator.is_main_process,
        )

    # Inject LoRA if enabled
    if cfg["training"]["finetune_mode"] == "lora":
        accelerator.print("Applying LoRA to the model...")

        # CRITICAL: If new tokens are added, we must train their embeddings and the lm_head
        modules_to_save = ["embed_tokens", "lm_head"] if cfg.get("new_tokens") else None

        lora_config = LoraConfig(
            r=cfg["training"]["lora_r"],
            lora_alpha=cfg["training"]["lora_alpha"],
            target_modules=cfg["training"]["lora_target_modules"],
            lora_dropout=cfg["training"]["lora_dropout"],
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)

        if accelerator.is_main_process:
            model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    return model, tokenizer


def load_dataset(cfg, tokenizer, accelerator):
    seed_worker, generator = activate_reproducibility(cfg["seed"])

    train_args = cfg["training"]
    verbose = accelerator.is_main_process
    model_config = AutoConfig.from_pretrained(cfg["model_dir"])
    model_type = getattr(model_config, "model_type", "unknown_arch")

    all_train_datasets = []  # holds Dataset objects across different categories (user/assistant)
    val_dataloaders = {}
    test_dataloaders = {}

    dataset_weight_registry = []
    global_dataset_id_counter = 0

    # iterate over the top-level keys in the dataset config (e.g. "user", "assistant")
    for category, category_config in cfg["dataset"].items():
        system_prompt_path = category_config["system_prompt_path"]
        dataset_paths = category_config["dataset_paths"]
        dataset_ratios = category_config["dataset_ratios"]
        dataset_loss_weights = category_config["dataset_loss_weights"]
        map_category_to_interlocutor = {
            "user": ASSISTANT_PREFIX,
            "assistant": USER_PREFIX,
        }  # note: this assigns the OTHER party's prefix, hence the swap

        accelerator.print(f"Processing Dataset Group: [{category}] | Prompt: {system_prompt_path}")

        # iterate over the concrete datasets in this group
        for ds_name, ds_path in dataset_paths.items():
            ratio = dataset_ratios.get(ds_name, 1.0)
            if ratio <= 0:
                accelerator.print(f"  - Skipping dataset: {ds_name} (ratio={ratio})")
                continue

            accelerator.print(f"  - Loading dataset: {ds_name} from {ds_path}")

            def load_json(split):
                path = os.path.join(ds_path, f"{split}.json")
                if not os.path.exists(path):
                    return None
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

            # Local cache policy: place .cache alongside the data directory
            local_cache_dir = os.path.join(ds_path, ".cache")
            os.makedirs(local_cache_dir, exist_ok=True)

            # 1. Collect raw training data (the base cache is always built from the 1.0-ratio raw data)
            raw_data = load_json("train")
            weights = dataset_loss_weights.get(ds_name, [0.5, 0.5])
            dataset_weight_registry.append(weights)
            current_ds_id = global_dataset_id_counter

            train_cache_name = f"{category}_train_max{train_args['max_length']}_seed{cfg['seed']}_{model_type}"
            train_cache_dir = os.path.join(local_cache_dir, train_cache_name)

            base_ds_instance = TapeDataset(
                raw_data,
                tokenizer,
                train_args["max_length"],
                system_prompt_path,
                verbose=verbose,
                interlocutor_prefix=map_category_to_interlocutor[category],
                dataset_id=current_ds_id,
                cache_dir=train_cache_dir,
            )

            # apply resampling dynamically (no extra storage or compute overhead)
            if ratio != 1.0:
                accelerator.print(
                    f"    -> Resampling {ds_name} (Packed): scale={ratio}, {len(base_ds_instance)} -> {int(len(base_ds_instance) * ratio)}"
                )

            ds_instance = ResampledTapeDataset(base_ds_instance, scale=ratio, seed=cfg["seed"])

            all_train_datasets.append(ds_instance)
            global_dataset_id_counter += 1
            accelerator.print(f"  -> Loaded {ds_name} (ID={current_ds_id}, Weights={weights}). Final Samples: {len(ds_instance)}")

            # 2. Build independent val and test sets for each dataset
            # Note: the key is prefixed with category to avoid name collisions between user and assistant datasets
            unique_ds_name = f"{category}_{ds_name}"
            v_raw = load_json("val")
            val_cache_dir = os.path.join(local_cache_dir, f"{category}_val_max{train_args['max_length']}_{model_type}")
            v_ds = TapeDataset(
                v_raw,
                tokenizer,
                train_args["max_length"],
                system_prompt_path,
                verbose=False,
                interlocutor_prefix=map_category_to_interlocutor[category],
                dataset_id=current_ds_id,
                cache_dir=val_cache_dir,
            )
            val_dataloaders[unique_ds_name] = DataLoader(
                v_ds, batch_size=train_args["per_device_train_batch_size"], num_workers=train_args["num_workers"]
            )

            t_raw = load_json("test")
            if t_raw is not None:
                test_cache_dir = os.path.join(local_cache_dir, f"{category}_test_max{train_args['max_length']}_{model_type}")
                t_ds = TapeDataset(
                    t_raw,
                    tokenizer,
                    train_args["max_length"],
                    system_prompt_path,
                    verbose=False,
                    interlocutor_prefix=map_category_to_interlocutor[category],
                    dataset_id=current_ds_id,
                    cache_dir=test_cache_dir,
                )
                test_dataloaders[unique_ds_name] = DataLoader(
                    t_ds, batch_size=train_args["per_device_train_batch_size"], num_workers=train_args["num_workers"]
                )

    combined_train_dataset = CombinedTapeDataset(all_train_datasets)
    weight_lookup_tensor = torch.tensor(dataset_weight_registry, dtype=torch.float32)  # (D, 2)

    maskid_to_tokenstr_map = combined_train_dataset.get_maskid_to_tokenstr_map()
    maskid_to_tokenid_map = combined_train_dataset.get_maskid_to_tokenid_map()
    accelerator.print(f"{maskid_to_tokenstr_map=}")
    accelerator.print(f"Total Combined Training Samples: {len(combined_train_dataset)}")

    train_dataloader = DataLoader(
        combined_train_dataset,
        shuffle=True,
        batch_size=train_args["per_device_train_batch_size"],
        num_workers=train_args["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_dataloader, val_dataloaders, test_dataloaders, maskid_to_tokenstr_map, maskid_to_tokenid_map, weight_lookup_tensor


def evaluate(
    model,
    dataloader,
    accelerator,
    maskid_to_tokenstr_map,
    maskid_to_tokenid_map,
    compute_loss_fn,
    cfg,
    return_conf_mat=False,
    save_per_sample=False,
    desc="Validation",
):
    """
    Validation Loop
    """
    model.eval()

    metrics_accum = defaultdict(lambda: torch.zeros(3, device=accelerator.device))

    # Per-sample result collection (for significance testing)
    per_sample_records = [] if save_per_sample else None

    # Initialize the confusion matrix structure
    if return_conf_mat:
        transition_mask_ids = sorted(list(maskid_to_tokenid_map.keys()))
        num_classes = len(transition_mask_ids) + 1
        vocab_size = model.get_input_embeddings().weight.shape[0]
        tokenid_to_class = torch.full((vocab_size,), num_classes - 1, dtype=torch.long, device=accelerator.device)
        for class_idx, mask_id in enumerate(transition_mask_ids):
            tokenid_to_class[maskid_to_tokenid_map[mask_id]] = class_idx

        global_conf_mat = torch.zeros(num_classes * num_classes, dtype=torch.long, device=accelerator.device)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not accelerator.is_main_process,
        desc=desc,
    )
    for step, batch in progress_bar:
        with torch.no_grad():
            input_ids = batch["input_ids"]
            mask = batch["mask"]

            outputs = model(input_ids, labels=input_ids)

            # 1. Losses
            loss, other_info = compute_loss_fn(batch, outputs)
            metrics_accum["loss"] += torch.tensor([loss.item(), 1.0, 0.0], device=accelerator.device)
            metrics_accum["entire_seq_loss"] += torch.tensor([outputs.loss.item(), 1.0, 0.0], device=accelerator.device)
            metrics_accum["response_loss"] += torch.tensor([other_info["response_loss"], 1.0, 0.0], device=accelerator.device)
            metrics_accum["transition_loss"] += torch.tensor([other_info["transition_loss"], 1.0, 0.0], device=accelerator.device)

            # 2. Compute metrics (local; returns counts only)
            batch_counts = compute_accuracy_counts(outputs.logits, input_ids, mask, maskid_to_tokenid_map)

            for mask_id, stats in batch_counts.items():
                if mask_id in maskid_to_tokenstr_map:
                    token_str = maskid_to_tokenstr_map[mask_id]
                    metrics_accum[f"stat_{token_str}"] += torch.tensor([stats["tp"], stats["fp"], stats["fn"]], device=accelerator.device)

            # collect per-sample TP/FP/FN (for significance testing)
            if save_per_sample:
                batch_per_sample = compute_accuracy_counts_per_sample(outputs.logits, input_ids, mask, maskid_to_tokenid_map)
                for sample_counts in batch_per_sample:
                    record = {}
                    f1_values_sample = []
                    for mid, stats in sample_counts.items():
                        if mid in maskid_to_tokenstr_map:
                            token_str = maskid_to_tokenstr_map[mid]
                            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                            record[token_str] = {"tp": tp, "fp": fp, "fn": fn}
                            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                            f1_values_sample.append(f1)
                    record["f1_avg"] = sum(f1_values_sample) / len(f1_values_sample) if f1_values_sample else 0.0
                    per_sample_records.append(record)

            # extract predictions and accumulate into the confusion matrix
            if return_conf_mat:
                batch_conf_mat = compute_batch_confusion_matrix(
                    logits=outputs.logits, labels=input_ids, mask=mask, tokenid_to_class=tokenid_to_class, num_classes=num_classes
                )
                global_conf_mat += batch_conf_mat

            # 3. Release GPU memory immediately
            del outputs, loss

    # End of loop, Global Reduce
    sorted_keys = sorted(metrics_accum.keys())
    if not sorted_keys:
        return {}

    # Stack & Reduce
    local_tensor = torch.stack([metrics_accum[k] for k in sorted_keys])
    global_tensor = accelerator.reduce(local_tensor, reduction="sum")

    # Compute the final results
    results = {}
    precision_values, recall_values, f1_values = [], [], []

    for key, tensor_val in zip(sorted_keys, global_tensor):
        if key.startswith("stat_"):
            # unpack into 3 variables
            tp, fp, fn = tensor_val.tolist()
            token_str = key.replace("stat_", "")

            # Compute Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[f"precision_{token_str}"] = precision
            results[f"recall_{token_str}"] = recall
            results[f"f1_{token_str}"] = f1
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        else:
            numerator, denominator, _ = tensor_val.tolist()
            metric_val = numerator / denominator if denominator > 0 else 0.0
            results[key] = metric_val

    if f1_values:
        results["precision_avg"] = sum(precision_values) / len(precision_values)
        results["recall_avg"] = sum(recall_values) / len(recall_values)
        results["f1_avg"] = sum(f1_values) / len(f1_values)

    # synchronize the final confusion matrix and place it into the returned dict
    if return_conf_mat:
        global_conf_mat = accelerator.reduce(global_conf_mat, reduction="sum")
        results["confusion_matrix"] = global_conf_mat.view(num_classes, num_classes)

    if save_per_sample:
        results["per_sample_records"] = per_sample_records

    torch.cuda.empty_cache()

    return results


def evaluate_datasets(
    model,
    dataloaders_dict,
    accelerator,
    maskid_to_tokenstr_map,
    maskid_to_tokenid_map,
    compute_loss_fn,
    cfg,
    log_prefix,
    return_conf_mat=False,
    save_per_sample=False,
):
    """
    Generic evaluation function: iterate over dataloaders_dict, evaluate the model, and return average metrics and a detailed log dict.
    """
    detailed_eval_results = {}

    target_keys = ["loss", "response_loss", "transition_loss", "precision_avg", "recall_avg", "f1_avg"]
    total_metrics = {k: 0.0 for k in target_keys}

    conf_mats_per_dataset = {}
    phase_name = "Test" if "test" in log_prefix.lower() else "Validation"

    for ds_name, dataloader in dataloaders_dict.items():
        dynamic_desc = f"{phase_name} ({ds_name})"
        eval_results = evaluate(
            model,
            dataloader,
            accelerator,
            maskid_to_tokenstr_map,
            maskid_to_tokenid_map,
            compute_loss_fn,
            cfg,
            return_conf_mat,
            save_per_sample,
            desc=dynamic_desc,
        )
        # pop the confusion matrix so it doesn't interfere with subsequent pure-numeric logs
        if return_conf_mat and "confusion_matrix" in eval_results:
            conf_mats_per_dataset[ds_name] = eval_results.pop("confusion_matrix")

        # pop the per-sample records to avoid interfering with pure-numeric logs
        per_sample_records_ds = eval_results.pop("per_sample_records", None)
        if save_per_sample and per_sample_records_ds is not None:
            for rec in per_sample_records_ds:
                rec["dataset"] = ds_name
            detailed_eval_results[f"{log_prefix}/{ds_name}/per_sample_records"] = per_sample_records_ds

        eval_log = {f"{log_prefix}/{ds_name}/{k}": v for k, v in eval_results.items()}
        detailed_eval_results.update(eval_log)

        for k in target_keys:
            total_metrics[k] += eval_results[k]

    # 3. Compute the averages
    avg_main_eval_results = {k: total_metrics[k] / len(dataloaders_dict) for k in target_keys}

    return avg_main_eval_results, detailed_eval_results, conf_mats_per_dataset


def save_checkpoint(accelerator, model, tokenizer, save_dir):
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        accelerator.print(f"Saving model to {save_dir}")

        tokenizer.save_pretrained(save_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)


def save_scores(results_to_log, output_dir, metric_prefix):

    json_path = os.path.join(output_dir, f"{metric_prefix}_assistant_state_transition_scores.json")
    with open(json_path, "w") as f:
        json.dump(results_to_log, f, indent=4)


def perform_final_testing(
    accelerator, config, test_dataloaders_dict, maskid_to_tokenstr_map, maskid_to_tokenid_map, compute_loss_fn, checkpoint_path, metric_prefix
):
    """
    Load the model at the given path and evaluate it on all test sets.
    """
    accelerator.print(f"Starting Final Testing for: {metric_prefix}")
    accelerator.print(f"Loading model from: {checkpoint_path}")

    # Correctly load model depending on LoRA or Full FT
    # The tokenizer is saved in the checkpoint directory in both cases
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if config["training"]["finetune_mode"] == "lora":
        accelerator.print("Loading Base Model + LoRA Adapters for evaluation...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_dir"],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # We must resize base_model embeddings before applying the LoRA adapter
        current_vocab_size = len(tokenizer)
        pad_to_multiple_of = 128
        target_vocab_size = ((current_vocab_size + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        base_model.resize_token_embeddings(target_vocab_size)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    elif config["training"]["finetune_mode"] == "full":
        accelerator.print("Loading Full-parameter fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    # 2. Prepare Model & Dataloaders
    model.eval()
    model.to(accelerator.device)
    # extract loaders and run prepare
    loader_names = list(test_dataloaders_dict.keys())
    loaders = list(test_dataloaders_dict.values())

    prepared_objs = accelerator.prepare(*loaders)
    if isinstance(prepared_objs, (list, tuple)):
        prepared_loaders = list(prepared_objs)
    else:
        prepared_loaders = [prepared_objs]
    prepared_test_dataloaders = dict(zip(loader_names, prepared_loaders))

    # 3. Run evaluation via the wrapper function
    # log_prefix example: "test/best_loss_model"
    log_prefix = f"test/{metric_prefix}"
    avg_results, results_to_log, conf_mats_per_dataset = evaluate_datasets(
        model,
        prepared_test_dataloaders,
        accelerator,
        maskid_to_tokenstr_map,
        maskid_to_tokenid_map,
        compute_loss_fn,
        config,
        log_prefix,
        return_conf_mat=True,
        save_per_sample=True,
    )

    # 4. Add the averages to the log
    results_to_log[f"{log_prefix}/avg_loss"] = avg_results["loss"]
    results_to_log[f"{log_prefix}/avg_response_loss"] = avg_results["response_loss"]
    results_to_log[f"{log_prefix}/avg_transition_loss"] = avg_results["transition_loss"]
    results_to_log[f"{log_prefix}/avg_precision"] = avg_results["precision_avg"]
    results_to_log[f"{log_prefix}/avg_recall"] = avg_results["recall_avg"]
    results_to_log[f"{log_prefix}/avg_f1"] = avg_results["f1_avg"]

    # 5. Compute the average F1 across the Human Dialogue datasets (Fisher & Switchboard, assistant perspective)
    # build the dict keys (matching the format produced by evaluate_datasets)
    fisher_key = f"{log_prefix}/assistant_fisher/f1_avg"
    swbd_key = f"{log_prefix}/assistant_switchboard/f1_avg"

    # use dict.get for safe lookup, in case a dataset is later removed from the config and would otherwise raise KeyError
    fisher_f1 = results_to_log.get(fisher_key)
    swbd_f1 = results_to_log.get(swbd_key)

    if fisher_f1 is not None and swbd_f1 is not None:
        human_f1_avg = (fisher_f1 + swbd_f1) / 2.0
        results_to_log[f"{log_prefix}/avg_assistant_humanData_f1"] = human_f1_avg
    else:
        accelerator.print("Warning: Could not find both 'assistant_fisher' and 'assistant_switchboard' in eval results to compute human dialogue F1.")

    # 1. Extract the true Step corresponding to the checkpoint (e.g. 480)
    if accelerator.is_main_process:
        step_to_log = 0
        if checkpoint_path and "_step_" in checkpoint_path:
            folder_name = os.path.basename(os.path.normpath(checkpoint_path))
            step_to_log = int(folder_name.split("_step_")[-1])
        accelerator.print(f"Extracted step={step_to_log} for WandB overwrite logging.")

        # 2. Write to History
        results_to_log["test/step"] = step_to_log
        if wandb.run is not None:
            # define an independent time variable
            wandb.define_metric("test/step")
            # tell WandB that for any metric starting with "test/", the X axis should be "test/step" instead of the global step
            wandb.define_metric("test/*", step_metric="test/step")
            # log directly, without any outer step argument
            accelerator.log(results_to_log)

            # 3. Force-write to Summary (the static metric table)
            # Summary is a static dict independent of the time axis; writing here ensures the WandB table panel always shows the latest test values
            wandb.run.summary.update(results_to_log)

    accelerator.print(f"Result ({metric_prefix}): Avg Loss: {avg_results['loss']:.4f} | Avg F1: {avg_results['f1_avg']:.4f}")

    # confusion matrix
    confusion_matrix_save_dir = os.path.join(config["training"]["output_dir"], "vis")
    plot_confusion_matrices(
        conf_mats_per_dataset=conf_mats_per_dataset,
        maskid_to_tokenstr_map=maskid_to_tokenstr_map,
        maskid_to_tokenid_map=maskid_to_tokenid_map,
        save_dir=confusion_matrix_save_dir,
        metric_prefix=metric_prefix,
        accelerator=accelerator,
    )

    # 6. Free GPU memory
    del model, prepared_test_dataloaders
    if len(loaders) > 0:
        del prepared_objs, prepared_loaders
    torch.cuda.empty_cache()
    gc.collect()

    # Save per-sample results (extracted from results_to_log to avoid serializing them to wandb)
    if accelerator.is_main_process:
        per_sample_all = {}
        keys_to_pop = [k for k in results_to_log if k.endswith("/per_sample_records")]
        for k in keys_to_pop:
            ds_name = k.split("/")[-2]
            per_sample_all[ds_name] = results_to_log.pop(k)
        if per_sample_all:
            per_sample_path = os.path.join(config["training"]["output_dir"], f"{metric_prefix}_per_sample_scores.json")
            with open(per_sample_path, "w") as f:
                json.dump(per_sample_all, f, indent=2)
            accelerator.print(f"Per-sample scores saved to {per_sample_path}")

        save_scores(results_to_log, config["training"]["output_dir"], metric_prefix)
    return results_to_log


def main(cfg):
    train_args = cfg["training"]

    # 1. Accelerator
    accelerator = setup_accelerator(cfg)

    # 2. Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, accelerator)

    # 3. Data Processing
    train_dataloader, val_dataloaders_dict, test_dataloaders_dict, maskid_to_tokenstr_map, maskid_to_tokenid_map, weight_lookup_tensor = load_dataset(
        cfg, tokenizer, accelerator
    )
    weight_lookup_tensor = weight_lookup_tensor.to(accelerator.device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])

    # 5. Loss Function
    compute_loss = build_compute_loss_fn(
        train_args=train_args,
        accelerator=accelerator,
        weight_lookup_tensor=weight_lookup_tensor,
        maskid_to_tokenid_map=maskid_to_tokenid_map,
        model_vocab_size=model.get_input_embeddings().weight.shape[0],
        train_dataset=train_dataloader.dataset,
    )

    # 6. Scheduler and epoch budget
    total_batch_size = train_args["per_device_train_batch_size"] * accelerator.num_processes * train_args["gradient_accumulation_steps"]
    num_update_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / total_batch_size)
    if train_args.get("max_train_steps") is not None:
        max_train_steps = train_args["max_train_steps"]
        # back-derive the total epochs needed from max_train_steps
        max_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        max_epochs = train_args["max_epochs"]
        max_train_steps = max_epochs * num_update_steps_per_epoch

    # multiply by num_processes upfront to cancel out the automatic division performed by accelerate.prepare
    target_warmup_steps = int(max_train_steps * train_args["warmup_ratio"]) * accelerator.num_processes
    target_train_steps = max_train_steps * accelerator.num_processes
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=target_warmup_steps, num_training_steps=target_train_steps)

    # Prepare everything including val_dataloader
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    prepared_val_objs = accelerator.prepare(*val_dataloaders_dict.values())
    if isinstance(prepared_val_objs, (list, tuple)):
        prepared_dataloaders = list(prepared_val_objs)
    else:
        prepared_dataloaders = [prepared_val_objs]
    val_dataloaders_dict = dict(zip(val_dataloaders_dict.keys(), prepared_dataloaders))

    # 7. Training Loop
    accelerator.print("Starting training...")
    accelerator.print(
        f"Total Batch Size: {train_args['per_device_train_batch_size'] * accelerator.num_processes * train_args['gradient_accumulation_steps']}, "
        f"Max Train Steps: {max_train_steps}, Max Epochs: {max_epochs}"
    )

    global_step = 0
    best_val_loss = float("inf")
    best_val_f1 = -float("inf")
    patience_counter = 0
    patience_limit = train_args["early_stopping_patience"]
    should_stop = False

    fixed_best_loss_dir = os.path.join(train_args["output_dir"], "checkpoint-best-loss")
    fixed_best_f1_dir = os.path.join(train_args["output_dir"], "checkpoint-best-f1")

    best_loss_meta = {}
    best_f1_meta = {}

    for epoch in range(max_epochs):
        if should_stop:
            break
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
        )
        for step, batch in progress_bar:
            if epoch == 0 and step == 0:
                accelerator.print("Example of one batch:")
                for i in range(train_args["per_device_train_batch_size"]):
                    accelerator.print(f"{tokenizer.decode(batch['input_ids'][i].tolist())=}")
                    accelerator.print(f"(token_ids, mask): {[i for i in zip(batch['input_ids'][i].tolist(), batch['mask'][i].tolist())]}")

            input_ids = batch["input_ids"]

            model.train()
            with accelerator.accumulate(model):
                outputs = model(input_ids, labels=input_ids)
                entire_seq_loss = outputs.loss
                loss, other_info = compute_loss(batch, outputs)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_args["max_grad_norm"])

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                global_step += 1  # a real batch instead of mini batch

                if global_step % train_args["logging_steps"] == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]

                    with torch.no_grad():
                        raw_train_metrics = compute_accuracy_counts(outputs.logits, batch["input_ids"], batch["mask"], maskid_to_tokenid_map)

                    log_data = {
                        "train_step": global_step,
                        "train/loss": loss.item(),
                        "train/response_loss": other_info["response_loss"],
                        "train/transition_loss": other_info["transition_loss"],
                        "train/entire_seq_loss": entire_seq_loss.item(),
                        "train/lr": current_lr,
                    }

                    precision_vals, recall_vals, f1_vals = [], [], []
                    for mask_id, stats in raw_train_metrics.items():
                        if mask_id in maskid_to_tokenstr_map:
                            token_str = maskid_to_tokenstr_map[mask_id]
                            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]

                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                            log_data[f"train/precision_{token_str}"] = precision
                            log_data[f"train/recall_{token_str}"] = recall
                            log_data[f"train/f1_{token_str}"] = f1

                            precision_vals.append(precision)
                            recall_vals.append(recall)
                            f1_vals.append(f1)

                    if f1_vals:
                        log_data["train/precision_avg"] = sum(precision_vals) / len(precision_vals)
                        log_data["train/recall_avg"] = sum(recall_vals) / len(recall_vals)
                        log_data["train/f1_avg"] = sum(f1_vals) / len(f1_vals)

                    accelerator.log(log_data, step=global_step)
                    accelerator.print(f"Step {global_step} | Loss: {loss.item():.4f}")

                if global_step % train_args["eval_steps"] == 0:
                    # Free unnecessary memory
                    try:
                        del outputs, loss, entire_seq_loss
                    except UnboundLocalError:
                        pass
                    torch.cuda.empty_cache()

                    avg_main_eval_results, detailed_eval_results, _ = evaluate_datasets(
                        model,
                        val_dataloaders_dict,
                        accelerator,
                        maskid_to_tokenstr_map,
                        maskid_to_tokenid_map,
                        compute_loss,
                        cfg,
                        log_prefix="val",
                        return_conf_mat=False,
                    )

                    # use the average Loss to drive early stopping
                    current_val_loss = avg_main_eval_results["loss"]
                    current_val_response_loss = avg_main_eval_results["response_loss"]
                    current_val_transition_loss = avg_main_eval_results["transition_loss"]
                    current_val_precision = avg_main_eval_results["precision_avg"]
                    current_val_recall = avg_main_eval_results["recall_avg"]
                    current_val_f1 = avg_main_eval_results["f1_avg"]

                    # Log Eval
                    accelerator.log(
                        {
                            "val/val_loss": current_val_loss,
                            "val/val_response_loss": current_val_response_loss,
                            "val/val_transition_loss": current_val_transition_loss,
                            "val/val_precision": current_val_precision,
                            "val/val_recall": current_val_recall,
                            "val/val_f1": current_val_f1,
                        },
                        step=global_step,
                    )
                    accelerator.log(detailed_eval_results, step=global_step)
                    accelerator.print(f"Validation (Step {global_step}): Loss {current_val_loss:.4f} | F1 Avg {current_val_f1:.4f}")

                    # Early Stopping Logic (Runs on all processes)
                    # Since evaluate gathers metrics, current_val_loss is consistent across GPUs
                    # The recall floor rejects the degenerate early checkpoints that reach a low loss
                    # by almost never emitting a transition token.
                    if current_val_loss < best_val_loss and current_val_recall >= 0.25:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        accelerator.print(f"  -> New best validation loss: {best_val_loss:.4f}. Resetting patience.")

                        save_checkpoint(accelerator, model, tokenizer, fixed_best_loss_dir)
                        best_loss_meta = {"val": best_val_loss, "step": global_step}
                    else:
                        patience_counter += 1
                        accelerator.print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{patience_limit}")
                        if patience_counter >= patience_limit:
                            accelerator.print(f"Early stopping triggered at step {global_step}!")
                            should_stop = True

                    if current_val_f1 > best_val_f1 and current_val_recall >= 0.25:
                        best_val_f1 = current_val_f1
                        accelerator.print(f"  -> New best validation f1: {best_val_f1:.4f}.")

                        save_checkpoint(accelerator, model, tokenizer, fixed_best_f1_dir)
                        best_f1_meta = {"val": best_val_f1, "step": global_step}

                    if should_stop:
                        break

                    if global_step >= max_train_steps:
                        accelerator.print(f"Reached max_train_steps ({max_train_steps}). Stopping training.")
                        should_stop = True
                        break

    final_loss_name = f"best_loss_{best_loss_meta.get('val', 0):.4f}_step_{best_loss_meta.get('step', 0)}"
    final_loss_path = os.path.join(train_args["output_dir"], final_loss_name)
    # best_f1_meta stays empty when no validation step cleared the F1 gate (a very
    # short run, for instance), so read it the same defensive way as the loss meta.
    final_f1_name = f"best_f1_{best_f1_meta.get('val', 0):.4f}_step_{best_f1_meta.get('step', 0)}"
    final_f1_path = os.path.join(train_args["output_dir"], final_f1_name)

    if accelerator.is_main_process:
        # rename the Best Loss folder
        if os.path.exists(fixed_best_loss_dir) and best_loss_meta:
            # if the target already exists (rare), remove it first
            if os.path.exists(final_loss_path):
                shutil.rmtree(final_loss_path)

            os.rename(fixed_best_loss_dir, final_loss_path)
            ic(f"Renamed {fixed_best_loss_dir} -> {final_loss_path}")
        else:
            ic(f"Rename failed! {fixed_best_loss_dir} not exists or {best_loss_meta=}")

        # rename the Best F1 folder
        if os.path.exists(fixed_best_f1_dir) and best_f1_meta:
            if os.path.exists(final_f1_path):
                shutil.rmtree(final_f1_path)

            os.rename(fixed_best_f1_dir, final_f1_path)
            ic(f"Renamed {fixed_best_f1_dir} -> {final_f1_path}")
        else:
            ic(f"Rename failed! {fixed_best_f1_dir} not exists or {best_loss_meta=}")

    # Final Test
    accelerator.print("Training finished. Clearing memory for testing phase...")

    del model, optimizer, lr_scheduler, train_dataloader, val_dataloaders_dict
    torch.cuda.empty_cache()
    gc.collect()

    accelerator.wait_for_everyone()

    # Final testing on the saved checkpoints is run separately, by
    # evaluation/state_transition_f1/state_transition_f1.sh.

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--override_path",
        type=str,
        nargs="*",
        default=[],
        help="Optional yaml files layered onto the base config, e.g. configs/experiments/wo_sac.yaml",
    )

    # parse known args (config_path); everything else goes into unknown_args for overrides
    args, unknown_args = parser.parse_known_args()

    print(parse_args_to_dict(unknown_args))

    # 1. Read the config, layer the override files, resolve ${...} paths
    cfg = load_config(args.config_path, args.override_path)

    # 2. Apply command-line overrides
    cfg = update_and_save_config(cfg, parse_args_to_dict(unknown_args))

    main(cfg=cfg)
