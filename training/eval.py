"""Re-run the test-split evaluation of a finished training run, without retraining.

Reads the config.yaml the run saved, so evaluation sees the same tokenizer, prompts and data.
"""

import os
import yaml
import argparse
import torch
import gc
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, ProjectConfiguration
from transformers import AutoTokenizer
from functools import partial

from utils import parse_args_to_dict, load_config, update_and_save_config, get_wandb_run_id
from train import load_model_and_tokenizer, load_dataset, perform_final_testing
from loss import build_compute_loss_fn


def setup_eval_accelerator(cfg, run_id=None, skip_wandb=False):
    train_args = cfg["training"]
    project_config = ProjectConfiguration(project_dir=train_args["output_dir"], logging_dir=os.path.join(train_args["output_dir"], "logs"))
    ds_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"], offload_optimizer_device=train_args["offload_optimizer_device"]
    )

    accelerator_kwargs = {
        "deepspeed_plugin": ds_plugin,
        "project_config": project_config,
    }
    if not skip_wandb:
        accelerator_kwargs["log_with"] = "wandb"

    accelerator = Accelerator(**accelerator_kwargs)

    set_seed(cfg["seed"])

    if skip_wandb:
        if accelerator.is_main_process:
            print("skip_wandb=True: skipping wandb tracker initialization.")
        return accelerator

    if accelerator.is_main_process:
        if run_id is not None:
            # resume the existing wandb run
            init_kwargs = {
                "wandb": {
                    "dir": train_args["output_dir"],
                    "name": os.path.basename(train_args["output_dir"]),
                    "id": run_id,
                    "resume": "must",
                }
            }
        else:
            # base model mode: create a new wandb run
            init_kwargs = {
                "wandb": {
                    "dir": train_args["output_dir"],
                    "name": os.path.basename(train_args["output_dir"]),
                }
            }
        accelerator.init_trackers(project_name=cfg["project_name"], config=cfg, init_kwargs=init_kwargs)

    return accelerator


def main(cfg, skip_existing=False, base_model=False, skip_wandb=False, metric_suffix=""):
    train_args = cfg["training"]
    output_dir = train_args["output_dir"]
    project_name = cfg["project_name"]

    # 0. Check whether testing can be skipped early, to save model and dataset loading time
    metric_prefix = "base_model" if base_model else "best_loss_model"
    if metric_suffix:
        metric_prefix = f"{metric_prefix}{metric_suffix}"
    score_path = os.path.join(output_dir, f"{metric_prefix}_assistant_state_transition_scores.json")
    sample_score_path = os.path.join(output_dir, f"{metric_prefix}_per_sample_scores.json")

    run_best_loss = True
    if skip_existing and os.path.exists(score_path) and os.path.exists(sample_score_path):
        print(f"Skipping evaluation for {metric_prefix}. Results already exist at: {score_path}")
        run_best_loss = False

    # if results already exist for every model that needs testing, exit immediately
    if not run_best_loss:
        print("All required evaluations are already completed. Exiting.")
        return

    # 1. Initialize the Accelerator
    if base_model:
        accelerator = setup_eval_accelerator(cfg, run_id=None, skip_wandb=skip_wandb)
    else:
        run_id = None if skip_wandb else get_wandb_run_id(output_dir, project_name)
        accelerator = setup_eval_accelerator(cfg, run_id, skip_wandb=skip_wandb)

    # 2. Prepare the Tokenizer and dataset mappings
    model, tokenizer = load_model_and_tokenizer(cfg, accelerator)

    # fetch test data; keep train_dataloader so we can compute logit_adjustments
    train_dataloader, val_dataloaders_dict, test_dataloaders_dict, maskid_to_tokenstr_map, maskid_to_tokenid_map, weight_lookup_tensor = load_dataset(
        cfg, tokenizer, accelerator
    )
    weight_lookup_tensor = weight_lookup_tensor.to(accelerator.device)

    # 3. Build the Loss function (matching train.py)
    compute_loss = build_compute_loss_fn(
        train_args=train_args,
        accelerator=accelerator,
        weight_lookup_tensor=weight_lookup_tensor,
        maskid_to_tokenid_map=maskid_to_tokenid_map,
        model_vocab_size=model.get_input_embeddings().weight.shape[0],
        train_dataset=train_dataloader.dataset,
    )

    # free data we no longer need to save memory
    del train_dataloader, val_dataloaders_dict
    torch.cuda.empty_cache()
    gc.collect()

    if base_model:
        # Base model mode: load_model_and_tokenizer already handled the new_tokens resize;
        # save the resized model and tokenizer to a temporary directory so perform_final_testing can load them
        base_model_dir = os.path.join(output_dir, "base_model_with_new_tokens")
        if not os.path.exists(base_model_dir):
            if accelerator.is_main_process:
                model.save_pretrained(base_model_dir)
                tokenizer.save_pretrained(base_model_dir)
            accelerator.wait_for_everyone()

        cfg["training"]["finetune_mode"] = "full"
        perform_final_testing(
            accelerator=accelerator,
            config=cfg,
            test_dataloaders_dict=test_dataloaders_dict,
            maskid_to_tokenstr_map=maskid_to_tokenstr_map,
            maskid_to_tokenid_map=maskid_to_tokenid_map,
            compute_loss_fn=compute_loss,
            checkpoint_path=base_model_dir,
            metric_prefix=metric_prefix,
        )
    else:
        # 4. Locate the Checkpoint and run testing
        best_loss_path, best_f1_path = None, None
        for folder in os.listdir(output_dir):
            full_path = os.path.join(output_dir, folder)
            if os.path.isdir(full_path):
                if folder.startswith("best_loss_") and "gguf" not in folder:
                    best_loss_path = full_path
                elif folder.startswith("best_f1_") and "gguf" not in folder:
                    best_f1_path = full_path

        # run the Best Loss evaluation
        if run_best_loss:
            perform_final_testing(
                accelerator=accelerator,
                config=cfg,
                test_dataloaders_dict=test_dataloaders_dict,
                maskid_to_tokenstr_map=maskid_to_tokenstr_map,
                maskid_to_tokenid_map=maskid_to_tokenid_map,
                compute_loss_fn=compute_loss,
                checkpoint_path=best_loss_path,
                metric_prefix=metric_prefix,
            )

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
    parser.add_argument("--skip_existing", action="store_true", help="Skip testing if the evaluation score JSON already exists")
    parser.add_argument("--base_model", action="store_true", help="Evaluate the base model directly without loading a fine-tuned checkpoint")
    parser.add_argument("--skip_wandb", action="store_true", help="Skip all wandb initialization and logging")
    parser.add_argument("--metric_suffix", type=str, default="", help="Suffix appended to metric_prefix so output JSON/CM filenames don't clash with prior runs (e.g. '_simul')")
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_path, args.override_path)
    cfg = update_and_save_config(cfg, parse_args_to_dict(unknown_args), save_filename="eval_config.yaml")

    main(
        cfg=cfg,
        skip_existing=args.skip_existing,
        base_model=args.base_model,
        skip_wandb=args.skip_wandb,
        metric_suffix=args.metric_suffix,
    )
