"""Transcribe a corpus with the same perception module the FSM deploys, one worker per GPU.

Writes one JSON of timestamped recognition results per audio file, and reports WER against
the reference transcripts so badly recognized dialogues can be excluded downstream.
"""

import os
import torch
import torch.multiprocessing as mp
from queue import Empty
import glob
import logging
from tqdm import tqdm
from def_fsm.asr_worker import FSMSimulWhisperASRWorker, FSMFasterWhisperASRWorker
from def_fsm.utils import SILENCE_TOKEN, SILENCE_TOKEN_DUR
from audio_dataset import SwitchboardDataset
import jiwer
import json
import argparse
import re
import random

import numpy as np
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
import seaborn as sns

from def_fsm.paths import expand

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s")
logger = logging.getLogger("BatchASR")


def get_gpu_count():
    """Returns the number of available GPUs."""
    count = torch.cuda.device_count()
    if count == 0:
        logger.warning("No GPU detected, falling back to CPU (Process 1 only).")
        return 0
    return count


def merge_asr_segments(raw_segments):
    """
    Merge streaming output of the form [(text, end_time), ...] into [(transcript, start, end), (SILENCE_TOKEN, start, end)].
    Consecutive text is merged, and so are consecutive SILENCE_TOKENs.

    Args:
        raw_segments: the input list, e.g. [("Hello", 1.0), (" world", 1.5), (SILENCE_TOKEN, 2.0), (SILENCE_TOKEN, 2.5)]

    Returns:
        The merged list, e.g. [("Hello world", 0.0, 1.5), (SILENCE_TOKEN, 1.5, 2.5)]
    """
    if not raw_segments:
        return []

    merged_results = []

    # Buffer accumulating consecutive text tokens
    text_buffer = []

    # Start time of the current segment (0.0 initially)
    current_start_time = 0.0

    # End time of the previous token, used to close a speech segment when a SIL arrives
    last_end_time = 0.0

    for token, receive_timestamp, end_timestamp, audio_start, audio_end in raw_segments:
        if token == SILENCE_TOKEN:
            # Case 1: a silence token

            # A. Text in the buffer means speech was in progress and has now ended
            if text_buffer:
                # Close the speech segment
                full_text = "".join(text_buffer)  # use " ".join(...) if the tokens carry no spaces
                merged_results.append((full_text, current_start_time, last_end_time))

                # Clear the buffer
                text_buffer = []
                # The next segment (this silence) starts where the speech just ended
                current_start_time = last_end_time

            # B. Handle the silence itself
            # If the last result is already a SILENCE_TOKEN, just extend its end time;
            if merged_results and merged_results[-1][0] == SILENCE_TOKEN:

                _, s_start, _ = merged_results[-1]
                merged_results[-1] = (SILENCE_TOKEN, s_start, end_timestamp)
            else:
                # otherwise (or on an empty list) start a new silence segment
                merged_results.append((SILENCE_TOKEN, current_start_time, end_timestamp))

            # The next segment may start where this silence ends
            current_start_time = end_timestamp

        else:
            # Case 2: ordinary text
            text_buffer.append(token)

        # Whatever the token was, advance the last-known time
        last_end_time = end_timestamp

    # Wind-down after the loop
    # Text still in the buffer means the last segment was speech, not silence, and must be saved
    if text_buffer:
        full_text = "".join(text_buffer)
        merged_results.append((full_text, current_start_time, last_end_time))

    return merged_results


def calculate_and_print_wer(gt_file_path, asr_json_path, dataset_type, dataset_obj=None, print_threshold=0.3):
    # 1. Parse the ground-truth data
    if dataset_type == "switchboard":
        gt_data = SwitchboardDataset._parse_gt_file(gt_file_path)
        # Concatenate all the ground-truth text
        ref_list = [SwitchboardDataset._clean_gt_text(item[1]) for item in gt_data]
    elif dataset_type == "fisher":
        filename = os.path.basename(asr_json_path)
        # Filenames look like fe_03_00001A.json or fe_03_00001-A.json
        role = "A" if "A.json" in filename else "B"

        full_gt_data = dataset_obj._parse_fisher_trans_file(gt_file_path)
        # Keep the text of the matching speaker and concatenate it
        ref_list = [item[1] for item in full_gt_data if item[0] == role]
    reference_text = " ".join(ref_list)

    # 2. Parse the ASR data
    with open(asr_json_path, "r", encoding="utf-8") as f:
        raw_asr_segments = json.load(f)

    # Preprocess with merge_asr_segments to join the fragments
    merged_segments = merge_asr_segments(raw_asr_segments)

    # Extract the text, ignoring the SILENCE_TOKEN segments
    hyp_list = [seg[0] for seg in merged_segments if seg[0] != SILENCE_TOKEN]
    hypothesis_text = " ".join(hyp_list)

    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
        ]
    )
    hypothesis_text = transformation(hypothesis_text)
    reference_text = transformation(reference_text)

    # 4. Compute the WER
    wer_score = jiwer.wer(reference_text, hypothesis_text)

    # Print part of the comparison when debugging
    if wer_score > print_threshold:
        if random.random() < 0.01:
            print(f"{asr_json_path=}")
            print(f"{gt_file_path=}")
            print(f"Reference (First 400 chars clean): {reference_text[:400]}...")
            print(f"Hypothesis (First 400 chars clean): {hypothesis_text[:400]}...")
            print(f"Final WER: {wer_score:.2%}\n")

    return wer_score


def worker_process(gpu_id, task_queue, result_queue, model_config):
    """
    Worker process logic:
    1. Initializes the model on the assigned GPU (once).
    2. Fetches tasks (audio files) from the queue and processes them.
    """
    device = f"cuda:{gpu_id}"
    logger.info(f"Initializing worker on {device}...")

    try:
        # 1. Initialize Worker and Load Model (Once per process)
        if model_config["asr_method"] == "simul":
            worker = FSMSimulWhisperASRWorker(
                model_path=model_config["model_path"],
                device=device,
                vac_chunk_size=model_config["vac_chunk_size"],
                chunk_duration=model_config["chunk_duration"],
                fit_chunk_duration=model_config["fit_chunk_duration"],
                add_silence_token=model_config["add_silence_token"],
                silence_token_dur=model_config["silence_token_dur"],
            )
        elif model_config["asr_method"] == "faster":
            worker = FSMFasterWhisperASRWorker(
                model_path=model_config["model_path"],
                device=device,
                vad_pause_dur=model_config["vad_pause_dur"],
                min_speech_dur=model_config["min_speech_dur"],
                chunk_duration=model_config["chunk_duration"],
                fit_chunk_duration=model_config["fit_chunk_duration"],
                add_silence_token=model_config["add_silence_token"],
                silence_token_dur=model_config["silence_token_dur"],
                simulate_realtime=model_config["simulate_realtime"],
                use_context_prompt=model_config["use_context_prompt"],
            )

        worker.load_model()
        logger.info(f"Model loaded successfully on {device}. Waiting for tasks...")

        # 2. Process Tasks from Queue
        while True:
            try:
                # Get a task with a timeout to allow graceful exit if queue is empty/stuck
                task = task_queue.get(timeout=3)
            except Empty:
                logger.info(f"Worker on {device} finished all tasks (Queue empty).")
                break

            audio_path, output_path = task

            logger.info(f"Processing: {os.path.basename(audio_path)} -> {device}")

            try:
                # Run the simulation using the method defined in asr_worker.py
                worker.run_file_simulation_and_save(audio_path, output_path)
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
            finally:
                # Signal completion on result_queue whether or not the job succeeded,
                # so the tqdm bar in the parent process advances instead of stalling
                result_queue.put(1)

    except Exception as e:
        logger.error(f"Worker process CRASHED on {device}: {e}", exc_info=True)


def evaluate_dataset_quality(output_folder, gt_folder, dataset_type="switchboard", threshold=0.35, save_bad_files_path=None):
    """
    Walk the output directory, match each file against its ground truth and compute the WER.
    Any file_id whose WER exceeds the threshold is written to the JSON at save_bad_files_path.

    Args:
        output_folder: directory of ASR JSON results
        gt_folder: directory of ground-truth transcripts
        threshold: WER above which a file counts as poor quality
        save_bad_files_path: JSON path where the list of poor-quality file ids is saved
    """
    logger.info(f"STARTING {dataset_type.upper()} QUALITY EVALUATION")

    json_files = glob.glob(os.path.join(output_folder, "*.json"))
    from audio_dataset import SwitchboardDataset, FisherDataset

    if dataset_type == "fisher":
        ds = FisherDataset(gt_folder, output_folder, add_punctuation_to_self=False)
        gt_pattern = "*.txt"
        # Matches fe_03_00001B.json; group 1 is the id and group 2 the speaker
        hypo_id_pattern = re.compile(r"(fe_03_\d{5})([AB])")
        # The ground-truth file is named fe_03_00001.txt
        ref_id_pattern = re.compile(r"(fe_03_\d{5})")
    elif dataset_type == "switchboard":
        ds = SwitchboardDataset(gt_folder, output_folder, add_punctuation_to_self=False)
        gt_pattern = "sw*-ms98-a-trans.text"
        # Matches sw02001A.json, taking "2001A" as the unique key
        hypo_id_pattern = re.compile(r"sw0(\d{4}[AB])")
        # Matches sw2001A-ms98-a-trans.text, taking "2001A"
        ref_id_pattern = re.compile(r"sw(\d{4}[AB])")
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        return

    # 1. Index the ground-truth files
    logger.info(f"Indexing GT files in {gt_folder}")
    all_gt_paths = glob.glob(os.path.join(gt_folder, "**", gt_pattern), recursive=True)
    gt_map = {}
    for p in all_gt_paths:
        match = ref_id_pattern.search(os.path.basename(p))
        if match:
            gt_map[match.group(1)] = p

    # Containers for the statistics
    all_wer_scores = []
    good_wer_scores = []
    bad_wer_scores = []
    high_wer_ids = []
    missing_gt_count = 0

    # 2. Walk the ASR results
    for json_path in tqdm(json_files, desc="Evaluating"):
        filename = os.path.basename(json_path)
        match = hypo_id_pattern.search(filename)

        if not match or match.group(1) not in gt_map:
            missing_gt_count += 1
            continue

        if dataset_type == "fisher":
            call_id = match.group(1)  # fe_03_00001
            speaker_role = match.group(2)  # A or B
        else:
            call_id = match.group(1)  # 2001A
            speaker_role = None

        if call_id not in gt_map:
            continue
        gt_path = gt_map[call_id]

        # A single call site, with the arguments chosen by dataset_type
        wer = calculate_and_print_wer(
            gt_path, json_path, dataset_type=dataset_type, dataset_obj=ds if dataset_type == "fisher" else None, print_threshold=threshold
        )

        all_wer_scores.append(wer)
        if wer > threshold:
            bad_wer_scores.append(wer)
            high_wer_ids.append(call_id)
        else:
            good_wer_scores.append(wer)

    # 3. Save the results
    if save_bad_files_path:
        if high_wer_ids:
            unique_bad_ids = sorted(list(set(high_wer_ids)))
            with open(save_bad_files_path, "w") as f:
                json.dump(unique_bad_ids, f, indent=4)
            logger.info(f"Saved {len(unique_bad_ids)} bad case IDs to {save_bad_files_path}")
        if all_wer_scores:
            save_wer_dist_path = save_bad_files_path.replace(".json", "_distribution.pdf").replace("bad_", "wer_")
            plot_wer_distribution(all_wer_scores, save_wer_dist_path)

    # 4. Print the statistics
    total_count = len(all_wer_scores)
    if total_count > 0:
        num_good = len(good_wer_scores)
        num_bad = len(bad_wer_scores)

        avg_all = sum(all_wer_scores) / total_count
        avg_good = sum(good_wer_scores) / num_good if num_good > 0 else 0
        avg_bad = sum(bad_wer_scores) / num_bad if num_bad > 0 else 0

        logger.info("\n" + "=" * 50)
        logger.info("DATASET QUALITY SUMMARY")
        logger.info("-" * 50)
        logger.info(f"Total Files Evaluated: {total_count}")
        logger.info(f"Missing GT Files:      {missing_gt_count}")
        logger.info(f"Threshold (WER):       {threshold:.2%}")
        logger.info("-" * 50)
        logger.info(f"GOOD Files: {num_good:<6} | Proportion: {num_good / total_count:>7.2%} | Avg WER: {avg_good:>7.2%}")
        logger.info(f"BAD  Files: {num_bad:<6} | Proportion: {num_bad / total_count:>7.2%} | Avg WER: {avg_bad:>7.2%}")
        logger.info("-" * 50)
        logger.info(f"Overall Average WER: {avg_all:.2%}")
        logger.info("=" * 50 + "\n")
    else:
        logger.warning("No files were successfully evaluated.")


def plot_wer_distribution(wer_scores, save_path):
    """
    WER distribution plot, tuned for the 0-1 range.
    A restricted main range plus summary text makes the dense region easier to read.
    """
    # Convert to a numpy array for the statistics
    scores = np.array(wer_scores)

    # 1. Clip extreme outliers to fix the main plotting range (show up to ~1.2 even if some
    #    values reach 6.0), noting in the figure that some data lies outside it
    plot_limit = 1.2  # a little headroom above the 0-1 region of interest
    filtered_scores = scores[scores <= plot_limit]
    outlier_count = np.sum(scores > plot_limit)

    # Publication style
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.figure(figsize=(10, 6))

    # 2. Plot the distribution
    # 50 bins bring out the structure between 0 and 1
    ax = sns.histplot(filtered_scores, kde=True, color="#34495e", bins=50, edgecolor="white", line_kws={"linewidth": 2.5})

    # 3. Reference lines for the mean and the median
    median_wer = np.median(scores)
    plt.axvline(median_wer, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {median_wer:.2%}")

    # 4. Annotate the outliers in the figure
    if outlier_count > 0:
        plt.text(
            0.95,
            0.95,
            f"Excluded outliers (>1.2): {outlier_count}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    # 5. Tidy the axes
    plt.xlim(0, plot_limit)
    plt.xticks(np.arange(0, 1.3, 0.2))  # fixed tick positions

    plt.title("Detailed Distribution of Word Error Rate (WER)", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Word Error Rate (WER)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.legend()

    # Drop the top and right spines
    sns.despine()

    plt.tight_layout()

    plt.savefig(save_path, format="pdf", dpi=300)
    plt.close()

    logger.info(f"Detailed WER plot saved to: {save_path} (Outliers: {outlier_count})")


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Distributed ASR Batch Processing")
    parser.add_argument("--rank", type=int, default=0, help="Current server ID, 0-based")
    parser.add_argument("--world-size", type=int, default=1, help="The number of servers")
    parser.add_argument("--dataset-type", type=str, default="fisher", help="Dataset name")  # switchboard or fisher
    parser.add_argument("--asr-method", type=str, default="faster", help="ASR method")  # "faster" or "simul"
    parser.add_argument("--postfix", type=str, default="_setting", help="Postfix and hyperparameters")
    parser.add_argument("--threshold", type=float, default=0.3, help="WER threshold for bad files")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of worker processes per GPU")
    args = parser.parse_args()

    dataset_type = args.dataset_type
    asr_method = args.asr_method
    postfix = args.postfix
    threshold = args.threshold

    if dataset_type == "switchboard":
        input_folder = expand("${SWITCHBOARD_ROOT}/audio_wav/")
        gt_trans_folder = expand("${SWITCHBOARD_ROOT}/transcripts/")
    elif dataset_type == "fisher":
        input_folder = expand("${FISHER_ROOT}/fe_03_audio")
        gt_trans_folder = expand("${FISHER_ROOT}/fe_03_ori")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Path to the folder where JSON transcripts will be saved
    output_folder = expand(f"${{DATA_ROOT}}/{dataset_type}_asr_{asr_method}{postfix}")
    logger.info(f"Output folder: {output_folder}")

    # ASR Worker Configuration
    if asr_method == "simul":
        if postfix == "_setting":
            model_config = {
                "asr_method": asr_method,
                "model_path": expand("${WHISPER_MODEL_PATH}"),
                "vac_chunk_size": 0.08,
                "chunk_duration": SILENCE_TOKEN_DUR,
                "fit_chunk_duration": True,
                "add_silence_token": True,
                "silence_token_dur": SILENCE_TOKEN_DUR,
            }
    elif asr_method == "faster":
        if postfix == "_setting":
            # The setting used for the released tapes
            model_config = {
                "asr_method": "faster",
                "model_path": "deepdml/faster-whisper-large-v3-turbo-ct2",
                "vad_pause_dur": 0.032,
                "min_speech_dur": 0.16,
                "chunk_duration": 0.032,
                "fit_chunk_duration": True,
                "add_silence_token": True,
                "silence_token_dur": SILENCE_TOKEN_DUR,
                "simulate_realtime": False,
                "use_context_prompt": False,
            }
        elif postfix == "_setting1":
            # Variant with a longer VAD pause threshold, i.e. coarser IPU segmentation
            model_config = {
                "asr_method": "faster",
                "model_path": "deepdml/faster-whisper-large-v3-turbo-ct2",
                "vad_pause_dur": 0.16,
                "min_speech_dur": 0.16,
                "chunk_duration": 0.032,
                "fit_chunk_duration": True,
                "add_silence_token": True,
                "silence_token_dur": SILENCE_TOKEN_DUR,
                "simulate_realtime": False,
                "use_context_prompt": False,
            }

    # 1. Setup Environment
    os.makedirs(output_folder, exist_ok=True)

    # IMPORTANT: 'spawn' is required for CUDA multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Context might be already set

    # 2. Scan Audio Files
    audio_files = []
    # Add extensions as needed
    audio_files.extend(glob.glob(os.path.join(input_folder, "*.wav")))
    # Sort files to ensure deterministic order before parallel processing
    audio_files.sort()

    total_files_count = len(audio_files)

    # Data sharding
    # Strided slicing, list[start:end:step]. With three machines:

    #   rank 0 takes 0, 3, 6, 9, ...
    #   rank 1 takes 1, 4, 7, 10, ...
    #   rank 2 takes 2, 5, 8, 11, ...
    my_files = audio_files[args.rank :: args.world_size]

    logger.info(f"Global task info: Total files {total_files_count} | World Size {args.world_size}")
    logger.info(f"Local task info:  Rank {args.rank} processing {len(my_files)} files.")

    # 3. Populate the Task Queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    skipped_count = 0
    queued_count = 0

    for audio_file in my_files:
        filename = os.path.basename(audio_file)
        # Change extension to .json
        name_without_ext = os.path.splitext(filename)[0]
        output_json = os.path.join(output_folder, f"{name_without_ext}.json")

        # Check if output already exists
        if os.path.exists(output_json):
            skipped_count += 1
            continue

        task_queue.put((audio_file, output_json))
        queued_count += 1

    logger.info(f"[Rank {args.rank}] Queue Summary: Queued {queued_count} | Skipped {skipped_count}")

    # If nothing to process, exit early
    if queued_count == 0:
        logger.info(f"[Rank {args.rank}] All assigned files processed. Exiting.")
    else:
        # 4. Launch Worker Processes
        gpu_count = get_gpu_count()
        processes = []

        # Launch one process per GPU
        logger.info(f"[Rank {args.rank}] Starting {gpu_count} GPU workers...")
        for gpu_id in range(gpu_count):
            # Start several processes per GPU
            for w_id in range(args.workers_per_gpu):
                p = mp.Process(
                    target=worker_process,
                    args=(
                        gpu_id,  # several processes share one gpu_id
                        task_queue,
                        result_queue,
                        model_config,
                    ),
                    # Name the process to make debugging easier
                    name=f"Worker-GPU{gpu_id}-{w_id}",
                )
                p.start()
                processes.append(p)

        logger.info(f"[Rank {args.rank}] Total {len(processes)} worker processes launched.")

        # 5. Wait for Completion
        check_interval = 50  # re-check the WER every 50 files

        with tqdm(total=queued_count, desc=f"ASR Processing (Rank {args.rank})", unit="file") as pbar:
            for i in range(1, queued_count + 1):
                result_queue.get()
                pbar.update(1)

                # Check every check_interval files, and again on the last file
                if i % check_interval == 0 or i == queued_count:
                    tqdm.write(f"\n[Periodic Check] Completed {i}/{queued_count}. Calculating intermediate WER...")

                    # Run the evaluation
                    evaluate_dataset_quality(
                        output_folder=output_folder,
                        gt_folder=gt_trans_folder,
                        threshold=threshold,
                        dataset_type=dataset_type,
                        # Intermediate checks do not save bad_files, so the final result is not overwritten
                        save_bad_files_path=None,
                    )

        for p in processes:
            p.join()

        logger.info(f"[Rank {args.rank}] Processing complete.")

    evaluate_dataset_quality(
        output_folder=output_folder,
        gt_folder=gt_trans_folder,
        threshold=threshold,
        dataset_type=dataset_type,
        save_bad_files_path=expand(f"${{DATA_ROOT}}/broken_asr_and_wer/bad_asr_{dataset_type}_{asr_method}{postfix}_threshold{threshold}.json"),
    )


if __name__ == "__main__":
    main()
