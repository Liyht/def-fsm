#!/usr/bin/env python3
"""Bidirectional interruption evaluation for a DEF-FSM checkpoint.

Follows the NFSM protocol on two synthetic test sets built by the data pipeline
(`bash scripts/prepare_data.sh nfsm-tape`):

  Machine-interrupts-User (MiU)  The user's closing statement carries a deliberate
      commonsense error. The tape is replayed position by position and we take the
      response at the first point where the model proactively emits [S.SPEAK]
      instead of [C.LISTEN] -- that is, where it decides to interrupt.
  User-interrupts-Machine (UiM)  The user interrupts mid-response for one of four
      reasons (denial, affirmation, environmental noise, topic shift). We take the
      first position where the model yields the floor with [S.LISTEN...], then
      force a response there to judge what it says next.

Both are scored by an LLM judge with structured output. The judge is a different
model family from the one that generated the test data, to avoid egocentric bias.

Reported metrics:

  MiU_F1        the headline MiU figure: harmonic mean of Precision and Recall,
                where Precision = PIR_mid * ir_mid + ir_end and Recall = 1 - MIR
  UiM_PRR_avg   mean Proper Response Rate, plus a per-reason breakdown

    python evaluation/fsm_bench/fsm_bench.py --model <checkpoint> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import string
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from def_fsm.paths import PROJECT_ROOT, expand
from def_fsm.utils import (
    INTERLOCUTOR_PREFIX,
    SILENCE_TOKEN,
    STATE_TRANSITION_TOKENS,
    USER_PREFIX,
    system_prompt_fillin,
)


class MiUJudgeOutput(BaseModel):
    analysis_for_assistant_interruption: str = Field(description="Analysis of the appropriateness of the interruption timing and the content")
    score_for_interruption: int = Field(description="0 for inappropriate, 1 for appropriate")
    score_for_content: int = Field(description="0 for inappropriate, 1 for appropriate")


class UiMJudgeOutput(BaseModel):
    analysis_for_user_interruption: str = Field(
        description="Analysis of the user's interruption intent and the appropriateness of the assistant's response"
    )
    score_for_assistant_last_content: int = Field(description="0 for inappropriate, 1 for appropriate")


def load_tape_data(file_paths):
    """Load the tape test sets, tagging each item with the file it came from."""
    all_data = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["source_file"] = os.path.basename(path)
        all_data.extend(data)
    return all_data


def remove_interlocutor_punctuation(tape):
    """Strip the trailing punctuation of user chunks.

    A perception module emits no punctuation, so the test tapes -- which are
    LLM-written and fully punctuated -- would otherwise cue the model with a
    sentence boundary it would never see at inference time.
    """
    new_tape = []
    for text in tape:
        text = text.strip()
        if text not in STATE_TRANSITION_TOKENS and text.startswith(INTERLOCUTOR_PREFIX) and text[-1] in string.punctuation:
            new_tape.append(text[:-1])
        else:
            new_tape.append(text)
    return new_tape


def clear_vllm_memory(llm_instance):
    """Release the generator before the judge is loaded; both are too large to co-reside."""
    del llm_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_dialogue_history(tape_slice, generated_response):
    """Render a tape as the plain 'USER: ... ASSISTANT: ...' transcript the judge reads.

    State tokens are dropped and consecutive turns by the same speaker are merged,
    so the judge sees an ordinary dialogue and never the tape machinery.
    """
    history_lines = []

    for turn in tape_slice + [generated_response]:
        # Remove the state tokens first, without disturbing the surrounding spacing.
        for token in STATE_TRANSITION_TOKENS | {SILENCE_TOKEN}:
            turn = turn.replace(token, "")
        if not turn.strip():
            continue

        if INTERLOCUTOR_PREFIX in turn:
            speaker, text = "USER", turn.replace(INTERLOCUTOR_PREFIX, "")
        else:
            speaker, text = "ASSISTANT", turn

        if history_lines and history_lines[-1].startswith(speaker):
            # Same speaker continuing: append, adding a space only if neither side has one.
            if not history_lines[-1].endswith(" ") and not text.startswith(" "):
                history_lines[-1] += " " + text
            else:
                history_lines[-1] += text
        else:
            history_lines.append(f"{speaker}:{text}" if text.startswith(" ") else f"{speaker}: {text}")

    return "\n".join(history_lines)


def extract_scores(judge_output, is_miu):
    """Parse one judge verdict. Structured output guarantees the schema."""
    parsed = json.loads(judge_output)
    if is_miu:
        return {
            "score_for_interruption": parsed.get("score_for_interruption"),
            "score_for_content": parsed.get("score_for_content"),
            "analysis": parsed.get("analysis_for_assistant_interruption"),
        }
    return {
        "score_for_assistant_last_content": parsed.get("score_for_assistant_last_content"),
        "analysis": parsed.get("analysis_for_user_interruption"),
    }


def build_force_response_tape(tape, target_idx):
    """Build the UiM probe tape: accept the interruption at target_idx, then ask to speak.

    The transition at the interruption point is forced to [S.LISTEN.INTERRUPT] and
    every later transition is pinned to [C.LISTEN], so the model stays listening
    through the rest of the user's turn. A final [S.SPEAK] then elicits the reply
    that gets judged.
    """
    forced = tape[:]
    forced[target_idx] = "[S.LISTEN.INTERRUPT]"
    for i in range(target_idx + 1, len(forced)):
        if forced[i] in STATE_TRANSITION_TOKENS:
            forced[i] = "[C.LISTEN]"
    if forced[-1] != "[S.LISTEN.INTERRUPT]":
        forced[-1] = "[S.SPEAK]"
    else:
        forced += [SILENCE_TOKEN, "[S.SPEAK]"]
    return forced


def run_generation_phase(args, raw_data, system_prompt):
    """Replay every tape from each candidate interruption point and collect the responses."""
    eval_tasks, prompts, pending = [], [], []

    for item in raw_data:
        tape = item["tape"]
        item["responses_dict"] = {}
        is_uim = "UiM" in item["id"]
        if is_uim:
            item["force_responses_dict"] = {}

        for target_idx in item["eval_targets"]:
            prefix = "".join(tape[:target_idx]).replace(INTERLOCUTOR_PREFIX, USER_PREFIX)
            task = {
                "id": item["id"],
                "source_file": item["source_file"],
                "target_idx": target_idx,
                "tape_slice": tape[:target_idx],
            }
            prompts.append(f"{system_prompt}{prefix}")
            pending.append(task)
            eval_tasks.append(task)

            if is_uim:
                forced = build_force_response_tape(tape, target_idx)
                prefix_forced = "".join(forced).replace(INTERLOCUTOR_PREFIX, USER_PREFIX)
                task_forced = {
                    "id": item["id"] + "_force_response",
                    "source_file": item["source_file"],
                    "target_idx": target_idx,
                    "tape_slice": forced,
                }
                prompts.append(f"{system_prompt}{prefix_forced}")
                pending.append(task_forced)
                eval_tasks.append(task_forced)

    logger.info(f"Generating {len(prompts)} responses with {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        max_tokens=args.fsm_max_tokens,
        temperature=0.0,
        stop=["[C.LISTEN]", "[S.LISTEN.NATURAL]", "[S.LISTEN.INTERRUPT]"],
        include_stop_str_in_output=True,  # the stop token IS the decision being measured
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    item_map = {(item["source_file"], item["id"]): item for item in raw_data}
    for task, output in zip(pending, outputs):
        text = output.outputs[0].text.strip()

        if not task["id"].endswith("_force_response"):
            # The model occasionally emits stray punctuation before the transition;
            # start from the earliest state token so the decision is read cleanly.
            positions = [i for i in (text.find(t) for t in STATE_TRANSITION_TOKENS) if i != -1]
            if positions:
                text = text[min(positions) :]

        task["generated_response"] = text
        if task["id"].endswith("_force_response"):
            item = item_map[(task["source_file"], task["id"].replace("_force_response", ""))]
            item["force_responses_dict"][task["target_idx"]] = text
        else:
            item_map[(task["source_file"], task["id"])]["responses_dict"][task["target_idx"]] = text

    clear_vllm_memory(llm)

    for item in raw_data:
        targets = item["eval_targets"]
        item["responses"] = [item["responses_dict"][t] for t in targets if t in item["responses_dict"]]
        if "force_responses_dict" in item:
            item["force_responses"] = [item["force_responses_dict"][t] for t in targets if t in item["force_responses_dict"]]

    return raw_data, eval_tasks


def filter_eval_tasks(raw_data, eval_tasks):
    """Keep the first position where the model actually acted, and tally hits and misses.

    MiU counts as acting when the model does not simply continue listening; UiM when
    it yields the floor. Only the earliest such position per dialogue is judged, and
    for UiM it is the forced-response probe that gets judged rather than the
    transition itself.
    """
    candidates = defaultdict(dict)
    force_tasks = defaultdict(dict)
    hits = set()

    for task in eval_tasks:
        acted = False
        if "MiU" in task["id"]:
            acted = task["generated_response"] != "[C.LISTEN]"
        elif "UiM" in task["id"]:
            if task["id"].endswith("_force_response"):
                force_tasks[task["id"]][task["target_idx"]] = task
                continue
            acted = task["generated_response"].startswith("[S.LISTEN")
        if acted:
            hits.add((task["id"], task["target_idx"]))
            candidates[task["id"]][task["target_idx"]] = task

    filtered = []
    for item_id, by_idx in candidates.items():
        first = min(by_idx)
        if "MiU" in item_id:
            filtered.append(by_idx[first])
        else:
            force_id = item_id + "_force_response"
            if first in force_tasks.get(force_id, {}):
                filtered.append(force_tasks[force_id][first])

    # Where the model acted: mid-utterance, only at the end, or never.
    statistics = {}
    for item in raw_data:
        targets = item["eval_targets"]
        mid_hit = sum(1 for t in targets[:-1] if (item["id"], t) in hits)
        miss = len(targets[:-1]) - mid_hit
        end_hit = 1 if targets and (item["id"], targets[-1]) in hits else 0
        if not end_hit:
            miss += 1
        statistics[item["id"]] = {"miss": miss, "mid_hit": mid_hit, "end_hit": end_hit}

    return filtered, statistics


def analyze_statistics(statistics):
    """Collapse the per-position tallies to one outcome per dialogue.

    The three outcomes are made mutually exclusive by priority -- an interruption
    mid-utterance takes precedence over one at the end -- so the rates sum to 1 and
    can be used directly in the precision formula.
    """
    summary = {}
    for data_type in ["MiU", "UiM"]:
        total = mid = end = miss = 0
        for item_id, stat in statistics.items():
            if data_type not in item_id:
                continue
            total += 1
            if stat["mid_hit"] > 0:
                mid += 1
            elif stat["end_hit"] > 0:
                end += 1
            else:
                miss += 1
        summary[data_type] = {"total": total, "mid_hits": mid, "end_hits": end, "miss": miss}
        if total:
            logger.info(f"{data_type}: n={total} mid={mid / total:.4f} end={end / total:.4f} miss={miss / total:.4f}")
    return summary


def run_evaluation_phase(args, raw_data, filtered_eval_tasks):
    """Score the selected responses with the LLM judge."""
    miu_template = Path(args.judge_prompt_miu).read_text(encoding="utf-8").strip()
    uim_template = Path(args.judge_prompt_uim).read_text(encoding="utf-8").strip()

    messages_list, is_miu_list = [], []
    for task in filtered_eval_tasks:
        history = format_dialogue_history(task["tape_slice"], task["generated_response"])
        is_miu = "MiU" in task["source_file"]
        template = miu_template if is_miu else uim_template
        messages_list.append([{"role": "user", "content": template.replace("{dialogue_history}", history)}])
        is_miu_list.append(is_miu)

    logger.info(f"Judging {len(messages_list)} responses with {args.judge_model}")
    judge = LLM(model=args.judge_model, tensor_parallel_size=args.tensor_parallel_size, reasoning_parser="qwen3")

    miu_params = StructuredOutputsParams(json=MiUJudgeOutput.model_json_schema())
    uim_params = StructuredOutputsParams(json=UiMJudgeOutput.model_json_schema())
    sampling_params_list = [
        SamplingParams(
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
            structured_outputs=miu_params if is_miu else uim_params,
        )
        for is_miu in is_miu_list
    ]

    judge_outputs = judge.chat(
        messages=messages_list,
        sampling_params=sampling_params_list,
        chat_template_kwargs={"enable_thinking": False},
    )

    for task, output, is_miu in zip(filtered_eval_tasks, judge_outputs, is_miu_list):
        text = output.outputs[0].text.strip()
        try:
            task["eval_score_record"] = {
                "target_idx": task["target_idx"],
                "raw_judge_output": text,
                "parsed_scores": extract_scores(text, is_miu),
            }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Unparseable judge output for {task['id']}: {e}")

    clear_vllm_memory(judge)

    item_map = {(item["source_file"], item["id"]): item for item in raw_data}
    for item in raw_data:
        item["eval_scores"] = []
    for task in filtered_eval_tasks:
        if "eval_score_record" in task:
            item = item_map[(task["source_file"], task["id"].replace("_force_response", ""))]
            item["eval_scores"].append(task["eval_score_record"])

    return raw_data


def aggregate_metrics(final_data, stats_summary, output_dir):
    """Compute the reported metrics from the judge scores and the hit statistics."""
    miu_pir, miu_prr, uim_all = [], [], []
    uim_by_reason = defaultdict(list)

    for item in final_data:
        item_id = item.get("id", "")
        is_miu = "MiU" in item_id or "MiU" in item.get("source_file", "")

        for entry in item.get("eval_scores", []):
            parsed = entry.get("parsed_scores", {})
            if is_miu:
                # Only interruptions before the user finished speaking are rated;
                # acting at the last position is ordinary turn-taking, not an interruption.
                if entry.get("target_idx") != item["eval_targets"][-1]:
                    if parsed.get("score_for_interruption") is not None:
                        miu_pir.append(parsed["score_for_interruption"])
                    if parsed.get("score_for_content") is not None:
                        miu_prr.append(parsed["score_for_content"])
            else:
                score = parsed.get("score_for_assistant_last_content")
                if score is not None:
                    uim_all.append(score)
                    # Ids look like "test_UiM_<reason>_<n>".
                    parts = item_id.split("_")
                    uim_by_reason[parts[2] if len(parts) > 2 else "unknown"].append(score)

    metrics, counts = {}, {}

    def record(name, scores):
        if scores:
            metrics[name] = sum(scores) / len(scores)
            counts[name] = len(scores)

    record("MiU_PIR_mid", miu_pir)
    record("MiU_PRR_mid", miu_prr)
    record("UiM_PRR_avg", uim_all)
    for reason, scores in uim_by_reason.items():
        record(f"UiM_PRR_{reason}", scores)

    miu = stats_summary["MiU"]
    if miu["total"]:
        ir_mid = miu["mid_hits"] / miu["total"]
        ir_end = miu["end_hits"] / miu["total"]
        mir = miu["miss"] / miu["total"]

        # An interruption mid-utterance counts only if the judge deemed it proper;
        # acting at the end is always proper, since the user had finished.
        precision = metrics.get("MiU_PIR_mid", 0.0) * ir_mid + ir_end
        recall = 1.0 - mir

        metrics.update({"MiU_ir_mid": ir_mid, "MiU_ir_end": ir_end, "MiU_MIR": mir})
        metrics["MiU_Precision"] = precision
        metrics["MiU_Recall"] = recall
        metrics["MiU_F1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for k in ["MiU_ir_mid", "MiU_ir_end", "MiU_MIR", "MiU_Precision", "MiU_Recall", "MiU_F1"]:
            counts[k] = miu["total"]

    uim = stats_summary["UiM"]
    if uim["total"]:
        metrics["UiM_ir_mid"] = uim["mid_hits"] / uim["total"]
        metrics["UiM_ir_end"] = uim["end_hits"] / uim["total"]
        metrics["UiM_MIR"] = uim["miss"] / uim["total"]
        for k in ["UiM_ir_mid", "UiM_ir_end", "UiM_MIR"]:
            counts[k] = uim["total"]

    logger.info("---------- Metrics ----------")
    for k, v in sorted(metrics.items()):
        logger.info(f"{k}: {v:.4f} (N={counts[k]})")
    logger.info(f"Reported: MiU F1 = {metrics.get('MiU_F1', float('nan')):.4f} | "
                f"UiM PRR = {metrics.get('UiM_PRR_avg', float('nan')):.4f}")

    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {metrics_path}")
    return metrics


def save_responses(final_data, output_dir):
    """Write the per-dialogue responses and judge verdicts, for inspecting a run."""
    by_file = defaultdict(list)
    for item in final_data:
        record = {
            "id": item["id"],
            "tape": item["tape"],
            "eval_targets": item["eval_targets"],
            "responses": item.get("responses", []),
        }
        if "force_responses" in item:
            record["force_responses"] = item["force_responses"]
        if "eval_scores" in item:
            record["eval_scores"] = item["eval_scores"]
        by_file[item["source_file"]].append(record)

    for source_file, records in by_file.items():
        path = Path(output_dir) / source_file.replace(".json", "_with_eval.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} records to {path}")


def main() -> None:
    ap = argparse.ArgumentParser("Bidirectional interruption evaluation for DEF-FSM")
    ap.add_argument("--model", required=True, help="Hugging Face id or local directory of the checkpoint to evaluate.")
    ap.add_argument("--output-dir", required=True, help="Where metrics.json and the per-dialogue records are written.")
    ap.add_argument("--tape-files", nargs="+", default=None,
                    help="MiU and UiM tape files. Default: the two written by scripts/prepare_data.sh nfsm-tape.")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-27B",
                    help="LLM judge. A different family from the model that wrote the test data, to avoid egocentric bias.")
    ap.add_argument("--system-prompt", default=str(PROJECT_ROOT / "prompts/fsm_assistant.txt"))
    ap.add_argument("--judge-prompt-miu", default=str(PROJECT_ROOT / "prompts/MiU_judge.txt"))
    ap.add_argument("--judge-prompt-uim", default=str(PROJECT_ROOT / "prompts/UiM_judge.txt"))
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--fsm-max-tokens", type=int, default=1024)
    ap.add_argument("--judge-max-tokens", type=int, default=8192)
    args = ap.parse_args()

    if args.tape_files is None:
        args.tape_files = [expand("${DATA_ROOT}/FSM/test_MiU_tape.json"), expand("${DATA_ROOT}/FSM/test_UiM_tape.json")]
    os.makedirs(args.output_dir, exist_ok=True)

    raw_data = load_tape_data(args.tape_files)
    for item in raw_data:
        item["tape"] = remove_interlocutor_punctuation(item["tape"])

    system_prompt = system_prompt_fillin(args.system_prompt, USER_PREFIX)
    raw_data, eval_tasks = run_generation_phase(args, raw_data, system_prompt)

    filtered_eval_tasks, statistics = filter_eval_tasks(raw_data, eval_tasks)
    stats_summary = analyze_statistics(statistics)

    final_data = run_evaluation_phase(args, raw_data, filtered_eval_tasks)
    save_responses(final_data, args.output_dir)
    aggregate_metrics(final_data, stats_summary, args.output_dir)


if __name__ == "__main__":
    main()
