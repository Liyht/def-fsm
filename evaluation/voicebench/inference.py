#!/usr/bin/env python3
"""Generate VoiceBench responses for a DEF-FSM checkpoint.

Runs the two halves of the cascade separately, because VoiceBench is an offline
benchmark: every clip is transcribed first (the perception module, one worker per
GPU), then all transcripts are answered in one vLLM batch. The FSM's real-time
coordination plays no part here -- what is being measured is whether the tape
format and the fine-tuning cost the model any spoken-language ability.

Two prompting modes:

  fsm   The tape prompt the model was trained on. The transcript is laid out as
        the interlocutor's turns, the model is cued with [S.SPEAK], and generation
        stops at the first [S.LISTEN...] transition; the response is then stripped
        of state transition tokens.
  chat  VoiceBench's own prompt, for the untrained baseline.

Both stages cache to disk and resume, which matters because the ASR pass over the
five datasets is the expensive part and is shared across checkpoints.

    python evaluation/voicebench/inference.py --model <hf-id-or-path> --result-dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import torch
from datasets import Audio, load_dataset
from loguru import logger
from tqdm import tqdm
from vllm import LLM, SamplingParams

from def_fsm.paths import PROJECT_ROOT, require
from def_fsm.utils import SILENCE_TOKEN, STATE_TRANSITION_TOKENS, USER_PREFIX, system_prompt_fillin

# The subsets scored by score.py, with the split each one is loaded at.
DATASETS = [
    ("advbench", "test"),
    ("openbookqa", "test"),
    ("sd-qa", "usa"),
    ("mmsu", "law+engineering+other+biology+business+economics+health+philosophy+"
             "psychology+history+chemistry+physics"),
]

CHAT_SYSTEM_PROMPT_AUDIO = (
    "You are a helpful assistant who tries to help answer the user's question. Please note that "
    "the user's query is transcribed from speech, and the transcription may contain errors."
)
CHAT_SYSTEM_PROMPT_TEXT = "You are a helpful assistant who tries to help answer the user's question."


def default_asr_model(asr_type: str) -> str:
    """Resolve the checkpoint for a perception module. WHISPER_MODEL_PATH is only
    required for SimulStreaming, so it is read lazily."""
    if asr_type == "turbo":
        return "openai/whisper-large-v3-turbo"
    if asr_type == "faster":
        return "deepdml/faster-whisper-large-v3-turbo-ct2"
    return require("WHISPER_MODEL_PATH")


class MultiGPUWhisper:
    """One ASR worker per visible GPU, handed out through a queue so a thread pool
    can transcribe in parallel without two threads sharing a device."""

    def __init__(self, asr_type: str, model_id: str):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available; the ASR stage needs at least one.")

        self.asr_type = asr_type
        self.models: dict[str, dict] = {}
        self.gpu_queue: Queue = Queue()
        logger.info(f"Loading {asr_type} ASR '{model_id}' across {self.num_gpus} GPUs...")

        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            self.models[device] = {"type": asr_type, "engine": self._build_engine(asr_type, model_id, device)}
            self.gpu_queue.put(device)
            logger.info(f"{asr_type} ASR loaded on {device}")

    @staticmethod
    def _build_engine(asr_type: str, model_id: str, device: str):
        if asr_type == "turbo":
            # Plain Whisper: one transcript per clip, no turn structure.
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            return pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch.float16,
                device=device,
            )

        if asr_type == "faster":
            from def_fsm.asr_worker import FSMFasterWhisperASRWorker

            worker = FSMFasterWhisperASRWorker(
                model_path=model_id, device=device, add_silence_token=False, simulate_realtime=False
            )
            worker.load_model()
            # The worker normally runs its transcription loop on its own thread; here
            # it is driven by hand, so give it the queue that loop would have read.
            worker.vad_to_asr_queue = Queue()
            worker.reset()
            return worker

        from def_fsm.asr_worker import FSMSimulWhisperASRWorker

        worker = FSMSimulWhisperASRWorker(
            model_path=model_id, device=device, add_silence_token=True, simulate_realtime=False
        )
        worker.load_model()
        return worker

    def transcribe(self, audio_data) -> str | list[str]:
        """Transcribe one clip on whichever GPU is free. Returns a plain string for
        `turbo`, or the FSM worker's list of segments for `faster` / `simul`."""
        device = self.gpu_queue.get()
        try:
            torch.cuda.set_device(device)
            engine = self.models[device]["engine"]

            if self.asr_type == "turbo":
                return engine(audio_data, generate_kwargs={"language": "english"})["text"].strip()

            return self._transcribe_streaming(engine, audio_data)
        finally:
            self.gpu_queue.put(device)

    def _transcribe_streaming(self, engine, audio_data) -> list[str]:
        """Feed a clip through an FSM perception worker in its native chunk size, so
        the segmentation matches what the model saw during training."""
        audio = audio_data["array"].astype(np.float32)
        engine.reset()

        output_queue: Queue = Queue()
        engine.load_state(
            {
                "asr_to_llm_queue": output_queue,
                "stop_event": Event(),
                "pipeline_timer": {"t_asr_output_text": []},
                "llm_state": "LISTEN",
            }
        )

        chunk = engine.chunk_samples
        for offset in range(0, len(audio), chunk):
            segment = audio[offset : offset + chunk]
            engine._process_audio_chunk(segment, is_final=offset + chunk >= len(audio))

            # FasterWhisper's worker queues work for a transcription thread that is
            # not running here, so drain that queue inline.
            if self.asr_type == "faster":
                while not engine.vad_to_asr_queue.empty():
                    item = engine.vad_to_asr_queue.get_nowait()
                    if not item:
                        continue
                    if item.get("type") == "SPEECH":
                        engine._transcribe_audio(*item["args"])
                    elif item.get("type") == "SILENCE":
                        engine._handle_output(SILENCE_TOKEN, 0, 0, *item["args"][2:])

        texts = []
        while not output_queue.empty():
            item = output_queue.get_nowait()
            text = item[0] if isinstance(item, tuple) else item
            if text:
                texts.append(text)
        return texts


def build_prompt(task: dict, modality: str, system_prompt: str | None):
    """Return either a tape prompt (fsm mode) or a chat message list (chat mode)."""
    content = task["transcript"] if modality == "audio" else task["prompt"]

    if system_prompt is None:
        sys_prompt = CHAT_SYSTEM_PROMPT_AUDIO if modality == "audio" else CHAT_SYSTEM_PROMPT_TEXT
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}]

    # A perception module that emits segments gives a list; lay them out as the
    # interlocutor's successive turns. Leading silence carries no content.
    if isinstance(content, list):
        while content and content[0] == SILENCE_TOKEN:
            content = content[1:]
        content = f"[C.LISTEN]{USER_PREFIX}".join(content)
    return f"{system_prompt}{USER_PREFIX}{content}[S.SPEAK]"


def clean_response(text: str) -> str:
    """Cut a tape-mode generation at the transition back to listening and drop any
    state transition tokens the model emitted inside the response."""
    text = text.split("[S.LISTEN")[0].strip() if "[S.LISTEN" in text else text.strip()
    for token in STATE_TRANSITION_TOKENS:
        text = text.replace(token, "")
    return text


def run_asr(all_tasks: list[dict], args, cache_path: Path) -> None:
    """Transcribe every clip, injecting the result into each task. Cached on disk:
    the transcripts depend only on the perception module, not on the checkpoint."""
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    pending = [t for t in all_tasks if t["global_id"] not in cache]

    if pending:
        engine = MultiGPUWhisper(args.asr_type, args.asr_model)
        logger.info(f"Transcribing {len(pending)} clips with {args.asr_type}...")
        with ThreadPoolExecutor(max_workers=engine.num_gpus) as pool:
            futures = {pool.submit(engine.transcribe, t["audio"]): t["global_id"] for t in pending}
            for future in tqdm(as_completed(futures), total=len(pending), desc="ASR", unit="clip"):
                cache[futures[future]] = future.result()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        del engine
        torch.cuda.empty_cache()

    for task in all_tasks:
        task["transcript"] = cache[task["global_id"]]
        task.pop("audio", None)  # the arrays are large and no longer needed


def generate(all_tasks: list[dict], args, system_prompt: str | None) -> None:
    """Answer every task in one vLLM batch, resuming from any cached chunks."""
    done = set()
    for path in glob.glob(str(args.temp_dir / "*.json")):
        try:
            done.update(item["global_id"] for item in json.load(open(path)))
        except (json.JSONDecodeError, KeyError, OSError):
            continue  # a partial chunk is simply re-generated

    remaining = [t for t in all_tasks if t["global_id"] not in done]
    logger.info(f"Tasks: {len(all_tasks)} | cached: {len(done)} | to generate: {len(remaining)}")

    if remaining:
        prompts = [build_prompt(t, args.modality, system_prompt) for t in remaining]
        llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len)

        if system_prompt is not None:
            params = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0,
                stop=["[S.LISTEN.NATURAL]", "[S.LISTEN.INTERRUPT]", "[S.LISTEN]"],
            )
            outputs = llm.generate(prompts, params, use_tqdm=True)
        else:
            params = SamplingParams(max_tokens=args.max_tokens, temperature=0)
            outputs = llm.chat(prompts, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})

        records = []
        for task, prompt, output in zip(remaining, prompts, outputs):
            raw = output.outputs[0].text
            record = {k: v for k, v in task.items() if k != "audio"}
            record["raw_prompt"] = prompt
            record["raw_response"] = raw
            record["response"] = clean_response(raw) if system_prompt is not None else raw
            records.append(record)

        args.temp_dir.mkdir(parents=True, exist_ok=True)
        with open(args.temp_dir / f"chunk_{uuid.uuid4().hex}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    write_results(args)


def write_results(args) -> None:
    """Split the cached generations into the per-dataset jsonl files score.py reads."""
    by_dataset: dict[str, list[dict]] = {}
    for path in glob.glob(str(args.temp_dir / "*.json")):
        for item in json.load(open(path)):
            by_dataset.setdefault(item["dataset"], []).append(item)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    for name, records in by_dataset.items():
        split = "all" if name == "mmsu" else records[0].get("split", "test")
        records.sort(key=lambda r: int(r["global_id"].split("_")[-1]))
        out_path = args.result_dir / f"{name}-{split}-{args.modality}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} records to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser("VoiceBench inference for DEF-FSM")
    ap.add_argument("--model", required=True, help="Hugging Face id or local directory of the checkpoint to evaluate.")
    ap.add_argument("--result-dir", required=True, help="Where the per-dataset jsonl files are written.")
    ap.add_argument("--mode", default="fsm", choices=["fsm", "chat"],
                    help="fsm: the tape prompt the model was trained on. chat: VoiceBench's own prompt.")
    ap.add_argument("--modality", default="audio", choices=["audio", "text"],
                    help="audio runs the perception module first; text feeds the reference prompt directly.")
    ap.add_argument("--asr-type", default="faster", choices=["turbo", "faster", "simul"])
    ap.add_argument("--asr-model", default=None, help="ASR checkpoint; default: matches --asr-type.")
    ap.add_argument("--prompt-path", default=str(PROJECT_ROOT / "prompts/fsm_assistant.txt"))
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--cache-dir", default=None,
                    help="Where ASR transcripts and generation chunks are cached. Default: <result-dir>/../cache.")
    args = ap.parse_args()

    if args.asr_model is None:
        args.asr_model = default_asr_model(args.asr_type)
    args.result_dir = Path(args.result_dir)
    cache_root = Path(args.cache_dir) if args.cache_dir else args.result_dir.parent / "cache"
    args.temp_dir = cache_root / f"generations_{args.result_dir.name}"

    system_prompt = system_prompt_fillin(args.prompt_path, USER_PREFIX) if args.mode == "fsm" else None

    logger.info(f"model={args.model} mode={args.mode} modality={args.modality} asr={args.asr_type}")
    logger.info(f"results -> {args.result_dir}")

    # Skip datasets whose output file is already complete.
    pending = [(name, split) for name, split in DATASETS
               if not (args.result_dir / f"{name}-{'all' if name == 'mmsu' else split}-{args.modality}.jsonl").exists()]
    if not pending:
        logger.info("All dataset files already exist; nothing to do.")
        return

    all_tasks = []
    for name, split in pending:
        logger.info(f"Loading {name} ({split})")
        ds = load_dataset("hlt-lab/voicebench", name, split=split)
        if args.modality == "audio":
            ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        for item in ds:
            task = dict(item)
            task.update(dataset=name, split=split, global_id=f"{name}_{len(all_tasks)}")
            all_tasks.append(task)

    if args.modality == "audio":
        run_asr(all_tasks, args, cache_root / f"asr_{args.asr_type}.json")

    generate(all_tasks, args, system_prompt)
    logger.info("Done.")


if __name__ == "__main__":
    main()
