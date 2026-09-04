#!/usr/bin/env python3
"""Full-Duplex-Bench inference adapter for the DEF-FSM agent (v1.0 and v1.5).

Streams each ``input.wav`` (and its ``clean_input.wav`` reference) through the
FSM pipeline in real time and records a time-synchronous response wav, matching
the per-sample layout the FDB evaluation scripts expect:

    {base_dir}/{task}/{ID}/input.wav        ->  output.wav
    {base_dir}/{task}/{ID}/clean_input.wav  ->  clean_output.wav

The FSM is a threaded, real-time agent, so we reproduce exactly the streaming
conditions it sees at deployment: feed the user audio at a fixed 32 ms cadence
into the ASR queue, and place the TTS chunks it emits back onto a continuous
timeline starting at the same t=0 as the input.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from def_fsm.architecture import FSM
from def_fsm.paths import PROJECT_ROOT, require
from def_fsm.utils import USER_PREFIX

V15_TASKS = ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]
# v1.0 subsets (arXiv 2503.04721). Unlike v1.5, v1 has no clean/overlap pair: the
# response window lives *inside* input.wav (trailing silence for turn-taking /
# interruption; the whole clip for pause / backchannel), so the output track must
# be truncated to the input duration and no extra tail silence is fed.
V1_TASKS = [
    "candor_pause_handling",
    "synthetic_pause_handling",
    "candor_turn_taking",
    "icc_backchannel",
    "synthetic_user_interruption",
]
IO_PAIRS_V15 = [("input.wav", "output.wav"), ("clean_input.wav", "clean_output.wav")]
IO_PAIRS_V1 = [("input.wav", "output.wav")]

ASR_SR = 16_000  # rate the ASR queue consumes
TTS_SR = 24_000  # rate the TTS worker emits
FEED_CHUNK_SEC = 0.032  # standard 32 ms VAD/feed cadence

# The perception module is tied to the LLM: a model trained with SimulStreaming
# must be evaluated with it (and its Whisper .pt); others use Faster-Whisper (CT2).
FASTER_ASR_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"


def default_asr_model(asr_worker: str) -> str:
    """Resolve the checkpoint for a perception module. WHISPER_MODEL_PATH is only
    required when SimulStreaming is selected, so it is read lazily."""
    return FASTER_ASR_MODEL if asr_worker == "faster" else require("WHISPER_MODEL_PATH")


def load_mono_16k(path: Path) -> np.ndarray:
    """Read a wav as mono float32 resampled to the ASR rate."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != ASR_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=ASR_SR)
    return audio


def run_one(agent: FSM, in_q: queue.Queue, out_q: queue.Queue, in_path: Path, out_path: Path, max_tail_sec: float, settle_sec: float, protocol: str = "v15") -> None:
    """Stream one input wav through the FSM and write a time-synchronous output wav.

    ``protocol`` selects the recording horizon:
      - ``v15``: feed the input, then keep feeding dithered silence until the FSM's
        response drains (early-stop on idle, capped by ``max_tail_sec``). The output
        track spans the full observed time, so a post-input response is captured.
      - ``v1``:  feed *only* input.wav and stop the moment it is exhausted; the
        response window is already inside input.wav (trailing silence). The output
        track is truncated to the input duration, matching the FDB v1 protocol
        (cf. upstream moshi adapter's ``max_samples = len(input)``).
    """
    audio = load_mono_16k(in_path)
    input_dur = len(audio) / ASR_SR
    chunk_n = int(ASR_SR * FEED_CHUNK_SEC)
    n_input_chunks = (len(audio) + chunk_n - 1) // chunk_n

    # Recorder thread: timestamp every TTS chunk relative to the feed start.
    events: list[tuple[float, np.ndarray]] = []
    stop_rec = threading.Event()
    t0 = time.perf_counter()

    def recorder() -> None:
        while not stop_rec.is_set():
            try:
                _sr, chunk = out_q.get(timeout=0.05)
            except queue.Empty:
                continue
            events.append((time.perf_counter() - t0, chunk))

    rec_thread = threading.Thread(target=recorder, daemon=True)
    rec_thread.start()

    # v1: feed only input.wav, stop the moment it drains (no tail). The response
    # window is inside input.wav, so its trailing silence is fed and any response
    # emitted during it is recorded; anything past input_dur is truncated below.
    if protocol == "v1":
        next_tick = t0
        for idx in range(n_input_chunks):
            seg = audio[idx * chunk_n : (idx + 1) * chunk_n]
            if len(seg) < chunk_n:
                seg = np.pad(seg, (0, chunk_n - len(seg)))
            in_q.put(seg)
            next_tick += FEED_CHUNK_SEC
            dt = next_tick - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        total_sec = input_dur
        stop_rec.set()
        rec_thread.join()
        _render_track(events, total_sec, out_path)
        return

    # Real-time feed: input chunks, then dithered silence until the turn drains.
    idx, next_tick = 0, t0
    input_end: float | None = None
    idle_since: float | None = None
    while True:
        if idx < n_input_chunks:
            seg = audio[idx * chunk_n : (idx + 1) * chunk_n]
            if len(seg) < chunk_n:
                seg = np.pad(seg, (0, chunk_n - len(seg)))
        else:
            # Tiny dither keeps Faster-Whisper from hallucinating on pure silence.
            if input_end is None:
                input_end = time.perf_counter()
            seg = np.random.normal(0, 1e-6, chunk_n).astype(np.float32)
        in_q.put(seg)
        idx += 1

        # Only consider stopping once the whole user turn has been streamed.
        if input_end is not None:
            spoke = len(events) > 0
            idle = agent.shared_state["llm_state"] == "LISTEN" and agent.tts_worker.num_pending_text == 0
            if spoke and idle:
                idle_since = idle_since or time.perf_counter()
                if time.perf_counter() - idle_since >= settle_sec:
                    break
            else:
                idle_since = None
            if time.perf_counter() - input_end >= max_tail_sec:
                break

        next_tick += FEED_CHUNK_SEC
        dt = next_tick - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    total_sec = time.perf_counter() - t0
    stop_rec.set()
    rec_thread.join()
    _render_track(events, total_sec, out_path)


def _render_track(events: list[tuple[float, np.ndarray]], total_sec: float, out_path: Path) -> None:
    """Render the sparse (timestamp, chunk) TTS events onto one continuous 24k track
    spanning [0, total_sec] and write it to out_path. Chunks past total_sec are clipped."""
    track = np.zeros(int(total_sec * TTS_SR) + 1, dtype=np.float32)
    for rel, chunk in events:
        start = int(rel * TTS_SR)
        end = min(start + len(chunk), len(track))
        if start < len(track):
            track[start:end] += chunk[: end - start]
    sf.write(out_path, track, TTS_SR)


def save_tape(agent: FSM, out_path: Path) -> Path:
    """Snapshot the FSM's central tape at the end of a turn, next to its output wav.

    The tape (``shared_state["tape"]``) is a list of ``(text, perf_counter_ts)``
    entries the FSM accumulates during a run; it is rebuilt on ``reset()``, so this
    must be called after ``run_one`` and before the next unit resets the agent.
    Timestamps are re-based to the tape's own start (the system-prompt entry) so
    each file is self-contained. Written as ``<stem>_tape.json`` (e.g.
    ``output_tape.json`` / ``clean_output_tape.json``).
    """
    with agent.shared_state["tape_lock"]:
        tape = list(agent.shared_state["tape"])
    t_start = tape[0][1] if tape else 0.0
    entries = [{"t": round(ts - t_start, 4), "text": text} for text, ts in tape]

    tape_path = out_path.with_name(f"{out_path.stem}_tape.json")
    with open(tape_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return tape_path


def build_agent(args) -> FSM:
    """Construct a single FSM instance reused across all samples (models stay loaded)."""
    return FSM(
        asr_model_path=args.asr_model_path,
        llm_model_path=args.llm_model_path,
        tts_model_path=args.tts_model_path,
        llm_prompt_path=args.prompt_path,
        devices=args.devices,
        enable_playback=False,  # route audio to queues only
        asr_worker_choice=args.asr_worker,
        tts_worker_choice=args.tts_worker,
        simulation_audio_path=None,
        interlocutor_prefix=USER_PREFIX,  # the assistant listens to the user
        llm_temperature=args.temperature,
        log_folder_path=None,
    )


def iter_units(base_dir: Path, tasks: list[str], overwrite: bool, io_pairs: list[tuple[str, str]]):
    """Yield (in_path, out_path) work items across tasks, sample dirs, and io pairs."""
    for task in tasks:
        for sample_dir in sorted(glob(str(base_dir / task / "*"))):
            sample_dir = Path(sample_dir)
            if not sample_dir.is_dir():
                continue
            for in_name, out_name in io_pairs:
                in_path, out_path = sample_dir / in_name, sample_dir / out_name
                if not in_path.exists():
                    continue
                if out_path.exists() and not overwrite:
                    print(f"[SKIP] {out_path}")
                    continue
                yield in_path, out_path


def main() -> None:
    ap = argparse.ArgumentParser("Full-Duplex-Bench inference for DEF-FSM")
    ap.add_argument("--base-dir", required=True, help="Root holding {task}/{ID}/ sample folders.")
    ap.add_argument("--protocol", default="v15", choices=["v15", "v1"],
                    help="Benchmark protocol. v15: feed tail silence, capture post-input response. "
                         "v1: feed only input.wav and truncate output to its duration.")
    ap.add_argument("--task", default="all", choices=["all", *V15_TASKS, *V1_TASKS])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N units (smoke test).")
    # Model configuration (defaults mirror test/app_test_local.py).
    ap.add_argument("--asr-worker", default=None, choices=["faster", "simul"],
                    help="ASR backend; default: auto ('simul' if 'simul' is in the LLM path, else 'faster').")
    ap.add_argument("--asr-model-path", default=None,
                    help="ASR model; default: matches the resolved --asr-worker.")
    ap.add_argument("--llm-model-path", required=True)
    ap.add_argument("--tts-worker", default="kokoro", choices=["kokoro"])
    ap.add_argument("--tts-model-path", default=None)
    ap.add_argument("--prompt-path", default=str(PROJECT_ROOT / "prompts/fsm_assistant.txt"))
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:1"])
    # Recording horizon.
    ap.add_argument("--max-tail-sec", type=float, default=10.0, help="Max silence fed after the input before force-stopping the turn.")
    ap.add_argument("--settle-sec", type=float, default=0.5, help="How long the FSM must stay idle (LISTEN, drained) to end a turn early.")
    args = ap.parse_args()

    # Auto-pair the perception module with the model unless explicitly overridden.
    if args.asr_worker is None:
        args.asr_worker = "simul" if "simul" in args.llm_model_path.lower() else "faster"
    if args.asr_model_path is None:
        args.asr_model_path = default_asr_model(args.asr_worker)
    print(f"[INFO] ASR worker = {args.asr_worker} | model = {args.asr_model_path}")

    base_dir = Path(args.base_dir)
    all_tasks = V1_TASKS if args.protocol == "v1" else V15_TASKS
    io_pairs = IO_PAIRS_V1 if args.protocol == "v1" else IO_PAIRS_V15
    tasks = all_tasks if args.task == "all" else [args.task]
    units = list(iter_units(base_dir, tasks, args.overwrite, io_pairs))
    if args.limit is not None:
        units = units[: args.limit]
    print(f"[INFO] {len(units)} units to process across tasks: {tasks}")
    if not units:
        print("[INFO] nothing to do; skipping model load.")
        return

    agent = build_agent(args)
    try:
        pbar = tqdm(units, desc="FSM", unit="unit")
        for i, (in_path, out_path) in enumerate(pbar):
            # reset() stops the previous run and rebuilds threads with a fresh state
            # (KV cache cleared, models kept). The very first unit uses the fresh
            # threads created in __init__.
            if i > 0:
                agent.reset()

            in_q, out_q = queue.Queue(), queue.Queue()
            agent.asr_worker.set_input_audio_queue(in_q)
            agent.tts_worker.set_output_queue(out_q)

            pbar.set_postfix_str(f"{in_path.parent.name}/{in_path.name}")
            agent.start()
            run_one(agent, in_q, out_q, in_path, out_path, args.max_tail_sec, args.settle_sec, protocol=args.protocol)
            # Persist the tape the FSM built this turn before the next unit resets it.
            save_tape(agent, out_path)
    finally:
        agent.stop()
    print("[DONE]")


if __name__ == "__main__":
    main()
