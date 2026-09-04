"""Talk to a trained DEF-FSM model through the local microphone and speakers.

Loads a GGUF checkpoint into the FSM pipeline, then records the session audio and the tape.
"""

import argparse
import time
import logging
import sys
import os
import queue
import threading
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from threading import Thread, Event

from def_fsm.architecture import FSM
from def_fsm.paths import expand
from def_fsm.utils import USER_PREFIX

# Defaults; every entry can be overridden on the command line (see parse_args).
# Example: python demo/app_local.py --llm-model-path /path/to/model.gguf
MODEL_CONFIG = {
    # Perception module: "faster" (IPU-level, the default) or
    # "simul" (word-level, needs the patched SimulStreaming checkout).
    "asr_worker_choice": "faster",
    "asr_model_path": "deepdml/faster-whisper-large-v3-turbo-ct2",
    # Cognitive module: a GGUF export of a fine-tuned checkpoint, produced by
    # scripts/convert_to_gguf.sh. Required.
    "llm_model_path": None,
    "prompt_path": expand("${PROJECT_ROOT}/prompts/fsm_assistant.txt"),
    "temperature": 1.0,
    # One device each for the LLM, the ASR and the TTS worker.
    "devices": ["cuda:0", "cuda:1", "cuda:1"],
    "output_dir": expand("${PROJECT_ROOT}/outputs/dialogue_sessions"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full-duplex FSM against the local microphone and speakers.")
    parser.add_argument("--llm-model-path", required=True, help="GGUF checkpoint for the cognitive module")
    parser.add_argument("--asr-worker", choices=["faster", "simul"], default=MODEL_CONFIG["asr_worker_choice"])
    parser.add_argument("--asr-model-path", default=MODEL_CONFIG["asr_model_path"])
    parser.add_argument("--prompt-path", default=MODEL_CONFIG["prompt_path"])
    parser.add_argument("--temperature", type=float, default=MODEL_CONFIG["temperature"])
    parser.add_argument("--devices", nargs=3, metavar=("LLM", "ASR", "TTS"), default=MODEL_CONFIG["devices"])
    parser.add_argument("--output-dir", default=MODEL_CONFIG["output_dir"], help="Where the recorded session wavs are written")
    args = parser.parse_args()

    MODEL_CONFIG.update(
        asr_worker_choice=args.asr_worker,
        asr_model_path=args.asr_model_path,
        llm_model_path=args.llm_model_path,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
        devices=list(args.devices),
        output_dir=args.output_dir,
    )
    return args


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"  # Agent Content
    GREEN = "\033[92m"  # User Content
    RED = "\033[91m"  # State Changes ([S.SPEAK])
    YELLOW = "\033[93m"  # Warnings/System
    RESET = "\033[0m"
    BOLD = "\033[1m"


class SessionRecorder(Thread):
    def __init__(self, asr_queue, tts_queue, output_prefix="session"):
        super().__init__()
        self.asr_queue = asr_queue
        self.tts_queue = tts_queue
        self.output_prefix = output_prefix
        self.stop_event = Event()

        self.start_time = None
        self.user_audio_buffer = []
        self.agent_audio_events = []  # (rel_time, chunk)

        # Sampling rates, which must match the ASR and TTS workers
        self.USER_SR = 16000
        self.AGENT_SR = 24000

    def run(self):
        print(f"{Colors.YELLOW}[Recorder] Background thread started...{Colors.RESET}")
        self.start_time = time.perf_counter()

        while not self.stop_event.is_set():
            # 1. Non-blocking read from ASR Queue (User Mic)
            try:
                while True:
                    chunk = self.asr_queue.get_nowait()
                    self.user_audio_buffer.append(chunk)
            except queue.Empty:
                pass

            # 2. Non-blocking read from TTS Queue (Agent Speech)
            try:
                while True:
                    item = self.tts_queue.get_nowait()
                    # TTS queue returns (samplerate, chunk_numpy)
                    sr, chunk = item

                    # Record relative timestamp for alignment
                    rel_time = time.perf_counter() - self.start_time
                    self.agent_audio_events.append((rel_time, chunk))
            except queue.Empty:
                pass

            time.sleep(0.01)  # Prevent CPU spin

    def stop(self):
        self.stop_event.set()
        self.join()

    def save_files(self):
        if not self.user_audio_buffer:
            print(f"{Colors.YELLOW}[Recorder] No audio recorded.{Colors.RESET}")
            return

        print(f"{Colors.YELLOW}[Recorder] Processing audio files...{Colors.RESET}")

        # 1. User Track (Continuous)
        user_audio = np.concatenate(self.user_audio_buffer)
        wav.write(f"{self.output_prefix}_user.wav", self.USER_SR, user_audio)

        duration_sec = len(user_audio) / self.USER_SR
        print(f"[Recorder] User duration: {duration_sec:.2f}s")

        # 2. Agent Track (Sparse -> Continuous)
        # Create a silent track matching user duration (in Agent SR)
        total_agent_samples = int(duration_sec * self.AGENT_SR)
        agent_track = np.zeros(total_agent_samples, dtype=np.float32)

        for rel_time, chunk in self.agent_audio_events:
            start_sample = int(rel_time * self.AGENT_SR)
            end_sample = start_sample + len(chunk)

            if start_sample < total_agent_samples:
                write_len = min(len(chunk), total_agent_samples - start_sample)
                agent_track[start_sample : start_sample + write_len] += chunk[:write_len]

        wav.write(f"{self.output_prefix}_agent.wav", self.AGENT_SR, agent_track)

        # 3. Merge Stereo (Resample User to match Agent)
        print("[Recorder] Merging to stereo (Resampling User 16k -> 24k)...")
        # Calculate target number of samples for user audio
        target_user_samples = int(len(user_audio) * self.AGENT_SR / self.USER_SR)
        user_resampled = signal.resample(user_audio, target_user_samples)

        # Ensure lengths match exactly for stereo stacking
        min_len = min(len(user_resampled), len(agent_track))
        stereo_data = np.vstack((user_resampled[:min_len], agent_track[:min_len])).T

        wav.write(f"{self.output_prefix}_stereo_merged.wav", self.AGENT_SR, stereo_data)
        print(f"{Colors.GREEN}[Recorder] Saved: {self.output_prefix}_stereo_merged.wav (L:User, R:Agent){Colors.RESET}")


def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [LocalRunner] - %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("LocalRunner")

    fsm_pipeline = None
    recorder = None

    # Create Queues for recording
    rec_asr_queue = queue.Queue()
    rec_tts_queue = queue.Queue()

    try:
        print(f"{Colors.HEADER}=== Initializing Full-Duplex FSM Agent (Local Mode) ==={Colors.RESET}")

        # Initialize FSM
        # Note: We set enable_playback=True to let the TTS worker use local speakers directly
        fsm_pipeline = FSM(
            asr_model_path=MODEL_CONFIG["asr_model_path"],
            llm_model_path=MODEL_CONFIG["llm_model_path"],
            llm_prompt_path=MODEL_CONFIG["prompt_path"],
            devices=MODEL_CONFIG["devices"],
            enable_playback=True,
            asr_worker_choice=MODEL_CONFIG["asr_worker_choice"],
            simulation_audio_path=None,  # Ensure this is None to trigger Microphone mode
            interlocutor_prefix=USER_PREFIX,
            log_folder_path=None,
            llm_temperature=MODEL_CONFIG["temperature"],
        )

        # Inject Queues for Recording
        # 1. ASR: Push raw mic data to rec_asr_queue
        if hasattr(fsm_pipeline.asr_worker, "set_record_audio_queue"):
            fsm_pipeline.asr_worker.set_record_audio_queue(rec_asr_queue)
        else:
            logger.warning("ASR Worker does not have 'set_record_audio_queue'. User audio will NOT be recorded.")

        # 2. TTS: Push generated audio to rec_tts_queue (Reuse existing output_queue logic)
        fsm_pipeline.tts_worker.set_output_queue(rec_tts_queue)

        # Start Recorder
        os.makedirs(MODEL_CONFIG["output_dir"], exist_ok=True)
        recorder = SessionRecorder(rec_asr_queue, rec_tts_queue, output_prefix=os.path.join(MODEL_CONFIG["output_dir"], "session"))
        recorder.start()

        print(f"{Colors.YELLOW}>>> Starting Pipeline... Please verify microphone and speakers are active.{Colors.RESET}")
        fsm_pipeline.start()

        print(f"{Colors.BOLD}System Ready. Start speaking.{Colors.RESET}")
        print(f"(Press {Colors.RED}Ctrl+C{Colors.RESET} to stop)")
        print("-" * 50)

        # Monitoring loop: poll the shared tape and print whatever has been appended since last time
        last_tape_index = 0

        while True:
            # Access the shared state safely
            shared_state = fsm_pipeline.shared_state

            # 1. Print State Changes
            current_state = shared_state.get("llm_state", "UNKNOWN")

            # 2. Print New Tape Entries
            with shared_state["tape_lock"]:
                current_tape = shared_state["tape"]
                total_len = len(current_tape)

                if total_len > last_tape_index:
                    new_items = current_tape[last_tape_index:]
                    last_tape_index = total_len

                    for token, ts in new_items:
                        # Color coding based on token type
                        if USER_PREFIX in token:
                            clean_text = token.strip()
                            print(f"{Colors.GREEN}[User]: {clean_text}{Colors.RESET}")
                        elif "[S." in token or "[C." in token:
                            print(f"{Colors.RED}[State]: {token}{Colors.RESET}")
                        else:
                            clean_text = token.strip()
                            print(f"{Colors.BLUE}[Agent]: {clean_text}{Colors.RESET}")

            time.sleep(0.05)  # Fast poll

    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Stopping System...{Colors.RESET}")
    except Exception as e:
        logger.error(f"Runtime Error: {e}", exc_info=True)
    finally:
        if fsm_pipeline:
            fsm_pipeline.stop()
            print("Pipeline Stopped.")

        # Stop and Save Recorder
        if recorder:
            print("Stopping recorder...")
            recorder.stop()
            recorder.save_files()


if __name__ == "__main__":
    parse_args()
    main()
