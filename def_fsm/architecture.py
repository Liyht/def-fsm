"""Thread-based streaming pipeline for full-duplex FSM speech dialogue.

Data flow: audio in -> ASR recognition -> LLM generation -> TTS synthesis.

This module defines the FSM coordinator, which assembles and schedules the ASR, LLM and TTS workers.
"""

import time
import sys
import os
import json
from queue import Queue
from threading import Thread, Event, Lock
import logging
from typing import List, Optional

from def_fsm.asr_worker import FSMSimulWhisperASRWorker, FSMFasterWhisperASRWorker
from def_fsm.llm_worker import FSMLlamaCppWorker
from def_fsm.tts_worker import FSMKokoroWorker
from def_fsm.utils import INTERLOCUTOR_PREFIX, system_prompt_fillin


def setup_logging(log_path, log_level=logging.INFO):
    """Configure the root logger to write to both a file and the console."""

    # Passing no name gives the root logger, so every child logger shares this configuration
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Drop existing handlers so repeated configuration does not duplicate log lines
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging setup complete. Logging to console and file: %s", log_path)


class FSM:
    """Full-duplex FSM streaming pipeline.

    A single central `shared_state` dict holds all state and inter-thread communication, with the LLM as the central controller.
    """

    def __init__(
        self,
        asr_model_path: str,
        llm_model_path: str,
        llm_prompt_path: str,
        devices: List[str] = ["cuda:0", "cuda:1", "cuda:2"],
        simulation_audio_path: Optional[str] = None,
        enable_playback: bool = True,
        asr_worker_choice: str = "faster",  # "faster" or "simul"
        llm_temperature: float = 0,
        interlocutor_prefix: str = INTERLOCUTOR_PREFIX,
        log_folder_path: str = "./log/",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_folder_path = log_folder_path
        if log_folder_path:
            os.makedirs(self.log_folder_path, exist_ok=True)

        # 1. Assign GPUs. With only two cards, ASR and TTS share the second one.
        if len(devices) == 2:
            self.llm_device, self.asr_device, self.tts_device = devices[0], devices[1], devices[1]
        elif len(devices) >= 3:
            self.llm_device, self.asr_device, self.tts_device = devices[0], devices[1], devices[2]
        else:
            raise Exception("Need at least two gpus")

        # 2. Load the system prompt
        self.logger.info(f"Loading system prompt from {llm_prompt_path}")
        self.system_prompt = system_prompt_fillin(llm_prompt_path, interlocutor_prefix)

        # 3. Initialize the workers

        # 3.1 ASR worker
        if asr_worker_choice == "faster":
            self.logger.info("--- Initializing FSM ASR Worker (Faster Whisper) ---")
            self.asr_worker = FSMFasterWhisperASRWorker(
                model_path=asr_model_path,
                device=self.asr_device,
                simulation_audio_path=simulation_audio_path,
            )
        elif asr_worker_choice == "simul":
            self.logger.info("--- Initializing FSM ASR Worker (Simul Whisper) ---")
            self.asr_worker = FSMSimulWhisperASRWorker(
                model_path=asr_model_path,
                device=self.asr_device,
                simulation_audio_path=simulation_audio_path,
            )
        else:
            raise ValueError(f"Unknown ASR backend: {asr_worker_choice}")

        # 3.2 LLM worker
        self.logger.info("--- Initializing FSM LLM (LlamaCpp) Worker ---")
        self.llm_worker = FSMLlamaCppWorker(
            system_prompt=self.system_prompt,
            model_path=llm_model_path,
            device=self.llm_device,
            max_length=1024,
            max_tokens=50,
            temperature=llm_temperature,
            interlocutor_prefix=interlocutor_prefix,
        )

        # 3.3 TTS worker
        self.logger.info("--- Initializing FSM TTS Worker (Kokoro) ---")
        self.tts_worker = FSMKokoroWorker(
            device=self.tts_device,
            enable_playback=enable_playback,
        )

        # 4. Load the models
        self._load_models()

        self._initialize_state()
        self._initialize_threads()

        self.logger.info("FSM initialized successfully.")

    def _load_models(self):
        """Load and warm up the ASR, LLM and TTS models concurrently, one thread each, to cut startup time."""
        self.logger.info("Starting concurrent model loading...")

        def _load_and_warmup(worker):
            worker.load_model()
            worker.warmup()

        asr_load_thread = Thread(target=_load_and_warmup, args=(self.asr_worker,))
        llm_load_thread = Thread(target=_load_and_warmup, args=(self.llm_worker,))
        tts_load_thread = Thread(target=_load_and_warmup, args=(self.tts_worker,))

        start_time = time.perf_counter()
        asr_load_thread.start()
        llm_load_thread.start()
        tts_load_thread.start()

        asr_load_thread.join()
        llm_load_thread.join()
        tts_load_thread.join()

        end_time = time.perf_counter()
        self.logger.info(f"All models loaded and warmed up concurrently in {end_time - start_time:.2f} seconds.")

    def _reset_pipeline_timer(self):
        """Clear the per-stage timing records in preparation for another run."""
        self.pipeline_timer = {
            "t_pipeline_start_time": [time.perf_counter()],
            "t_receive_audio_chunk": [],
            "t_asr_output_text": [],
            "t_llm_output_token": [],
            "t_tts_receive_token": [],
            "t_tts_output_chunk": [],
            "t_playback_chunk": [],
            "t_asr_compute_start": [],
            "t_asr_compute_end": [],
            "t_llm_compute_start": [],
            "t_tts_compute_start": [],
        }

    def _initialize_state(self):
        """Initialize the central `shared_state`: every communication queue, event and the FSM state itself."""
        self._reset_pipeline_timer()

        self.shared_state = {
            # FSM state: SPEAK / LISTEN
            "llm_state": "LISTEN",
            # The central tape; its first entry is the system prompt
            "tape": [(self.system_prompt, time.perf_counter())],
            "tape_lock": Lock(),
            "tts_num_pending_text_lock": Lock(),
            # Inter-thread communication queues
            "asr_to_llm_queue": Queue(),
            "llm_to_tts_queue": Queue(),
            # Control
            "stop_event": Event(),
            "tts_worker_ref": self.tts_worker,
            # Timing
            "pipeline_timer": self.pipeline_timer,
        }

        self.asr_worker.load_state(self.shared_state)
        self.llm_worker.load_state(self.shared_state)
        self.tts_worker.load_state(self.shared_state)

        # Build the KV cache for the system prompt
        self.llm_worker.warmup()

    def _initialize_threads(self):
        """Create every worker thread (created only, not started)."""
        self.asr_thread = Thread(target=self.asr_worker.run_worker)
        self.llm_thread = Thread(target=self.llm_worker.run_worker)
        self.tts_thread = Thread(target=self.tts_worker.run_tts_worker)
        self.playback_thread = Thread(target=self.tts_worker.run_playback_worker)

    def start(self, init_sentence: Optional[str] = None):
        """Start every worker thread.

        If init_sentence is given, the FSM is put straight into SPEAK, the sentence is
        written onto the tape and pushed to TTS immediately, so the agent speaks first.
        """
        if init_sentence:
            self.logger.info(f"--- Initializing FSM with start sentence: '{init_sentence}' ---")
            timestamp = time.perf_counter()

            # 1. Force the SPEAK state
            self.shared_state["llm_state"] = "SPEAK"

            # 2. Write the control token [S.SPEAK] and the opening sentence onto the tape
            ts_gen = time.perf_counter()
            with self.shared_state["tape_lock"]:
                self.shared_state["tape"].append(("[S.SPEAK]", timestamp))
                self.shared_state["tape"].append((init_sentence, ts_gen))

            # 3. Push it to the TTS queue and increment the in-flight count
            with self.shared_state["tts_num_pending_text_lock"]:
                self.shared_state["tts_worker_ref"].num_pending_text += 1

            self.shared_state["llm_to_tts_queue"].put((init_sentence, ts_gen))

        self.logger.info("--- Starting FSM streaming pipeline ---")
        self.asr_thread.start()
        self.llm_thread.start()
        self.tts_thread.start()
        self.playback_thread.start()

    def stop(self):
        """Stop every worker thread."""
        self.logger.info("--- Stopping FSM streaming pipeline ---")
        self.shared_state["stop_event"].set()

        # Send a sentinel to the ASR input queue to release its blocking read
        if hasattr(self, "asr_worker") and self.asr_worker.input_audio_queue:
            self.asr_worker.input_audio_queue.put(None)

        # Wait for the threads to exit
        if hasattr(self, "asr_thread") and self.asr_thread.is_alive():
            self.asr_thread.join()
        if hasattr(self, "llm_thread") and self.llm_thread.is_alive():
            self.llm_thread.join()
        if hasattr(self, "tts_thread") and self.tts_thread.is_alive():
            self.tts_thread.join()
        if hasattr(self, "playback_thread") and self.playback_thread.is_alive():
            self.playback_thread.join()

        # Save the timing data
        if self.log_folder_path:
            timing_path = os.path.join(self.log_folder_path, "timing.json")
            self.logger.info(f"Saving timing data to {timing_path}")
            with open(timing_path, "w") as f:
                json.dump(self.pipeline_timer, f, indent=4)

        self.logger.info("--- FSM Pipeline stopped ---")

    def reset(self):
        """Reset the pipeline to a clean state that can be started again.

        The loaded models are kept; state and queues are cleared, events reset and threads rebuilt.
        """
        self.logger.info("Resetting FSM pipeline...")
        self.stop()

        # Check whether any thread is still alive
        if (
            hasattr(self, "asr_thread")
            and self.asr_thread.is_alive()
            or hasattr(self, "llm_thread")
            and self.llm_thread.is_alive()
            or hasattr(self, "tts_thread")
            and self.tts_thread.is_alive()
            or hasattr(self, "playback_thread")
            and self.playback_thread.is_alive()
        ):
            self.logger.warning("Resetting pipeline while threads are still alive.")

        # Reset ASR / LLM / TTS
        self.asr_worker.reset()

        if isinstance(self.llm_worker, FSMLlamaCppWorker):
            self.logger.info("LLM Context reset (KV cache cleared, model weights intact).")
            self.llm_worker.model.reset()
        else:
            self.logger.info(f"Reloading LLM worker model ({self.llm_worker.__class__.__name__})...")
            self.llm_worker.load_model()

        self.tts_worker.reset()

        # Rebuild the state and the threads
        self._initialize_state()
        self._initialize_threads()
        self.logger.info("FSM Pipeline reset complete. Ready to start.")
