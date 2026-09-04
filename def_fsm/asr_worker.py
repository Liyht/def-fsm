"""Strategy-pattern definitions for the ASR workers.

Every worker follows the FSM protocol: it communicates through the shared_state dict and emits timestamped recognition results.
"""

import sys
import os
import torch
import numpy as np
import time
import logging
from queue import Queue, Empty
from threading import Event, Thread
from abc import ABC, abstractmethod
from argparse import Namespace
import json
from collections import deque
import traceback

import soundfile as sf
import librosa

from def_fsm.paths import require
from def_fsm.utils import is_punctuation, SILENCE_TOKEN, SILENCE_TOKEN_DUR

# FasterWhisper + Silero VAD
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, read_audio


def _load_simulstreaming():
    """Import the SimulStreaming factories on first use.

    SimulStreaming is an out-of-tree checkout (see third_party/patches/), so it
    is imported lazily, so the FasterWhisper perception module used by default
    works without it installed.
    """
    root = require("SIMULSTREAMING_PATH")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"SIMULSTREAMING_PATH points at '{root}', which does not exist. "
            "Clone ufal/SimulStreaming at commit 240be1f, apply "
            "third_party/patches/simulstreaming.patch, and set the path in "
            "configs/paths.yaml."
        )
    if root not in sys.path:
        sys.path.append(root)

    from simulstreaming_whisper import simul_asr_factory
    from whisper_streaming.whisper_online_main import asr_factory

    return simul_asr_factory, asr_factory


HALLUCINATION_TRIGGERS_FASTER = {"Thank you.", "Bye.", "We'll be right back.", "see you next time."}

HALLUCINATION_TRIGGERS_SIMUL = {"thank you", "bye", "we'll be right back", "see you next time"}


class BaseASRWorker(ABC):
    """Abstract base class for FSM ASR workers.

    It holds the shared FSM logic (load_state, run_worker, run_loop,
    run_file_simulation_and_save). A subclass only has to implement load_model(), reset()
    and _process_audio_chunk().
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        language: str = "en",
        sampling_rate: int = 16000,
        chunk_duration: float = 0.32,
        simulation_audio_path: str = None,
        simulate_realtime: bool = True,
        fit_chunk_duration: bool = True,
        add_silence_token: bool = True,
        silence_token_dur: float = SILENCE_TOKEN_DUR,
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.language = language
        self.simulation_audio_path = simulation_audio_path
        self.simulate_realtime = simulate_realtime
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(self.sampling_rate * self.chunk_duration)
        self.fit_chunk_duration = fit_chunk_duration
        self.add_silence_token = add_silence_token
        self.silence_token_dur = silence_token_dur
        self.silence_token_samples = int(silence_token_dur * self.sampling_rate)
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

        self.record_audio_queue = None
        self.input_audio_queue = None

    def set_record_audio_queue(self, q: Queue):
        self.record_audio_queue = q

    def set_input_audio_queue(self, q: Queue):
        self.input_audio_queue = q

    @abstractmethod
    def load_model(self):
        """Load the model onto the given device."""
        pass

    def warmup(self):
        """Run one dummy inference to warm up."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the ASR internal state, such as the streaming buffer."""
        pass

    @abstractmethod
    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_final: bool):
        """Process a single audio chunk."""
        pass

    def load_state(self, shared_state: dict):
        """Load the shared FSM state."""
        self.output_queue = shared_state["asr_to_llm_queue"]
        self.stop_event = shared_state["stop_event"]
        self.pipeline_timer = shared_state["pipeline_timer"]
        self.shared_state = shared_state

    def run_worker(self):
        """Pick the audio source configured at construction (microphone / file / queue) and start the processing loop."""
        torch.cuda.set_device(self.device)
        try:
            if self.input_audio_queue:
                self.logger.info("[ASR] Mode: Queue Stream (Blocking & Sync)")
                self.stream_context = QueueInputStream(self.input_audio_queue, sampling_rate=self.sampling_rate)
            elif self.simulation_audio_path:
                self.logger.info(f"[ASR] Mode: File Simulation ({self.simulation_audio_path}) | Realtime: {self.simulate_realtime}")
                self.stream_context = RealtimeFileStream(
                    self.simulation_audio_path, sampling_rate=self.sampling_rate, real_time=self.simulate_realtime
                )
            else:
                self.logger.info("[ASR] Mode: Live Microphone")
                import sounddevice as sd

                self.stream_context = sd.InputStream(
                    samplerate=self.sampling_rate,
                    channels=1,
                    dtype="float32",
                )

            with self.stream_context as stream:
                self._run_loop(stream)

        except Exception as e:
            self.logger.error(f"[ASR] ASR worker failed: {e}", exc_info=True)

    def _run_loop(self, audio_source):
        """Processing loop shared by the microphone and file sources; handles variable-length chunking and ASR."""
        stop_event = self.stop_event
        pipeline_timer = self.pipeline_timer
        self.last_output_timestamp = time.perf_counter()

        is_final = False
        processed_audio_end = time.perf_counter()

        while not stop_event.is_set():
            if not self.simulate_realtime and self.simulation_audio_path:
                # Accelerated simulation: ignore the clock and ask for a full chunk
                need_process_samples = self.chunk_samples
                processed_audio_end += need_process_samples / self.sampling_rate
            else:
                # Real-time mode: derive the sample count from the elapsed wall-clock time
                need_process_samples = int((time.perf_counter() - processed_audio_end) * self.sampling_rate)

                if self.fit_chunk_duration:
                    need_process_samples = max(need_process_samples, self.chunk_samples)
                    processed_audio_end = time.perf_counter()
                else:
                    need_process_samples = self.chunk_samples
                    processed_audio_end += need_process_samples / self.sampling_rate

            indata, overflowed = audio_source.read(need_process_samples)
            if overflowed:
                self.logger.error("[ASR] Audio buffer overflowed. Processing may be too slow.")
            is_final = len(indata) < need_process_samples
            audio_chunk = indata.flatten()

            if self.record_audio_queue:
                self.record_audio_queue.put(audio_chunk)

            pipeline_timer["t_receive_audio_chunk"].append(time.perf_counter())

            self._process_audio_chunk(audio_chunk, is_final)
            if is_final:
                self._process_audio_chunk(np.array([], dtype=np.float32), True)
                self.logger.info("[ASR] File simulation finished.")
                break

    def run_file_simulation_and_save(self, audio_file_path: str, output_file_path: str):
        """Process an audio file under realistic conditions and save the recognition results.

        Processing follows wall-clock time to reproduce real latency, and result timestamps are aligned to t=0 at the start of the audio.
        """
        import threading

        # 1. Configure the audio source
        self.simulation_audio_path = audio_file_path

        # 2. Build a mock shared_state
        self.reset()
        mock_queue = Queue()
        stop_event = Event()

        mock_shared_state = {
            "asr_to_llm_queue": mock_queue,
            "stop_event": stop_event,
            "pipeline_timer": {"t_receive_audio_chunk": [], "t_asr_output_text": []},
            "llm_state": "LISTEN",
        }

        # Load the state and reset
        self.load_state(mock_shared_state)

        results = []

        # 3. Start the processing thread
        self.logger.info(f"Starting file simulation: {audio_file_path}")
        print(f"Processing {audio_file_path} ... (Ctrl+C to stop)")

        t = threading.Thread(target=self.run_worker)
        t.start()

        # 4. The main thread reads results off the queue and records them
        try:
            while True:
                try:
                    item = mock_queue.get(timeout=0.5)
                except Empty:
                    if not t.is_alive():
                        break
                    continue

                if item is None:
                    break

                start_time = self.stream_context.start_time
                text, receive_timestamp, finish_timestamp, audio_start, audio_end = item

                rel_receive = max(0.0, receive_timestamp - start_time)
                rel_finish = max(0.0, finish_timestamp - start_time)

                results.append((text, rel_receive, rel_finish, audio_start, audio_end))

        except KeyboardInterrupt:
            self.logger.warning("Simulation interrupted by user.")
            stop_event.set()
        finally:
            t.join()
            self.reset()

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Transcription saved to {output_file_path}")
        print(f"Done. Results saved to: {output_file_path}")


class FSMSimulWhisperASRWorker(BaseASRWorker):
    """FSM ASR worker built on SimulWhisper (word-level perception)."""

    def __init__(self, vac_chunk_size=0.08, **kwargs):
        super().__init__(**kwargs)
        self.vac_chunk_size = vac_chunk_size
        self.reset()

    def load_model(self):
        """Load the SimulWhisper ASR model."""
        asr_logger = logging.getLogger("simul_whisper.simul_whisper")
        asr_logger.setLevel(logging.ERROR)
        asr_logger = logging.getLogger("whisper_streaming.vac_online_processor")
        asr_logger.setLevel(logging.ERROR)

        torch.cuda.set_device(self.device)

        args = Namespace(
            # processor_args
            min_chunk_size=self.chunk_duration,
            lan=self.language,
            task="transcribe",
            vac=True,
            vac_chunk_size=self.vac_chunk_size,
            log_level="ERROR",
            logdir=None,
            # simulation_args
            audio_path=None,
            start_at=0.0,
            offline=False,
            comp_unaware=False,
            # simulwhisper_args
            model_path=self.model_path,
            beams=1,
            decoder=None,
            audio_max_len=30.0,
            audio_min_len=0.0,
            frame_threshold=25,
            cif_ckpt_path=None,
            never_fire=False,
            init_prompt=None,
            static_init_prompt=None,
            max_context_tokens=None,
            # Added by third_party/patches/simulstreaming.patch
            device=self.device,
            min_silence_duration_ms=self.silence_token_dur * 1000,
        )
        simul_asr_factory, asr_factory = _load_simulstreaming()
        self.asr, self.asr_online = asr_factory(args, simul_asr_factory)

        self.chunk_samples = int(self.sampling_rate * self.chunk_duration)

    def warmup(self):
        """Run a few dummy inferences to warm up SimulWhisper."""
        self.logger.info("Warming up ASR...")
        dummy_audio = np.random.uniform(low=-0.1, high=0.1, size=self.sampling_rate).astype(np.float32)
        for _ in range(5):
            self.asr.warmup(dummy_audio)
        self.logger.info("ASR warm-up complete.")

    def reset(self):
        """Reset the SimulWhisper online streaming state."""
        if hasattr(self, "asr_online"):
            self.asr_online.init()
            self.logger.debug("ASR online state reset.")

        self.global_sample_idx = 0
        self.last_speech_end_sample = 0

    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_final: bool):
        """Process a single audio chunk and queue the result on shared_state according to the current FSM state."""
        receive_timestamp = time.perf_counter()
        self.asr_online.insert_audio_chunk(audio_chunk)

        compute_start = time.perf_counter()
        if not is_final:
            result = self.asr_online.process_iter()
        else:
            result = self.asr_online.finish()
        compute_end = time.perf_counter()

        # Advance the global sample count (the audio clock)
        chunk_len = len(audio_chunk)
        current_end_sample = self.global_sample_idx + chunk_len

        # Handle the recognition result
        if result and result.get("text"):
            text = result["text"]

            for hallucination_trigger in HALLUCINATION_TRIGGERS_SIMUL:
                if hallucination_trigger in text.lower():
                    avg_logprob = result["avg_logprob"]

                    if avg_logprob < -0.3:
                        self.logger.warning(f"[ASR] Possible hallucination filtered: '{text}' (avg_logprob: {avg_logprob:.2f})")
                        return
                    else:
                        self.logger.warning(f"[ASR] Possible hallucination not filtered: '{text}' (avg_logprob: {avg_logprob:.2f})")

            if text and not is_punctuation(text):
                finish_timestamp = time.perf_counter()

                audio_start = result["start"]
                audio_end = min(result["end"], current_end_sample / self.sampling_rate)

                self.logger.info(f"[ASR] Detected: {text}")

                self.pipeline_timer.setdefault("t_vad_speech_end_wall", []).append(receive_timestamp)
                self.pipeline_timer.setdefault("t_asr_compute_start", []).append(compute_start)
                self.pipeline_timer.setdefault("t_asr_compute_end", []).append(compute_end)

                self.pipeline_timer["t_asr_output_text"].append(finish_timestamp)
                self.output_queue.put((text, receive_timestamp, finish_timestamp, audio_start, audio_end))

                self.last_speech_end_sample = current_end_sample

                self.global_sample_idx = current_end_sample
                return

        silence_duration_samples = current_end_sample - self.last_speech_end_sample

        if self.shared_state["llm_state"] == "LISTEN" and self.add_silence_token and silence_duration_samples >= self.silence_token_samples:
            finish_timestamp = time.perf_counter()

            sil_audio_end = current_end_sample / self.sampling_rate
            sil_audio_start = (current_end_sample - self.silence_token_samples) / self.sampling_rate

            self.output_queue.put((SILENCE_TOKEN, receive_timestamp, finish_timestamp, sil_audio_start, sil_audio_end))

            self.last_speech_end_sample = current_end_sample

        self.global_sample_idx += chunk_len


class FSMFasterWhisperASRWorker(BaseASRWorker):
    """FSM ASR worker built on Faster-Whisper plus Silero VAD (IPU-level perception).

    It puts timestamped tuples onto shared_state["asr_to_llm_queue"].
    """

    def __init__(
        self,
        chunk_duration=0.032,
        vad_pause_dur=0.032,
        vad_threshold_start=0.5,
        min_speech_dur=0.16,
        num_pre_pad_chunks=1,
        num_post_pad_chunks=2,
        use_context_prompt=False,
        **kwargs,
    ):
        super().__init__(chunk_duration=chunk_duration, **kwargs)

        # VAD thresholds: speech starts on the start threshold and ends on a lower exit threshold (hysteresis, to avoid flapping)
        self.vad_window_size = 512 if self.sampling_rate == 16000 else 256
        self.vad_threshold_start = vad_threshold_start
        self.vad_threshold_exit = self.vad_threshold_start - 0.15

        self.chunk_samples = int(self.chunk_duration * self.sampling_rate)

        # Time-related quantities, all converted to sample counts
        self.vad_pause_samples = int(vad_pause_dur * self.sampling_rate)
        self.min_speech_samples = int(min_speech_dur * self.sampling_rate)
        self.num_pre_pad_chunks = int(num_pre_pad_chunks)
        self.num_post_pad_chunks = int(num_post_pad_chunks)

        self.use_context_prompt = use_context_prompt

        self.reset()

    def load_model(self):
        """Load Faster-Whisper and Silero VAD."""
        self.logger.info(f"Loading Faster Whisper ({self.model_path}) and Silero VAD to {self.device}...")
        self.asr_model = WhisperModel(self.model_path, device="cuda", device_index=int(self.device[-1]), compute_type="float16")
        self.vad_model = load_silero_vad(onnx=False, opset_version=16)

    def warmup(self):
        """Run a few dummy inferences to warm up the VAD and Faster-Whisper, then reset the streaming state."""
        self.logger.info("Warming up VAD and ASR...")
        dummy_audio = np.random.uniform(low=-0.9, high=0.9, size=self.vad_window_size * 100).astype(np.float32)
        for _ in range(10):
            self.vad_model(torch.from_numpy(dummy_audio[: self.vad_window_size]), self.sampling_rate)
            self.asr_model.transcribe(dummy_audio, beam_size=5)
        self.reset()
        self.logger.info("VAD and ASR warm-up complete.")

    def reset(self):
        # VAD state variables
        self.triggered = False
        self.temp_end = 0
        self.current_speech_buffer = []
        self.pre_chunks = deque(maxlen=self.num_pre_pad_chunks)
        self.silence_accumulator = 0
        self.leftover_audio = np.array([], dtype=np.float32)

        if self.use_context_prompt:
            self.prompt_history = ""
            self.max_prompt_chars = 200

        # Time tracking
        self.vad_global_sample_idx = 0
        self.current_speech_start_info = {"wall": 0.0, "sample": 0}

        if hasattr(self, "vad_model"):
            self.vad_model.reset_states()
        self.logger.debug("FasterWhisper VAD state reset.")

        if hasattr(self, "vad_to_asr_queue"):
            while not self.vad_to_asr_queue.empty():
                try:
                    self.vad_to_asr_queue.get_nowait()
                except Empty:
                    pass
        else:
            self.vad_to_asr_queue = Queue()

    def run_worker(self):
        """Override BaseASRWorker.run_worker: additionally start a dedicated ASR inference thread."""
        # 1. Start the dedicated ASR inference thread, decoupled from the VAD so inference never blocks it
        self.asr_inference_thread = Thread(target=self._run_asr_inference_loop)
        self.asr_inference_thread.start()

        # 2. Run the main audio/VAD loop on this thread (blocking)
        super().run_worker()

        # 3. Wind down: tell the inference thread to stop
        self.vad_to_asr_queue.put(None)

        # 4. Wait for the inference thread to finish every remaining chunk on the queue
        self.asr_inference_thread.join()
        self.logger.info("[ASR] ASR Inference thread joined.")

        # 5. Only then send the completion signal downstream
        self.output_queue.put(None)

    def _run_asr_inference_loop(self):
        """Consumer loop: take VAD events off the queue and either run ASR or handle silence.

        Runs on its own thread so inference never blocks the VAD.
        """
        self.logger.info("[ASR] Inference thread started.")
        while True:
            try:
                item = self.vad_to_asr_queue.get()

                if item is None:
                    break

                event_type = item.get("type")
                args = item.get("args")

                if event_type == "SPEECH":
                    self._transcribe_audio(*args)
                elif event_type == "SILENCE":
                    self._handle_output(SILENCE_TOKEN, *args)

            except Exception as e:
                self.logger.error(f"[ASR] Error in inference loop: {e}", exc_info=True)

        self.logger.info("[ASR] Inference thread stopping.")

    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_final: bool):
        """Buffer audio and process it window by window at the fixed size the VAD requires."""
        full_audio = np.concatenate((self.leftover_audio, audio_chunk))

        offset = 0
        total_len = len(full_audio)

        while offset + self.vad_window_size <= total_len:
            window = full_audio[offset : offset + self.vad_window_size]
            window_tensor = torch.from_numpy(window)
            self._process_vad_window(window_tensor)
            offset += self.vad_window_size

        self.leftover_audio = full_audio[offset:]

    def _process_vad_window(self, chunk: torch.Tensor):
        """Core VAD logic: a state machine deciding where speech starts, continues and ends."""
        current_wall_time = time.perf_counter()

        window_start_sample = self.vad_global_sample_idx
        self.vad_global_sample_idx += self.vad_window_size
        speech_prob = self.vad_model(chunk, self.sampling_rate).item()

        # Silence token: accumulate silence duration and emit a SILENCE event once it crosses the threshold
        if speech_prob < self.vad_threshold_exit:
            self.silence_accumulator += self.vad_window_size
            if self.add_silence_token and self.silence_accumulator >= self.silence_token_samples:
                sil_audio_end = self.vad_global_sample_idx / self.sampling_rate
                sil_audio_start = (self.vad_global_sample_idx - self.silence_token_samples) / self.sampling_rate
                wall_finish = current_wall_time
                # SILENCE_TOKEN must not go straight onto the output queue, or the ordering of text and SIL can be disturbed
                self.vad_to_asr_queue.put({"type": "SILENCE", "args": (wall_finish, wall_finish, sil_audio_start, sil_audio_end)})
                self.silence_accumulator -= self.silence_token_samples
        else:
            self.silence_accumulator = 0

        # 1. Onset: the probability crosses the start threshold; include pre_chunks so the beginning of the speech is not clipped
        if (speech_prob >= self.vad_threshold_start) and not self.triggered:
            self.triggered = True

            start_offset = len(self.pre_chunks) * self.vad_window_size
            self.current_speech_start_info["sample"] = window_start_sample - start_offset
            self.current_speech_start_info["wall"] = current_wall_time - (start_offset / self.sampling_rate)
            self.current_speech_buffer.extend(list(self.pre_chunks))
            self.current_speech_buffer.append(chunk)

        # 2. Continuation: the probability stays above the exit threshold, so speech is ongoing
        elif (speech_prob >= self.vad_threshold_exit) and self.triggered:
            self.current_speech_buffer.append(chunk)
            self.temp_end = 0

        # 3. Possible end: the probability falls below the exit threshold; note the candidate end point and cut the utterance once the silence is long enough
        elif (speech_prob < self.vad_threshold_exit) and self.triggered:
            self.current_speech_buffer.append(chunk)

            if self.temp_end == 0:
                self.temp_end = (len(self.current_speech_buffer) - 1) * self.vad_window_size

            silence_duration = len(self.current_speech_buffer) * self.vad_window_size - self.temp_end

            if silence_duration >= self.vad_pause_samples:
                valid_speech_len = self.temp_end

                # Only send segments long enough for recognition, filtering out very short noise
                if valid_speech_len >= self.min_speech_samples:
                    padding_samples = self.vad_window_size * self.num_post_pad_chunks
                    chunks_to_keep = int((self.temp_end + padding_samples) / self.vad_window_size)
                    chunks_to_keep = min(chunks_to_keep, len(self.current_speech_buffer))

                    final_chunks = self.current_speech_buffer[:chunks_to_keep]

                    self.vad_to_asr_queue.put(
                        {"type": "SPEECH", "args": (final_chunks, self.current_speech_start_info["sample"], self.current_speech_start_info["wall"])}
                    )

                self.triggered = False
                self.current_speech_buffer = []
                self.temp_end = 0

        # 4. Silent state: nothing triggered and no speech, nothing to do
        else:
            pass

        self.pre_chunks.append(chunk)

    def _transcribe_audio(self, chunk_list, segment_start_sample_idx, segment_start_wall_time):
        """Concatenate the chunks and run Faster-Whisper recognition."""
        if not chunk_list:
            return

        audio_np = torch.cat(chunk_list).squeeze().cpu().numpy()

        chunk_duration = len(audio_np) / self.sampling_rate
        true_vad_end_wall = segment_start_wall_time + chunk_duration
        self.pipeline_timer.setdefault("t_vad_speech_end_wall", []).append(true_vad_end_wall)
        self.pipeline_timer.setdefault("t_asr_compute_start", []).append(time.perf_counter())

        if self.use_context_prompt:
            if len(self.prompt_history) > self.max_prompt_chars:
                slice_text = self.prompt_history[-self.max_prompt_chars :]
                first_space = slice_text.find(" ")
                if first_space != -1:
                    current_prompt = slice_text[first_space + 1 :]
                else:
                    current_prompt = slice_text
            else:
                current_prompt = self.prompt_history
            segments_generator, _ = self.asr_model.transcribe(audio_np, beam_size=5, language=self.language, initial_prompt=current_prompt)
        else:
            segments_generator, _ = self.asr_model.transcribe(audio_np, beam_size=5, language=self.language)

        segments = list(segments_generator)

        self.pipeline_timer.setdefault("t_asr_compute_end", []).append(time.perf_counter())

        if not segments:
            return

        wall_finish = time.perf_counter()
        full_text = "".join([s.text for s in segments])

        if not full_text or is_punctuation(full_text):
            return

        # Soft filter: drop common hallucination phrases only when confidence is very low (avg_logprob < -0.3)
        for hallucination_trigger in HALLUCINATION_TRIGGERS_FASTER:
            if hallucination_trigger in full_text:
                avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)

                if avg_logprob < -0.3:
                    self.logger.warning(f"[ASR] Possible hallucination filtered: '{full_text}' (avg_logprob: {avg_logprob:.2f})")
                    return
                else:
                    self.logger.warning(f"[ASR] Possible hallucination not filtered: '{full_text}' (avg_logprob: {avg_logprob:.2f})")

        if self.use_context_prompt:
            self.prompt_history += full_text

        rel_start = segments[0].start
        chunk_duration = len(audio_np) / self.sampling_rate
        rel_end = min(segments[-1].end, chunk_duration)
        audio_start = (segment_start_sample_idx / self.sampling_rate) + rel_start
        audio_end = (segment_start_sample_idx / self.sampling_rate) + rel_end

        self.pipeline_timer.setdefault("t_asr_output_text", []).append(wall_finish)

        self.logger.info(f"[ASR] Detected: {full_text}")
        self._handle_output(full_text, segment_start_wall_time, wall_finish, audio_start, audio_end)

    def _handle_output(self, text, wall_receive, wall_finish, audio_start, audio_end):
        """Put (text, wall_receive, wall_finish, audio_start, audio_end) onto the output queue.

        Silence tokens are only sent while the LLM is in the LISTEN state.
        """
        if text and text != SILENCE_TOKEN:
            self.output_queue.put((text, wall_receive, wall_finish, audio_start, audio_end))
            self.pipeline_timer.setdefault("t_speech_end_audio_relative", []).append(audio_end)

        elif text == SILENCE_TOKEN:
            if self.add_silence_token and self.shared_state.get("llm_state") == "LISTEN":
                self.output_queue.put((SILENCE_TOKEN, wall_receive, wall_finish, audio_start, audio_end))


class RealtimeFileStream:
    """Mimics sounddevice.InputStream but reads its data from a file."""

    def __init__(self, file_path, sampling_rate=16000, dtype="float32", real_time=True):
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.real_time = real_time
        self.start_time = None
        self.cursor = 0

        self.audio_data, _ = librosa.load(file_path, sr=sampling_rate, mono=True, dtype=dtype)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self.start_time = time.perf_counter()
        self.cursor = 0

    def stop(self):
        pass

    def read(self, frames):
        """Read `frames` samples, with the same signature as sounddevice.InputStream.read.

        Reading faster than real time blocks; reading slower returns immediately so the stream can catch up.
        """
        if self.start_time is None:
            self.start()

        if self.real_time:
            target_time = self.start_time + ((self.cursor + frames) / self.sampling_rate)
            current_time = time.perf_counter()
            wait_time = target_time - current_time

            if wait_time > 0:
                time.sleep(wait_time)

        remaining = len(self.audio_data) - self.cursor

        if remaining <= 0:
            return np.array([], dtype=self.dtype), False

        read_count = min(frames, remaining)
        chunk = self.audio_data[self.cursor : self.cursor + read_count]
        self.cursor += read_count

        return chunk, False


class QueueInputStream:
    """A queue-backed input stream modelled on RealtimeFileStream.

    Two properties:
    1. Time synchronization: sleep when reading faster than real time (handles a bulk upload arriving at once).
    2. Strict blocking: never pad with zeros when data is short; block on the queue instead (handles network stalls).
    """

    def __init__(self, q: Queue, sampling_rate: int, channels: int = 1, dtype="float32"):
        self.q = q
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.dtype = dtype

        self.start_time = None
        self.samples_read = 0
        self.closed = False

        self.leftover_buffer = np.zeros((0, channels), dtype=dtype)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.closed = True

    def start(self):
        self.start_time = time.perf_counter()
        self.samples_read = 0

    def read(self, frames: int):
        """Read `frames` samples: rate-limit by time first (sleep if reading too fast), then block on data (wait on the queue if short)."""
        if self.start_time is None:
            self.start()

        target_time = self.start_time + ((self.samples_read + frames) / self.sampling_rate)
        current_time = time.perf_counter()
        wait_time = target_time - current_time

        if wait_time > 0:
            time.sleep(wait_time)

        collected_data = []
        collected_frames = 0

        if len(self.leftover_buffer) > 0:
            take = min(len(self.leftover_buffer), frames)
            collected_data.append(self.leftover_buffer[:take])
            self.leftover_buffer = self.leftover_buffer[take:]
            collected_frames += take

        while collected_frames < frames:
            try:
                chunk = self.q.get()

                if chunk is None:
                    self.closed = True
                    break

                if isinstance(chunk, np.ndarray):
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(-1, 1)
                    if chunk.shape[1] != self.channels:
                        chunk = chunk[:, : self.channels]

                needed = frames - collected_frames

                if len(chunk) <= needed:
                    collected_data.append(chunk)
                    collected_frames += len(chunk)
                else:
                    to_take = chunk[:needed]
                    to_store = chunk[needed:]

                    collected_data.append(to_take)
                    self.leftover_buffer = to_store
                    collected_frames += needed

            except Exception as e:
                print(f"[QueueInputStream] Error: {e}")
                break

        self.samples_read += collected_frames

        if len(collected_data) == 0:
            return np.zeros((frames, self.channels), dtype=self.dtype), False

        final_output = np.concatenate(collected_data, axis=0)

        if len(final_output) < frames:
            padding = np.zeros((frames - len(final_output), self.channels), dtype=self.dtype)
            final_output = np.concatenate([final_output, padding], axis=0)

        return final_output, False
