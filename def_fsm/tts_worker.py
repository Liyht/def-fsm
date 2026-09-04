"""Strategy-pattern definitions for the TTS workers.

Every worker follows the same FSM protocol and sends three kinds of message to the playback queue:
START_TEXT / AUDIO_CHUNK / END_TEXT.
"""

import torch
import time
import logging
from queue import Queue, Empty
from threading import Event
from abc import ABC, abstractmethod
from collections import deque
import contextlib

import sounddevice as sd
import threading
from typing import Optional, Generator

from def_fsm.utils import clear_queue

from kokoro import KPipeline


# The LLM often emits typographic Unicode punctuation (curly quotes, em dashes), which the misaki
# G2P used by Kokoro can fail to phonemize, returning empty audio ("No audio generated"). They are
# therefore mapped back to ASCII. Every substitution is length-preserving and 1:1 (one code point to
# one character), so the char offsets in _synthesize_chunks still line up with the original clause text and the simultaneous-speech cut point is unaffected.
_TTS_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": ".",
        " ": " ",  # non-breaking space
    }
)


class BaseTTSWorker(ABC):
    """Abstract base class for FSM TTS workers.

    It holds the shared FSM logic: load_state, interrupt, reset, the synthesis loop and the playback loop.
    A subclass only has to implement load_model() and _synthesize_chunks().
    """

    def __init__(self, device: str, enable_playback: bool = True, **kwargs):
        self.device = device
        self.sampling_rate = 24000
        self.enable_playback = enable_playback
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tts_interrupt_event = Event()
        self.playback_interrupt_event = Event()

        self.output_audio_queue = None
        self.num_pending_text = 0

        # Precise playback-progress tracking.
        # _clause_records is a ring of playback records for the most recent clauses. Each is a dict:
        #   text:           the clause text being spoken
        #   start_wall:     perf_counter when the first audio chunk started playing (None before then)
        #   end_wall:       perf_counter when it finished or was interrupted (None while playing)
        #   played_samples: cumulative samples handed to playback
        #   timeline:       list of (cumulative samples, char_end) checkpoints, where char_end is how many
        #                   characters of `text` have been spoken by then
        self.progress_lock = threading.Lock()
        self._clause_records = deque(maxlen=8)
        self._current_record = None
        # Soft-stop flag for [S.LISTEN.NATURAL] under look-ahead: let the playing clause finish, then drop the
        # queued look-ahead tail. Set by flush_after_current() and acted on by the playback loop at END_TEXT.
        self._stop_after_current = False

    def set_output_queue(self, q: Queue):
        self.output_audio_queue = q

    @abstractmethod
    def load_model(self):
        """Load the TTS model onto the given device."""
        pass

    def warmup(self):
        """Run one dummy synthesis to warm up (kernel compilation and other one-off costs). No-op by default; backends that need it override this."""
        pass

    @abstractmethod
    def _synthesize_chunks(self, text: str) -> Generator[tuple, None, None]:
        """Synthesize audio for `text`, yielding ``(audio_tensor, char_end)`` chunk by chunk.

        char_end is how many characters of `text` have been spoken by the end of the chunk.
        self.tts_interrupt_event must be checked between chunks so interruption works.
        """
        pass

    def load_state(self, shared_state):
        self.tts_input_queue = shared_state["llm_to_tts_queue"]
        self.tts_to_playback_queue = Queue()
        self.stop_event = shared_state["stop_event"]
        self.pipeline_timer = shared_state["pipeline_timer"]
        self.num_pending_text_lock = shared_state["tts_num_pending_text_lock"]

    def interrupt(self):
        """Hard-interrupt synthesis and playback synchronously, clearing the queues."""
        self.tts_interrupt_event.set()
        self.playback_interrupt_event.set()
        self._stop_after_current = False  # a hard interrupt takes precedence over a soft stop

        clear_queue(self.tts_input_queue)
        clear_queue(self.tts_to_playback_queue)

        with self.num_pending_text_lock:
            self.num_pending_text = 0

    def flush_after_current(self):
        """Soft stop for [S.LISTEN.NATURAL] under look-ahead.

        The clause currently playing is allowed to finish, but the queued look-ahead tail (the clauses after
        it) is dropped. Unlike interrupt(), the interrupt event is not set, so the current clause plays to its
        natural end, and the playback loop discards the rest of the queue at its END_TEXT.

        num_pending is set to 1 if something is playing and 0 otherwise, so the END_TEXT of the current clause
        drops it to 0 and triggers the commit rule in the LLM. (Rare race: if a tail clause is still being
        synthesized a short burst of audio can escape, but the tape has already dropped the tail, so it is only a slight artifact.)
        """
        clear_queue(self.tts_input_queue)  # drop the tail that has not been synthesized yet
        self._stop_after_current = True
        with self.progress_lock:
            playing = self._current_record is not None and self._current_record["start_wall"] is not None
        with self.num_pending_text_lock:
            self.num_pending_text = 1 if playing else 0

    def reset(self):
        """Reset every stateful variable in preparation for the next dialogue."""
        self.logger.info("Resetting TTS state...")
        self.tts_interrupt_event.clear()
        self.playback_interrupt_event.clear()
        self._stop_after_current = False

        with self.num_pending_text_lock:
            self.num_pending_text = 0

        with self.progress_lock:
            self._clause_records.clear()
            self._current_record = None

    def progress_at(self, t_wall: float):
        """How far into a clause the agent had actually spoken at perf_counter time t_wall.

        Returns ``(tape_ts, clause_text, char_offset)``. tape_ts is the stable handle of the tape entry for
        that clause, assigned at START_TEXT, and char_offset is how many leading characters of clause_text had
        been spoken by t_wall; None if no clause covers that instant. The Kokoro word-level timeline is used
        when available, which is exact; otherwise this falls back to the characters-per-second estimate of the backend.
        """
        with self.progress_lock:
            chosen = None
            for rec in self._clause_records:
                if rec["start_wall"] is not None and rec["start_wall"] <= t_wall:
                    chosen = rec  # take the last clause that had started before t_wall
            if chosen is None:
                return None

            text = chosen["text"]
            tape_ts = chosen["tape_ts"]
            end_wall = chosen["end_wall"]
            if end_wall is not None and t_wall >= end_wall:
                # The clause finished before t_wall, so the whole thing was spoken
                return tape_ts, text, len(text)

            elapsed = max(0.0, t_wall - chosen["start_wall"])
            approx_samples = int(elapsed * self.sampling_rate)
            timeline = list(chosen["timeline"])  # snapshot while holding the lock

        # An empty timeline means no chunk has been played yet, so nothing has been spoken.
        char_off = 0
        for cum_samples, char_end in timeline:
            if cum_samples <= approx_samples:
                char_off = char_end
            else:
                break
        return tape_ts, text, max(0, min(len(text), char_off))

    def run_tts_worker(self):
        """Main synthesis loop: read text off the input queue, synthesize it and forward it to the playback queue."""
        try:
            input_queue = self.tts_input_queue
            output_queue = self.tts_to_playback_queue
            stop_event = self.stop_event
            pipeline_timer = self.pipeline_timer

            torch.cuda.set_device(self.device)
            self.logger.info("TTS thread started, waiting for LLM tokens...")

            while not stop_event.is_set():
                if self.tts_interrupt_event.is_set():
                    self.logger.info("Interrupt detected, clearing queues.")
                    clear_queue(input_queue)
                    clear_queue(output_queue)
                    self.tts_interrupt_event.clear()
                    continue

                try:
                    queue_item = input_queue.get(timeout=1.0)
                    if queue_item is None:
                        break
                    # The payload is (text, ts_gen). ts_gen is the stable handle the LLM assigned when it wrote
                    # the tape, carried through so the clause record and any truncation feedback share one identity.
                    next_text, ts_gen = queue_item

                    pipeline_timer["t_tts_receive_token"].append(time.perf_counter())
                    self.logger.info(f"Synthesizing: {next_text.replace(chr(10), '\\n')}")

                    # 1. Send the START marker, carrying the handle through
                    output_queue.put(("START_TEXT", (next_text, ts_gen)))

                    # 2. Stream out the audio chunks
                    try:
                        chunk_count = 0
                        pipeline_timer.setdefault("t_tts_compute_start", []).append(time.perf_counter())
                        for chunk, char_end in self._synthesize_chunks(next_text):
                            chunk_count += 1
                            pipeline_timer["t_tts_output_chunk"].append(time.perf_counter())
                            if stop_event.is_set():
                                break
                            output_queue.put(("AUDIO_CHUNK", (chunk, char_end)))

                        if chunk_count == 0:
                            if self.tts_interrupt_event.is_set():
                                # Pre-empted by a barge-in: synthesis returned at the very start with zero chunks.
                                # That is expected (the model yielded the floor, so this clause should not be voiced), not a failure.
                                self.logger.info(f"TTS clause preempted by interrupt (no audio): {next_text!r}")
                            else:
                                self.logger.warning(f"No audio generated for: '{next_text}' | repr={next_text!r}")
                    except Exception as e:
                        self.logger.error(f"TTS inference error: {e}", exc_info=True)

                    # 3. Send the END marker
                    if not self.tts_interrupt_event.is_set():
                        output_queue.put(("END_TEXT", next_text))

                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"TTS worker error: {e}", exc_info=True)

            output_queue.put(None)
            self.logger.info("TTS thread stopped.")

        except Exception as e:
            self.logger.error(f"run_tts_worker failed: {e}", exc_info=True)

    def _playback_loop_internal(self, input_queue: Queue, stop_event: Event, pipeline_timer: dict, stream: Optional["sd.OutputStream"]):
        """The single playback loop, handling START_TEXT / AUDIO_CHUNK / END_TEXT messages.

        `stream` is an sd.OutputStream, or None for dummy mode where nothing is actually played.
        """
        current_sentence_start_time = None
        current_text = None
        current_tape_ts = None  # the stable handle shared with the tape entry
        is_playing = False

        while not stop_event.is_set():
            try:
                if self.playback_interrupt_event.is_set():
                    self.logger.info("Interrupt received, clearing playback queue.")

                    # Truncation belongs to the simultaneous-speech path, whose cut point from progress_at is far
                    # more accurate than an elapsed * chars-per-second estimate. Playback therefore no longer reports TRUNCATE and just resets state, records end_wall and clears the queues.
                    is_playing = False
                    current_text = None
                    current_tape_ts = None
                    current_sentence_start_time = None
                    with self.progress_lock:
                        if self._current_record is not None:
                            self._current_record["end_wall"] = time.perf_counter()
                            self._current_record = None
                    clear_queue(input_queue)

                    self.playback_interrupt_event.clear()
                    self.logger.info("Playback queue cleared, interrupt handled.")
                    continue

                msg = input_queue.get(timeout=1.0)
                if msg is None:
                    self.logger.info("Stop signal received.")
                    break

                msg_type, msg_data = msg

                if msg_type == "START_TEXT":
                    # msg_data is (text, ts_gen). ts_gen is the handle the LLM assigned when writing the tape,
                    # carried all the way here so the clause record and the tape entry share one identity rather
                    # than minting a new handle. No feedback goes back to the LLM: the content is already on the
                    # tape, and the look-ahead driver throttles on num_pending rather than on a pulse signal.
                    text, ts_gen = msg_data
                    current_text = text
                    current_tape_ts = ts_gen
                    with self.progress_lock:
                        self._current_record = {
                            "text": text,
                            "tape_ts": ts_gen,
                            "start_wall": None,
                            "end_wall": None,
                            "played_samples": 0,
                            "timeline": [],
                        }
                        self._clause_records.append(self._current_record)
                    self.logger.info(f"Started playing: {text}")

                elif msg_type == "AUDIO_CHUNK":
                    pipeline_timer["t_playback_chunk"].append(time.perf_counter())

                    chunk, char_end = msg_data
                    chunk_np = chunk.cpu().numpy()
                    self.logger.debug(f"Chunk Len: {len(chunk_np)}, Max Amp: {chunk_np.max():.4f}")

                    if self.output_audio_queue:
                        self.output_audio_queue.put((self.sampling_rate, chunk_np))

                    if not is_playing:
                        current_sentence_start_time = time.perf_counter()
                        is_playing = True
                        with self.progress_lock:
                            if self._current_record is not None:
                                self._current_record["start_wall"] = current_sentence_start_time

                    with self.progress_lock:
                        rec = self._current_record
                        if rec is not None:
                            rec["played_samples"] += len(chunk_np)
                            rec["timeline"].append((rec["played_samples"], char_end))

                    if stream:
                        stream.write(chunk_np)
                    else:
                        duration = len(chunk_np) / self.sampling_rate
                        time.sleep(duration)

                elif msg_type == "END_TEXT":
                    is_playing = False
                    current_text = None
                    current_tape_ts = None
                    with self.progress_lock:
                        if self._current_record is not None:
                            self._current_record["end_wall"] = time.perf_counter()
                            self._current_record = None
                    with self.num_pending_text_lock:
                        if self.num_pending_text > 0:
                            self.num_pending_text -= 1
                        else:
                            self.logger.warning(f"num_pending_text={self.num_pending_text} cannot decrement, set to 0")
                            self.num_pending_text = 0
                    self.logger.info(f"Finished playing: {msg_data}, TTS Pending: {self.num_pending_text + 1} -> {self.num_pending_text}")

                    # Soft stop ([S.LISTEN.NATURAL] under look-ahead): the current clause has finished, so drop
                    # the queued look-ahead tail so the agent does not keep talking after yielding the floor.
                    if self._stop_after_current:
                        clear_queue(input_queue)
                        self._stop_after_current = False
                        self.logger.info("Soft-stop: look-ahead tail dropped after natural-end clause.")

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in playback loop: {e}", exc_info=True)

    def run_playback_worker(self):
        """Playback thread entry point: take audio chunks off the TTS queue and play them."""
        input_queue = self.tts_to_playback_queue
        stop_event = self.stop_event
        pipeline_timer = self.pipeline_timer

        if self.enable_playback:
            self.logger.info("Playback thread started (Audio Enabled).")
            stream_ctx = sd.OutputStream(samplerate=self.sampling_rate, channels=1, dtype="float32")
        else:
            self.logger.info("Playback thread started (Dummy Mode).")
            stream_ctx = contextlib.nullcontext()

        try:
            with stream_ctx as stream:
                self._playback_loop_internal(input_queue, stop_event, pipeline_timer, stream=stream)
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}", exc_info=True)
        finally:
            self.logger.info("Playback thread stopped.")


class FSMKokoroWorker(BaseTTSWorker):
    """FSM TTS worker built on Kokoro TTS, chunking the audio so it can be interrupted mid-clause."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, lang_code: str = "a", repo_id: str = "hexgrad/Kokoro-82M", **kwargs):
        super().__init__(**kwargs)
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.repo_id = repo_id
        # Chunk size in samples for mid-clause interruption, about 0.5 s at 24 kHz
        self.AUDIO_CHUNK_SIZE = int(0.5 * self.sampling_rate)

    def load_model(self):
        torch.cuda.set_device(self.device)
        self.pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=self.repo_id,
            device=self.device,
        )

    def warmup(self):
        """Run one dummy synthesis to warm up Kokoro TTS."""
        self.logger.info("Warming up Kokoro TTS...")
        for _ in self.pipeline("Warming up.", voice=self.voice, speed=self.speed):
            pass
        self.logger.info("Kokoro TTS warm-up complete.")

    def _synthesize_chunks(self, text: str) -> Generator[tuple, None, None]:
        """Synthesize `text`, yielding ``(audio_chunk, char_end)`` chunk by chunk.

        char_end is how many leading characters of the original `text` have been fully spoken by the end of
        this chunk, derived from the word-level timestamps of Kokoro (result.tokens[*].end_ts). It is
        monotonic and never exceeds len(text). When the surface form of a token cannot be found in the
        original text (a number or symbol rewritten by normalization, say), char_end does not advance: under-reporting is the safe direction for the causal anchor of simultaneous speech.
        """
        # Normalize typographic punctuation to ASCII so misaki G2P always produces audio. The mapping is 1:1
        # and length-preserving, so the char offsets from text.find below still align with the original clause text.
        text = text.translate(_TTS_PUNCT_MAP)
        sample_base = 0  # cumulative sample count across results
        char_cursor = 0  # search cursor into the original text
        char_end = 0  # offset of the last character spoken (monotonic)
        for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None):
            if self.tts_interrupt_event.is_set():
                return
            audio = result.audio
            if audio is None:
                continue

            # Token boundaries within this result: (absolute end sample, character offset)
            boundaries = []
            for t in result.tokens or []:
                piece = t.text or ""
                if piece:
                    idx = text.find(piece, char_cursor)
                    if idx != -1:
                        char_cursor = idx + len(piece)
                        # Absorb trailing whitespace so char_end lands in the gap between words
                        while char_cursor < len(text) and text[char_cursor].isspace():
                            char_cursor += 1
                if t.end_ts is None:
                    continue
                abs_end = sample_base + int(round(t.end_ts * self.sampling_rate))
                boundaries.append((abs_end, char_cursor))

            bi = 0
            for offset in range(0, len(audio), self.AUDIO_CHUNK_SIZE):
                if self.tts_interrupt_event.is_set():
                    return
                chunk = audio[offset : offset + self.AUDIO_CHUNK_SIZE]
                chunk_abs_end = sample_base + offset + len(chunk)
                while bi < len(boundaries) and boundaries[bi][0] <= chunk_abs_end:
                    char_end = boundaries[bi][1]
                    bi += 1
                yield chunk, char_end

            sample_base += len(audio)

