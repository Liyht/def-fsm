"""Strategy-pattern definitions for the LLM workers.

Every implementation follows the BaseLLMWorker interface. The LLM is the central controller of the whole FSM.
"""

import logging
import time
from queue import Empty

import numpy as np
from abc import ABC, abstractmethod

from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE

from def_fsm.utils import is_meaningful_text, STATE_TRANSITION_TOKENS, SLISTEN_STATE_TRANSITION_TOKENS, INTERLOCUTOR_PREFIX


class BaseLLMWorker(ABC):
    """Abstract base class for LLM workers, defining the common strategy interface."""

    def __init__(
        self,
        model_path: str,
        device: str,
        max_tokens: int = 100,
        temperature: float = 0,
        top_p: float = 0.95,
        interlocutor_prefix: str = INTERLOCUTOR_PREFIX,
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.interlocutor_prefix = interlocutor_prefix

        self.kwargs = kwargs  # implementation-specific parameters
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer onto the given device."""
        pass

    def warmup(self):
        """Warm the model up with the system prompt. No-op by default; backends that need it override this."""
        pass

    @abstractmethod
    def run_worker(self):
        """Main loop of the LLM thread. An implementation must:

        1. read text from input_queue,
        2. generate tokens and put them on output_queue,
        3. use stop_event to decide when to exit,
        4. clean up in a finally block (close the model, wind down) so resources are always released,
        5. put None on output_queue as an end sentinel before exiting.
        """
        pass


class FSMLlamaCppWorker(BaseLLMWorker):
    """FSM LLM worker built on llama.cpp."""

    def __init__(
        self,
        system_prompt: str,
        model_path: str,
        device: str,
        max_length: int = 1024,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        verbose: bool = False,
        seed: int = 42,
        lookahead_k: int = 2,
        transition_temperature: float = 0,
        **kwargs,
    ):
        super().__init__(model_path, device, **kwargs)

        self.system_prompt = system_prompt
        # Sampling temperature for state transition tokens, independent of self.temperature for the response.
        # Transitions are discrete control decisions and are decided greedily at 0; the response can use a higher temperature for diversity.
        self.transition_temperature = transition_temperature
        # Context length matching training (max_length in the training config)
        self.max_length = max_length
        self.max_single_inference_tokens = 100  # guards against runaway repetition
        self._tok_cache = {}
        # Look-ahead bound: while in SPEAK, keep generating ahead until num_pending_text reaches K,
        # which just fills the LLM+TTS pipeline so playback never starves. K only has to cover the
        # latency of producing one clause; a larger K wastes generated content on simultaneous speech and lengthens the commit window.
        self.lookahead_k = lookahead_k

        # llama.cpp-specific parameters
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.seed = seed

    def load_model(self):
        self.logger.info(f"Loading Llama.cpp model from {self.model_path}")

        try:
            self.model = Llama(
                model_path=self.model_path,
                split_mode=LLAMA_SPLIT_MODE_NONE,
                main_gpu=int(self.device.split(":")[1]),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                flash_attn=True,  # use the fused attention kernel built into ggml: faster prefill, less memory
                verbose=self.verbose,
                seed=self.seed,
            )
        except Exception as e:
            self.logger.error(f"Failed to load Llama.cpp model: {e}", exc_info=True)
            raise

        self.logger.info("Llama.cpp model loaded.")

        # Cache the id of each state transition token. Each must encode to exactly one token so it can be used with logit_bias.
        self.state_transition_token_ids_map = {}
        for token in STATE_TRANSITION_TOKENS:
            token_ids_list = self.model.tokenizer().encode(token)
            assert len(token_ids_list) == 1
            self.state_transition_token_ids_map[token] = token_ids_list[0]

    def warmup(self):
        """Run one short generation on the system prompt to warm up fully and build its KV cache."""

        try:
            self.logger.info("[LLM-LlamaCpp] Warming up with system prompt...")
            stream = self.model(self.system_prompt, max_tokens=8, temperature=0.0, stream=True)
            for _ in stream:
                pass
            self.logger.info("[LLM-LlamaCpp] Warm-up complete.")
        except Exception as e:
            self.logger.error(f"[LLM-LlamaCpp] Warmup failed: {e}", exc_info=True)

    def load_state(self, shared_state: dict):
        self.shared_state = shared_state
        self.shared_state["llm_state"] = "LISTEN"

    def _fit_to_budget(self, content):
        """Prepend the system prompt, then keep the most recent segments from the tail within a token budget (matching the training context length).

        Per-segment token counts are cached in _tok_cache. The count is a pure function of the segment
        string and so never goes stale, keeping the per-token hot path a table lookup rather than repeated tokenization.
        """

        def chunk_len(chunk):
            n = self._tok_cache.get(chunk)
            if n is None:
                n = self._tok_cache[chunk] = len(self.model.tokenizer().encode(chunk))
            return n

        budget = self.max_length - chunk_len(self.system_prompt) - self.max_single_inference_tokens - 1
        kept, total = [], 0
        for chunk in reversed(content):
            n = chunk_len(chunk)
            if total + n > budget and kept:
                break
            total += n
            kept.append(chunk)
        return self.system_prompt + "".join(reversed(kept))

    def _get_context_from_tape(self):
        """Assemble the LLM context from the shared tape: the system prompt plus the most recent content within budget."""
        with self.shared_state["tape_lock"]:
            content = [chunk for (chunk, ts) in self.shared_state["tape"][1:]]
        return self._fit_to_budget(content)

    def _log_transition_probs(self, logit_bias, picked, label):
        """Log the probability of every transition token at the step that just evaluated the context.

        llama.cpp logprobs require logits_all=True, which computes logits at every position and is far too
        expensive. Instead, after a single max_tokens=1 call, read the underlying last-position logits buffer
        directly: it is exactly the next-token distribution and has not yet seen the sampled token. Applying the same logit_bias and a softmax gives each transition token its probability.
        """
        try:
            logits = np.ctypeslib.as_array(self.model._ctx.get_logits(), shape=(self.model._n_vocab,)).astype(np.float64)
            for tid, b in logit_bias.items():
                logits[tid] += b
            exps = np.exp(logits - logits.max())
            probs_all = exps / exps.sum()
            ranked = ", ".join(
                f"{tok}={probs_all[tid]:.3f}" for tok, tid in sorted(self.state_transition_token_ids_map.items(), key=lambda kv: -probs_all[kv[1]])
            )
            self.logger.info(f"[LLM-FSM] {label}: {ranked} | picked={picked!r}")
        except Exception as e:
            self.logger.debug(f"[LLM-FSM] {label} prob logging skipped: {e}")

    def _inference(self):
        context = self._get_context_from_tape()

        self.logger.info("[LLM-FSM] Running streaming inference.")

        # Use logit_bias to forbid the state transition tokens that are illegal in the current state
        logit_bias = {}
        current_state = self.shared_state["llm_state"]
        if current_state == "LISTEN":
            logit_bias[self.state_transition_token_ids_map["[S.LISTEN.INTERRUPT]"]] = -100
            logit_bias[self.state_transition_token_ids_map["[S.LISTEN.NATURAL]"]] = -100
            logit_bias[self.state_transition_token_ids_map["[C.SPEAK]"]] = -100
        elif current_state == "SPEAK":
            logit_bias[self.state_transition_token_ids_map["[S.SPEAK]"]] = -100
            logit_bias[self.state_transition_token_ids_map["[C.LISTEN]"]] = -100
            # [S.LISTEN.INTERRUPT] can only come from simultaneous speech
            logit_bias[self.state_transition_token_ids_map["[S.LISTEN.INTERRUPT]"]] = -100
        # Forbid the native EOS: a sentence may only end on a state transition token.
        logit_bias[self.model.token_eos()] = -100

        self.shared_state["pipeline_timer"]["t_llm_compute_start"].append(time.perf_counter())

        # Phase 1: the transition probe decides "transition now?" deterministically at self.transition_temperature.
        # On the round where a transition token is actually accepted it always appears as the first token
        # (the response went into the context last round and the token that triggered the exit was discarded),
        # so probing a single token covers the decision. The probe only evaluates the context into the KV cache; the continuation below reuses it by prefix match and evaluates just this one extra token.
        probe = self.model(
            context,
            max_tokens=1,
            temperature=self.transition_temperature,
            top_p=1.0,
            stop=None,
            logit_bias=logit_bias,
        )
        first = probe["choices"][0].get("text", "")
        self.shared_state["pipeline_timer"]["t_llm_output_token"].append(time.perf_counter())

        # Observe the competition between [C.SPEAK], [S.LISTEN.NATURAL] and friends, and any premature yielding.
        self.logger.info(f"[LLM-FSM] probe context ({current_state}): {context!r}")
        self._log_transition_probs(logit_bias, first.strip(), f"probe transition probs ({current_state})")
        hit = first.strip()
        if hit in STATE_TRANSITION_TOKENS:
            # The transition was accepted at transition_temperature
            self._process_llm_output(hit, time.perf_counter())
            return

        # Phase 2: continue the response. Stream on from the probe token, sampling at self.temperature.
        # Sampling happens inside the C layer token by token; Python only accumulates text pieces and closes
        # the stream as soon as a state transition token appears, discarding it so phase 1 re-decides it next round at transition_temperature.
        new_text = first
        transition_token = ""
        buf = first
        stream = self.model.create_completion(
            context + first,
            max_tokens=self.max_single_inference_tokens - 1,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=None,
            logit_bias=logit_bias,
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0].get("text", "")
            if not piece:
                continue
            self.shared_state["pipeline_timer"]["t_llm_output_token"].append(time.perf_counter())
            buf += piece
            # Each state transition token encodes to exactly one token, so it appears atomically within a single piece
            hit = next((t for t in STATE_TRANSITION_TOKENS if t in buf), None)
            if hit:
                transition_token = hit
                new_text = buf[: buf.index(hit)]
                stream.close()  # closing stops further generation and leaves the KV cache in a valid state
                break
            new_text = buf
            self.logger.debug(f"[LLM-FSM] New text: {new_text} | New token: {piece}")
        else:
            self.logger.warning(f"[LLM-FSM] Reached max token limit ({self.max_single_inference_tokens}) without transition token.")

        if is_meaningful_text(new_text):
            # Discard the state transition token that triggered the exit; it is re-decided next round
            self._process_llm_output(new_text, time.perf_counter())
        else:
            self._process_llm_output(transition_token, time.perf_counter())

    def _process_llm_output(self, text, ts_gen):
        """Handle one span of text generated by the LLM.

        The LLM only writes state transition tokens onto the tape; response text in the SPEAK state is written when it is handed to TTS.
        """
        current_state = self.shared_state["llm_state"]

        if text in STATE_TRANSITION_TOKENS:
            if text in SLISTEN_STATE_TRANSITION_TOKENS:
                # A spontaneous turn-end marker. SPEAK->LISTEN is not flipped here but left to the
                # commit rule in the main loop, which waits until playback has fully drained. The marker itself is simply appended to the tape below.
                # [S.LISTEN.INTERRUPT] should only come from simultaneous speech; if it is emitted spontaneously it is still treated as a turn end.
                if text == "[S.LISTEN.INTERRUPT]":
                    self.logger.warning("[LLM-FSM] Unexpected self-emitted [S.LISTEN.INTERRUPT]; will commit as turn-end.")
                else:
                    self.logger.info("[LLM-FSM] Turn-end marker [S.LISTEN.NATURAL] (commit deferred to drain).")
            elif text == "[S.SPEAK]":
                self.shared_state["llm_state"] = "SPEAK"
                self.logger.info("[LLM-FSM] State -> SPEAK")

            elif text == "[C.SPEAK]" or text == "[C.LISTEN]":
                pass

            with self.shared_state["tape_lock"]:
                self.shared_state["tape"].append((text, ts_gen))
                self.logger.debug(f"[LLM-FSM] TAPE APPEND: {text}")
        else:
            if current_state == "SPEAK":
                # Response tokens in the SPEAK state. The tape has a single writer: the LLM writes the response as it generates.
                # `ts_gen` identifies a piece of text, distinguishing otherwise identical strings, and is carried through the TTS queue into the clause record.
                self.logger.info(f"[LLM-FSM] -> TTS: {text}")
                with self.shared_state["tape_lock"]:
                    self.shared_state["tape"].append((text, ts_gen))
                with self.shared_state["tts_num_pending_text_lock"]:
                    self.shared_state["tts_worker_ref"].num_pending_text += 1
                    self.logger.info(
                        f"[LLM-FSM] TTS Pending: {self.shared_state['tts_worker_ref'].num_pending_text - 1} -> {self.shared_state['tts_worker_ref'].num_pending_text}"
                    )
                self.shared_state["llm_to_tts_queue"].put((text, ts_gen))
            else:
                self.logger.info(f"[LLM-FSM] New Content in LISTEN mode: {text}")

    def _locate_clause_entry(self, handle_ts, clause_text):
        """Locate the index of the currently playing clause on the tape.

        Match on the ts handle first (the float identity carried through), falling back to a text match.
        The caller must already hold tape_lock.
        """
        tape = self.shared_state["tape"]
        for i in range(len(tape) - 1, -1, -1):
            if tape[i][1] == handle_ts:
                return i
        for i in range(len(tape) - 1, -1, -1):
            if tape[i][0] == clause_text:
                return i
        return None

    def _context_with_concatenate(self, target_idx, pre_text, asr_chunk):
        """Assemble the context used to adjudicate simultaneous speech.

        It is the tape before the playing clause, plus the prefix already spoken, plus the interlocutor ASR
        text: the causal anchor of what the user had actually heard when they broke in. The real tape is not modified here.
        """
        with self.shared_state["tape_lock"]:
            content = [tok for (tok, _ts) in self.shared_state["tape"][1:target_idx]]
        content.append(pre_text)
        content.append(self.interlocutor_prefix + asr_chunk)
        return self._fit_to_budget(content)

    def _judge_simultaneous_speech(self, context):
        """Adjudicate simultaneous speech with one constrained greedy decoding step.

        Given [...pre][<interlocutor>ASR], the model picks one of the three transitions that are legal in
        the SPEAK state. A strong positive bias on those three makes the next token itself the verdict.
        """
        legal = ("[C.SPEAK]", "[S.LISTEN.INTERRUPT]", "[S.LISTEN.NATURAL]")
        logit_bias = {}
        for t in legal:
            logit_bias[self.state_transition_token_ids_map[t]] = 10

        output = self.model(
            context,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            logit_bias=logit_bias,
        )
        tok = output["choices"][0].get("text", "").strip()

        # Observe the competition between the three legal transitions (positively biased).
        self.logger.info(f"[LLM-FSM] simultaneous_speech judge context: {context!r}")
        self._log_transition_probs(logit_bias, tok, "simultaneous_speech judge probs")

        if tok not in legal:
            self.logger.warning(f"[LLM-FSM] simultaneous_speech judge produced {tok!r}; defaulting [C.SPEAK]")
            tok = "[C.SPEAK]"
        return tok

    def _handle_speak_simultaneous_speech(self, asr_chunk, ts):
        """Handle ASR arriving while the agent is speaking, i.e. simultaneous speech (backchannel / floor-taking interruption / butting-in).

        Ignoring negligible intervals there are three usable time points: when the user started speaking, when they finished (when ASR began), and when ASR finished (now).
        A single anchor is used: playback progress is queried once via progress_at(now), and that one value drives both the continue-or-yield verdict and the tape splice.
        The sizeable delay between the interruption and now (ASR latency plus look-ahead lag) is absorbed at now; only the tiny delay of the judging step itself is ignored. Returns one of:
          "NOPLAY"   -> no playing clause could be located; fall back to a plain append
          "CONTINUE" -> [C.SPEAK]; audio is untouched and generation continues under the num_pending gate
          "STOP"     -> [S.LISTEN.*]; the state will switch back to LISTEN
        """
        tts = self.shared_state["tts_worker_ref"]
        interloc = self.interlocutor_prefix + asr_chunk

        res = tts.progress_at(time.perf_counter())
        if res is None:
            with self.shared_state["tape_lock"]:
                self.shared_state["tape"].append((interloc, ts))
            self.logger.info("[LLM-FSM] simultaneous_speech: nothing playing -> plain append.")
            return "NOPLAY"

        handle_ts, clause_text, split = res
        with self.shared_state["tape_lock"]:
            target = self._locate_clause_entry(handle_ts, clause_text)
        if target is None:
            with self.shared_state["tape_lock"]:
                self.shared_state["tape"].append((interloc, ts))
            self.logger.warning(f"[LLM-FSM] simultaneous_speech: clause entry not found (handle_ts={handle_ts}) -> plain append.")
            return "NOPLAY"

        pre = clause_text[:split]
        context = self._context_with_concatenate(target, pre, asr_chunk)
        decision = self._judge_simultaneous_speech(context)
        now = time.perf_counter()

        if decision == "[C.SPEAK]":
            # Do not yield: the clause plays out as normal and the queued look-ahead tail (after target) is kept.
            # The tape records: prefix / interruption / [C.SPEAK] / the remainder of the clause.
            post = clause_text[split:]
            new_entries = []
            if pre:
                new_entries.append((pre, now))
            new_entries.append((interloc, ts))
            new_entries.append((decision, now))
            if post:
                new_entries.append((post, now))
            with self.shared_state["tape_lock"]:
                self.shared_state["tape"][target : target + 1] = new_entries
            self.logger.info(f"[LLM-FSM] simultaneous_speech -> [C.SPEAK] (continue). split={split} clause={clause_text!r}")
            return "CONTINUE"

        elif decision == "[S.LISTEN.INTERRUPT]":
            # Yield: stop the audio immediately. The cut is at `split`, what had been heard at now; everything
            # from target on -- the unheard remainder of this clause plus any look-ahead tail -- is deleted from
            # the tape, and interrupt() clears the corresponding audio queues in step.
            tts.interrupt()
            new_entries = []
            if pre:
                new_entries.append((pre, now))
            new_entries.append((interloc, ts))
            new_entries.append((decision, now))
            with self.shared_state["tape_lock"]:
                self.shared_state["tape"][target:] = new_entries
            # The state flip is left to the commit rule: interrupt() has set num_pending to 0 and the tape now
            # ends in [S.LISTEN.INTERRUPT], so the next iteration commits SPEAK->LISTEN.
            self.logger.info(f"[LLM-FSM] simultaneous_speech -> [S.LISTEN.INTERRUPT]. cut@{split} clause={clause_text!r}")
            return "STOP"

        elif decision == "[S.LISTEN.NATURAL]":
            # The agent was finishing anyway. Let the current clause play out rather than cutting it,
            # record it whole, drop the unheard look-ahead tail, and yield the floor.
            tts.flush_after_current()
            with self.shared_state["tape_lock"]:
                orig_ts = self.shared_state["tape"][target][1]
                self.shared_state["tape"][target:] = [
                    (clause_text, orig_ts),
                    (interloc, ts),
                    (decision, now),
                ]
            # The state flip is left to the commit rule: the clause keeps playing uninterrupted, and once it has
            # drained (num_pending == 0) with [S.LISTEN.NATURAL] at the end of the tape, SPEAK->LISTEN is committed.
            self.logger.info(f"[LLM-FSM] simultaneous_speech -> [S.LISTEN.NATURAL] (finish clause, drop tail, listen). clause={clause_text!r}")
            return "STOP"

        else:
            raise AssertionError(f"[LLM-FSM] simultaneous_speech: unexpected decision {decision!r}")

    def _transit_speak_to_listen_if_drained(self, last_token):
        """The single place where SPEAK->LISTEN happens.

        The turn-end marker reaches the end of the tape as soon as it is generated, but the state flip waits
        until playback has fully drained (num_pending == 0). Until then llm_state stays "SPEAK", so a user
        interruption during the drain still goes through simultaneous speech instead of being mistaken for a new turn. Returns True if a commit happened.
        """
        if (
            self.shared_state["llm_state"] == "SPEAK"
            and last_token in SLISTEN_STATE_TRANSITION_TOKENS
            and self.shared_state["tts_worker_ref"].num_pending_text == 0
        ):
            self.shared_state["llm_state"] = "LISTEN"
            self.logger.info("[LLM-FSM] State -> LISTEN (commit: turn-end drained).")
            return True
        return False

    def run_worker(self):
        """Main FSM event loop."""
        self.logger.info("[LLM-FSM] Worker thread started, FSM loop active.")
        trigger_inference = False
        last_tape = []

        try:
            # This loop must never block, or ASR cannot trigger an interrupt in time and simultaneous speech is delayed
            while not self.shared_state["stop_event"].is_set():
                # Read the token at the end of the tape; snapshot when the tape changed, and log outside the lock
                with self.shared_state["tape_lock"]:
                    last_token = self.shared_state["tape"][-1][0]

                    tape_changed = self.shared_state["tape"] != last_tape
                    if tape_changed:
                        last_tape = self.shared_state["tape"].copy()

                if tape_changed:
                    self.logger.info(f"[LLM-FSM] Tape updated: {last_tape}")

                # The only SPEAK->LISTEN: flip once the trailing turn-end marker has fully drained from playback
                self._transit_speak_to_listen_if_drained(last_token)

                # Check ASR every iteration, so simultaneous speech is still caught between clauses while generating ahead
                try:
                    asr_item = self.shared_state["asr_to_llm_queue"].get_nowait()
                    if asr_item is None:
                        self.logger.info("[LLM-FSM] Received the ASR stop signal (None).")
                        continue  # skip this iteration and wait for stop_event

                    (asr_chunk, receive_ts, ts, audio_start, audio_end) = asr_item

                    if self.shared_state["llm_state"] == "SPEAK":
                        # ASR while in SPEAK means simultaneous speech, adjudicated from the single anchor.
                        # CONTINUE: later iterations let the driver (num_pending < K) generate on from the spliced tape.
                        # STOP: the tape already ends in a turn-end marker, and the commit rule flips SPEAK->LISTEN once playback drains.
                        action = self._handle_speak_simultaneous_speech(asr_chunk, ts)
                        if action == "NOPLAY":
                            trigger_inference = True
                        self.logger.info("[LLM-FSM] Event: ASR during SPEAK (simultaneous speech).")
                    else:
                        # In LISTEN: plainly append the interlocutor text and generate a response right away
                        with self.shared_state["tape_lock"]:
                            self.shared_state["tape"].append((self.interlocutor_prefix + asr_chunk, ts))
                        trigger_inference = True
                        self.logger.info("[LLM-FSM] Event: ASR input during LISTEN.")
                except Empty:
                    pass  # the ASR queue is empty, which is normal

                # Bounded look-ahead driver: while in SPEAK, the tape does not end in a turn-end marker and the
                # buffer still has room (num_pending < K), keep generating ahead. Re-read the tail, because the ASR branch above may just have spliced the tape.
                if not trigger_inference:
                    with self.shared_state["tape_lock"]:
                        frontier = self.shared_state["tape"][-1][0]
                    if (
                        self.shared_state["llm_state"] == "SPEAK"
                        and frontier not in SLISTEN_STATE_TRANSITION_TOKENS
                        and self.shared_state["tts_worker_ref"].num_pending_text < self.lookahead_k
                    ):
                        trigger_inference = True

                # Run one inference if triggered
                if trigger_inference:
                    self._inference()

                    trigger_inference = False
                else:
                    time.sleep(0.001)  # keep this: yields the CPU when idle so the loop does not busy-wait a core

        except Exception as e:
            self.logger.error(f"[LLM-FSM] Worker error: {e}", exc_info=True)
        finally:
            self.logger.info("[LLM-FSM] Cleaning up and stopping...")
            self.shared_state["llm_to_tts_queue"].put(None)
