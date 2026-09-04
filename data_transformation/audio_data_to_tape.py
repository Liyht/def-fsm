"""Serialize classified turn-taking segments into FSM training tapes.

The interlocutor side keeps its raw ASR text; the agent side is the reference transcript
with punctuation restored. Entry point: run this file to build the Switchboard and Fisher
tape datasets.
"""

import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import json
import glob
import re
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from icecream import ic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import random
import sys
from deepmultilingualpunctuation import PunctuationModel
import random
from collections import defaultdict

from statistics_event import compute_metrics, plot_event_stats, LABEL_ORDER
from turn_taking_event import merge_and_classify_turn_taking_events, search_last_nonSIL_label, TIME_SCALE
from audio_dataset import SwitchboardDataset, BaseDataset, FisherDataset
from def_fsm.paths import expand
from def_fsm.utils import (
    STATE_TRANSITION_TOKENS,
    SLISTEN_STATE_TRANSITION_TOKENS,
    is_interloctor_text,
    add_interlocutor_prefix_to_asr,
    self_text_process,
    is_self_text,
    INTERLOCUTOR_PREFIX,
    SILENCE_TOKEN,
    SILENCE_TOKEN_DUR,
)

WORKER_PUNCTUATION_MODEL = None


class TapeGenerator:
    """
    Converts merged word-level data and turn-taking event segments into a serialized token tape.
    Encapsulates the Finite State Machine (FSM) logic for Full-Duplex LLM training data generation.
    """

    def __init__(
        self,
        interlocutor_stream,
        self_stream,
        segments,
        remove_gap_sil_before_self=True,
        add_clisten_first=False,
        complement_silence=False,
    ):
        self.interlocutor_stream = interlocutor_stream
        self.self_stream = self_stream
        self.remove_gap_sil_before_self = remove_gap_sil_before_self
        self.add_clisten_first = add_clisten_first
        self.complement_silence = complement_silence

        # State variables
        self.tape = []
        self.u_idx = 0
        self.a_idx = 0
        self.fsm_state = "LISTEN"  # Initial State
        self.tape_to_seg_map = []

        # Preparation
        self.segments = []
        floor_holder = None
        for i, seg in enumerate(segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            label = seg["label"]
            active_spks = seg["active_spks"]
            action_initiator = seg["action_initiator"]

            seg_self_words, seg_interlocutor_words = self._collect_segment_words(seg_start, seg_end)
            seg["self_words"] = seg_self_words
            seg["interlocutor_words"] = seg_interlocutor_words

            # floor_holder changes only when the speaker holding the floor does:
            # at T (the new holder is the active speaker) and at IF (the
            # interrupter takes it). C, BC, IB, P and G inherit it.
            if label == "IF":
                floor_holder = action_initiator
            elif label == "T":
                assert len(active_spks) == 1
                floor_holder = active_spks[0]
            seg["floor_holder"] = floor_holder

            self.segments.append(seg)

    def generate(self):
        # 0 is the interlocutor, 1 is self.
        # In every overlap the segment text is ordered self first, then interlocutor: content occurring
        # at the same instant means self had already committed to speaking beforehand. For example:
        # 1. self makes a floor-taking interruption: self then interlocutor (break in, hear it, keep going)
        # 2. the interlocutor does: self then interlocutor (speak, get interrupted, stop)

        # As a rule the local tape starts with a state transition token and does not end with one

        # 1. Insert every switch token
        for seg_idx, seg in enumerate(self.segments):
            try:
                if seg["label"] == "T":
                    if seg["action_initiator"] == 0:
                        # User turn-taking
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], join_token="[C.LISTEN]")
                        if seg_idx - 1 >= 0 and self.segments[seg_idx - 1]["label"] == "G":
                            # Natural turn-taking, and gap before has already set [S.LISTEN.NATURAL]
                            local_tape = ["[C.LISTEN]", *interlocutor_words]
                        elif seg_idx - 1 < 0:
                            # Start of the audio
                            local_tape = interlocutor_words
                        else:
                            # Natural turn-taking, but no gap before
                            local_tape = ["[S.LISTEN.NATURAL]", *interlocutor_words]
                    elif seg["action_initiator"] == 1:
                        # Agent turn-taking
                        local_tape = ["[S.SPEAK]", *seg["self_words"]]
                    else:
                        self._print_local(seg_idx)
                        raise
                elif seg["label"] == "P":
                    seg_dur = seg["end"] - seg["start"]
                    if seg["floor_holder"] == 0:
                        # User's pause
                        silence_interlocutor_words = self._complement_silence_token(seg["interlocutor_words"], seg_dur)
                        interlocutor_words = self._wrap_interlocutor_words(silence_interlocutor_words, join_token="[C.LISTEN]")
                        local_tape = ["[C.LISTEN]", *interlocutor_words] if interlocutor_words else []

                    elif seg["floor_holder"] == 1:
                        # Agent's pause
                        assert len(seg["self_words"]) == 0, f"Error! {seg['self_words']=}"
                        local_tape = []
                    else:
                        self._print_local(seg_idx)
                        raise
                elif seg["label"] == "G":
                    seg_dur = seg["end"] - seg["start"]
                    if seg["floor_holder"] == 0:
                        # Gap between interlocutor and self
                        # Hope Agent respond as soon as possible, so we remove <SIL> here
                        if self.remove_gap_sil_before_self:
                            local_tape = []
                        else:
                            silence_interlocutor_words = self._complement_silence_token(seg["interlocutor_words"], seg_dur)
                            interlocutor_words = self._wrap_interlocutor_words(silence_interlocutor_words, join_token="[C.LISTEN]")
                            local_tape = ["[C.LISTEN]", *interlocutor_words]
                    elif seg["floor_holder"] == 1:
                        # Gap between self and interlocutor
                        silence_interlocutor_words = self._complement_silence_token(seg["interlocutor_words"], seg_dur)
                        interlocutor_words = self._wrap_interlocutor_words(silence_interlocutor_words, join_token="[C.LISTEN]")
                        local_tape = ["[S.LISTEN.NATURAL]", *interlocutor_words]

                    elif seg["floor_holder"] is None:
                        # Start of the audio
                        silence_interlocutor_words = self._complement_silence_token(seg["interlocutor_words"], seg_dur)
                        interlocutor_words = self._wrap_interlocutor_words(silence_interlocutor_words, join_token="[C.LISTEN]")
                        local_tape = ["[C.LISTEN]", *interlocutor_words] if self.add_clisten_first else interlocutor_words

                    else:
                        self._print_local(seg_idx)
                        raise
                elif seg["label"] == "IF":
                    if seg["action_initiator"] == 0:
                        # User's Floor-Taking Interruption
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], join_token="[C.LISTEN]")
                        local_tape = ["[C.SPEAK]", *seg["self_words"], *interlocutor_words, "[C.LISTEN]"]
                        # find and replace first [C.LISTEN] to "[S.LISTEN.INTERRUPT]"
                        for i in range(len(local_tape)):
                            if local_tape[i] == "[C.LISTEN]":
                                local_tape[i] = "[S.LISTEN.INTERRUPT]"
                                break
                    elif seg["action_initiator"] == 1:
                        # Agent's Floor-Taking Interruption
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], remove_sil=True, join_token="[C.SPEAK]")
                        local_tape = ["[S.SPEAK]", *seg["self_words"], *interlocutor_words]
                    else:
                        self._print_local(seg_idx)
                        raise
                elif seg["label"] == "IB" or seg["label"] == "BC":
                    if seg["action_initiator"] is None:
                        # Both User and Agent are Butting-in: the initiator is whoever did not hold the floor
                        if seg["floor_holder"] is not None:
                            seg["action_initiator"] = 1 - seg["floor_holder"]
                        else:
                            self._print_local(seg_idx)
                            raise

                    if seg["action_initiator"] == 0:
                        # User's Butting-in
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], remove_sil=True, join_token="[C.SPEAK]")
                        local_tape = ["[C.SPEAK]", *seg["self_words"], *interlocutor_words]
                    elif seg["action_initiator"] == 1:
                        # Agent's Backchannel / Butting-in
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], remove_sil=True, join_token="[C.LISTEN]")
                        local_tape = ["[S.SPEAK]", *seg["self_words"], *interlocutor_words, "[C.LISTEN]"]
                        replacement = "[S.LISTEN.NATURAL]" if seg["label"] == "BC" else "[S.LISTEN.INTERRUPT]"
                        for i in range(len(local_tape)):
                            if local_tape[i] == "[C.LISTEN]":
                                local_tape[i] = replacement
                                break
                    else:
                        self._print_local(seg_idx)
                        raise

                elif seg["label"] == "C":
                    if seg["action_initiator"] == 0:
                        # User's continuation
                        interlocutor_words = self._wrap_interlocutor_words(seg["interlocutor_words"], join_token="[C.LISTEN]")
                        local_tape = ["[C.LISTEN]", *interlocutor_words] if self.tape[-1] not in STATE_TRANSITION_TOKENS else interlocutor_words
                    elif seg["action_initiator"] == 1:
                        # Agent's continuation
                        local_tape = ["[C.SPEAK]", *seg["self_words"]] if self.tape[-1] not in STATE_TRANSITION_TOKENS else seg["self_words"]
                    else:
                        self._print_local(seg_idx)
                        raise
                else:
                    self._print_local(seg_idx)
                    raise
            except Exception as e:
                ic.enable()
                ic(self.segments[max(0, seg_idx - 2) : min(len(self.segments), seg_idx + 2)])
                raise

            local_tape = self._merge_continuous_state_transition_tokens(local_tape)
            local_tape = self._remove_potential_interlocutor_self_gap(local_tape)
            self.tape.extend(local_tape)
            self.tape_to_seg_map.extend([seg_idx] * len(local_tape))
        return self.tape, self.tape_to_seg_map

    def _collect_segment_words(self, seg_start, seg_end):
        """Collects User and Agent words falling into the current segment time window."""
        seg_start_int = int(round(seg_start * TIME_SCALE))
        seg_end_int = int(round(seg_end * TIME_SCALE))

        # 1. Collect Agent Words
        seg_self_words = []
        while self.a_idx < len(self.self_stream):
            _, text, s, e = self.self_stream[self.a_idx]

            s_int = int(round(s * TIME_SCALE))
            e_int = int(round(e * TIME_SCALE))

            if s_int >= seg_end_int:
                break
            # Reaching here means s < seg_end, so the word overlaps the segment iff e > seg_start
            if e_int > seg_start_int:
                seg_self_words.append(text)

            self.a_idx += 1

        # 2. Collect user words (note that these include <SIL>)
        seg_interlocutor_words = []
        while self.u_idx < len(self.interlocutor_stream):
            _, text, s, e = self.interlocutor_stream[self.u_idx]

            s_int = int(round(s * TIME_SCALE))
            e_int = int(round(e * TIME_SCALE))

            if s_int >= seg_end_int:
                # Belongs to next segment
                break
            if e_int > seg_start_int:
                seg_interlocutor_words.append(text)

            self.u_idx += 1

        return seg_self_words, seg_interlocutor_words

    def _add_prefix(self, text_list):
        return [add_interlocutor_prefix_to_asr(text) for text in text_list]

    def _join_state_transition_token(self, text_list, join_token):
        new_text_list = []
        for text in text_list:
            new_text_list.append(text)
            new_text_list.append(join_token)
        if new_text_list:
            new_text_list.pop()
        return new_text_list

    def _remove_SIL_text(self, text_list):
        return [text for text in text_list if SILENCE_TOKEN not in text]

    def _wrap_interlocutor_words(self, text_list, remove_sil=False, add_prefix=True, join_token=None):
        if remove_sil:
            text_list = self._remove_SIL_text(text_list)
        if add_prefix:
            text_list = self._add_prefix(text_list)
        if join_token:
            text_list = self._join_state_transition_token(text_list, join_token)
        return text_list

    def _remove_potential_interlocutor_self_gap(self, local_tape):
        if not self.remove_gap_sil_before_self:
            return local_tape

        def _is_gap():
            i = 0
            for i in range(len(self.tape) - 1, -1, -1):
                if SILENCE_TOKEN in self.tape[i] or self.tape[i] == "[C.LISTEN]":
                    continue
                else:
                    if self.tape[i].startswith(INTERLOCUTOR_PREFIX):
                        return True, i
                    else:
                        return False, i
            return False, i

        is_gap, i = _is_gap()

        if local_tape and local_tape[0] == "[S.SPEAK]" and self.tape and is_gap:
            while self.tape and (SILENCE_TOKEN in self.tape[-1] or self.tape[-1] == "[C.LISTEN]"):
                self.tape = self.tape[: i + 1]
                self.tape_to_seg_map = self.tape_to_seg_map[: i + 1]
        return local_tape

    def _complement_silence_token(self, seg_interlocutor_words, seg_dur):
        assert seg_interlocutor_words.count(SILENCE_TOKEN) == len(seg_interlocutor_words), f"Error! {seg_interlocutor_words=} for Gap/Pause segment"
        if not self.complement_silence:
            return seg_interlocutor_words
        if int(seg_dur // SILENCE_TOKEN_DUR) > seg_interlocutor_words.count(SILENCE_TOKEN):
            seg_interlocutor_words = [SILENCE_TOKEN] * int(seg_dur // SILENCE_TOKEN_DUR)
        return seg_interlocutor_words

    def _merge_continuous_state_transition_tokens(self, local_tape):
        """Collapse a pair of consecutive state transition tokens at the boundary
        between self.tape and local_tape into a single token.

        The merge is determined entirely by the FSM state immediately before the
        first token fires and immediately after the second token fires, giving a
        2x2 lookup over {LISTEN, SPEAK}^2:

            (L, L) -> [C.LISTEN]                  (continuation in LISTEN)
            (S, S) -> [C.SPEAK]                   (continuation in SPEAK)
            (L, S) -> [S.SPEAK]                   (genuine switch L->S)
            (S, L) -> [S.LISTEN.*]                (genuine switch S->L;
                                                subtype taken from the
                                                [S.LISTEN.*] token in the pair)

        The pair [C.SPEAK]+[C.LISTEN] in the (S, L) case is structurally impossible:
        a segment ending in [C.SPEAK] leaves the agent holding the floor, while one
        starting with [C.LISTEN] requires the user to hold it. We assert this.
        """
        # Tokens implying the FSM is in the SPEAK state immediately *before* they fire.
        # - [C.SPEAK]      : continuation while speaking, so previously SPEAK
        # - [S.LISTEN.*]   : switching from SPEAK to LISTEN, so previously SPEAK
        _SPEAK_BEFORE = {"[C.SPEAK]"} | SLISTEN_STATE_TRANSITION_TOKENS

        # Tokens implying the FSM is in the SPEAK state immediately *after* they fire.
        # - [C.SPEAK], [S.SPEAK]
        _SPEAK_AFTER = {"[C.SPEAK]", "[S.SPEAK]"}

        if not (local_tape and self.tape):
            return local_tape

        prev_tok, next_tok = self.tape[-1], local_tape[0]
        if prev_tok not in STATE_TRANSITION_TOKENS or next_tok not in STATE_TRANSITION_TOKENS:
            return local_tape

        state_before = "S" if prev_tok in _SPEAK_BEFORE else "L"
        state_after = "S" if next_tok in _SPEAK_AFTER else "L"

        if (state_before, state_after) == ("L", "L"):
            merged = "[C.LISTEN]"
        elif (state_before, state_after) == ("S", "S"):
            merged = "[C.SPEAK]"
        elif (state_before, state_after) == ("L", "S"):
            merged = "[S.SPEAK]"
        else:  # (S, L)
            # Exactly one of the two tokens is [S.LISTEN.*]; take its subtype.
            # The other token is [C.SPEAK] or [C.LISTEN], absorbed as continuation.
            prev_is_slisten = prev_tok in SLISTEN_STATE_TRANSITION_TOKENS
            next_is_slisten = next_tok in SLISTEN_STATE_TRANSITION_TOKENS
            assert prev_is_slisten or next_is_slisten, (
                f"Impossible (S,L) boundary: {prev_tok}+{next_tok}. "
                f"[C.SPEAK]+[C.LISTEN] cannot arise from any valid pair of "
                f"adjacent segments: they disagree on who holds the floor."
            )
            merged = prev_tok if prev_is_slisten else next_tok

        # Apply the merge while preserving the original implementation's
        # tape_to_seg_map attribution:
        #   - If the merged token equals prev_tok, keep it on self.tape side
        #     (drop local_tape[0]); the merged token stays attributed to the
        #     previous segment.
        #   - Otherwise, drop self.tape[-1] and overwrite local_tape[0]; the
        #     merged token is attributed to the current segment.
        if merged == prev_tok:
            return local_tape[1:]
        else:
            self.tape.pop()
            self.tape_to_seg_map.pop()
            local_tape[0] = merged
            return local_tape

    def _print_local(self, seg_idx):
        ic.enable()
        ic(f"{len(self.segments)=}, {seg_idx=}")
        ic(self.segments[max(0, seg_idx - 2) : min(len(self.segments), seg_idx + 3)])


def merge_tape_self_tokens(tape, tape_map):
    """Re-segment the agent's own words on the tape from word level to clause level.

    The generator emits one entry per agent word. Here they are joined back into continuous
    text, then split into clauses on restored punctuation, with [C.SPEAK] between clauses.
    ``tape_map`` (token index -> segment index) is kept in step with the result.
    """
    merged_tape = []
    merged_map = []  # kept in step with the merged text
    last_is_self_token = False

    def _is_self_content_boundary(token):
        if token in STATE_TRANSITION_TOKENS - set(["[C.SPEAK]"]) or is_interloctor_text(token):
            return True
        return False

    # Step 1: join the self words into continuous text; the ASR side is already continuous.
    for i, token in enumerate(tape):
        current_seg_idx = tape_map[i]

        if _is_self_content_boundary(token):
            merged_tape.append(token)
            merged_map.append(current_seg_idx)
            last_is_self_token = False
        elif token == "[C.SPEAK]":
            if last_is_self_token and i + 1 < len(tape) and not _is_self_content_boundary(tape[i + 1]):
                # Throw away "[C.SPEAK]" between self words
                pass
            else:
                merged_tape.append(token)
                merged_map.append(current_seg_idx)
                last_is_self_token = False
        else:
            if last_is_self_token:
                merged_tape[-1] += " " + token
            else:
                merged_tape.append(token)
                merged_map.append(current_seg_idx)
            last_is_self_token = True

    # Step 2: restore punctuation to segment the text and insert "[C.SPEAK]"
    split_merged_tape = []
    split_merged_map = []  # kept in step with the split text

    for text, seg_idx in zip(merged_tape, merged_map):
        if not is_self_text(text):
            split_merged_tape.append(text)
            split_merged_map.append(seg_idx)
        else:
            # Split into sentences and then clauses; every clause inherits the original seg_idx
            processed_texts = self_text_process(text)
            split_merged_tape.extend(processed_texts)
            split_merged_map.extend([seg_idx] * len(processed_texts))

    return split_merged_tape, split_merged_map


def is_FSM_tape_valid(tape):
    """Replay a tape through the FSM and check every transition is legal.

    Returns ``(True, None)`` or ``(False, index of the first offending token)``.
    """
    state = "LISTEN"
    last_state_transition_token = None
    possible_transition_to_listen = {None, "[C.LISTEN]", "[S.LISTEN.NATURAL]", "[S.LISTEN.INTERRUPT]"}
    possible_transition_to_speak = {None, "[C.SPEAK]", "[S.SPEAK]"}

    ic.enable()
    for i, token in enumerate(tape):
        if token == "[S.SPEAK]":
            if state != "LISTEN":
                ic()
                return False, i
            if last_state_transition_token not in possible_transition_to_listen:
                ic()
                return False, i

            last_state_transition_token = token
            state = "SPEAK"

        elif token in SLISTEN_STATE_TRANSITION_TOKENS:
            if state != "SPEAK":
                ic()
                return False, i
            if last_state_transition_token not in possible_transition_to_speak:
                ic()
                return False, i

            last_state_transition_token = token
            state = "LISTEN"

        elif token == "[C.SPEAK]":
            if state != "SPEAK":
                ic()
                return False, i
            if last_state_transition_token not in possible_transition_to_speak:
                ic()
                return False, i

            last_state_transition_token = token

        elif token == "[C.LISTEN]":
            if state != "LISTEN":
                ic()
                return False, i
            if last_state_transition_token not in possible_transition_to_listen:
                ic()
                return False, i

            last_state_transition_token = token

        elif token.startswith(INTERLOCUTOR_PREFIX):
            if tape and i - 1 >= 0 and tape[i - 1].startswith(INTERLOCUTOR_PREFIX):
                ic()
                return False, i

        all_state_transition_tokens = possible_transition_to_listen | possible_transition_to_speak - {None}
        if i > 0 and token in all_state_transition_tokens and tape[i - 1] in all_state_transition_tokens:
            ic()
            return False, i
    ic.disable()
    return True, None


def _replace_user_words_in_asr_stream(asr_with_sil, clipped_user_words):
    """Rebuild the with-SIL user stream so its non-SIL words carry the clipped
    timestamps from merge_and_classify_turn_taking_events. SIL placeholders
    are passed through unchanged. The two sets share order by word start, so
    we substitute them positionally to avoid timestamp-based lookups.
    """
    sil_words = [w for w in asr_with_sil if w[1] == SILENCE_TOKEN]
    new_stream = sil_words + list(clipped_user_words)
    new_stream.sort(key=lambda x: x[2])
    return new_stream


def _process_single_entry(file_info, dataset_instance, vis_root, remove_gap_sil_before_self, clip_small_overlaps=True):
    global WORKER_PUNCTUATION_MODEL
    if WORKER_PUNCTUATION_MODEL is not None:
        dataset_instance.punctuation_model = WORKER_PUNCTUATION_MODEL
    # 1. Prepare the data. This logic moved out of the generator so it can run in parallel.
    prepared_data = dataset_instance.process_single_dialogue(file_info)

    # 2. Interleave the interlocutor ASR with the self ground truth and classify the turn-taking events
    streams_to_merge = prepared_data["asr_stream_without_sil"] + prepared_data["trans_stream"]
    segments, clipped_user_words = merge_and_classify_turn_taking_events(streams_to_merge, clip_small_overlaps=clip_small_overlaps)

    # Propagate clipped timestamps into the with-SIL stream consumed by
    # TapeGenerator so segment classification and word collection agree on
    # where boundary words live.
    asr_stream = _replace_user_words_in_asr_stream(prepared_data["asr_stream"], clipped_user_words)

    # 3. Generate Tape
    generator = TapeGenerator(asr_stream, prepared_data["trans_stream"], segments, remove_gap_sil_before_self=remove_gap_sil_before_self)
    raw_tape, raw_tape_map = generator.generate()

    merged_tape, merged_tape_map = merge_tape_self_tokens(raw_tape, raw_tape_map)

    # 4. Validate
    valid, error_idx = is_FSM_tape_valid(merged_tape)
    if not valid:
        ic.enable()
        ic(f"Invalid tape for {prepared_data['meta']['id']} at token index {error_idx}")

        # Segment index where the error occurred
        if 0 <= error_idx < len(merged_tape_map):
            seg_idx = merged_tape_map[error_idx]

            ic(f"\n--- DEBUG INFO for {prepared_data['meta']['id']} ---")
            ic(f"Error Token: {merged_tape[error_idx]}")

            # Print the current segment together with its neighbours
            start_seg = max(0, seg_idx - 4)
            end_seg = min(len(segments), seg_idx + 4)

            for i in range(start_seg, end_seg):
                prefix = ">> ERROR SEG >> " if i == seg_idx else "                "
                ic(f"{prefix}Segment {i}: {segments[i]}")

            # Print the local tape context
            start_tape = max(0, error_idx - 5)
            end_tape = min(len(merged_tape), error_idx + 5)
            ic(f"Local Tape: {merged_tape[start_tape:end_tape]}\n")

            # Print the local tape alongside its segment ids
            start_tape = max(0, error_idx - 5)
            end_tape = min(len(merged_tape), error_idx + 6)

            local_tokens = merged_tape[start_tape:end_tape]
            local_seg_ids = merged_tape_map[start_tape:end_tape]

            ic("\nLocal Tape Context (Token -> SegID):")
            for i, (token, sid) in enumerate(zip(local_tokens, local_seg_ids)):
                abs_idx = start_tape + i
                marker = " <--- ERROR HERE" if abs_idx == error_idx else ""
                # Print aligned
                ic(f"  [{abs_idx}] {token:<25} -> Seg {sid}{marker}")
            ic("------------------------------------------------\n")

        else:
            ic("Error index out of bounds for map lookup.")

        ic.disable()
        return None

    counts = defaultdict(int)
    durations = defaultdict(float)
    total_duration = 0.0
    for seg in segments:
        label = seg["label"]
        dur = max(0.0, seg["end"] - seg["start"])
        total_duration += dur
        counts[label] += 1
        durations[label] += dur

    # 5. Visualize
    if vis_root:
        _save_visualization(dataset_instance, vis_root, prepared_data["meta"]["id"], merged_tape)

    # 6. Finalize
    sample = prepared_data["meta"]
    sample["tape"] = merged_tape

    sample["stats"] = {"counts": dict(counts), "durations_sec": dict(durations), "total_duration_sec": total_duration}

    if random.random() < 0.002 / 16:
        ic.enable()
        ic(sample)
        ic.disable()
    return sample


def process_dataset_pipeline(
    dataset_instance: BaseDataset,
    vis_root: str = None,
    broken_conv_ids: list = None,
    remove_gap_sil_before_self: bool = True,
    num_workers=16,
    resume=True,
    checkpoint_dir=None,
    checkpoint_prefix="part",
    batch_size=500,
    only_keep_both_directions_exist: bool = True,
    clip_small_overlaps: bool = True,
):
    # 1. Scan and Prepare
    dataset_instance.scan_files(broken_ids=broken_conv_ids, only_keep_both_directions_exist=only_keep_both_directions_exist)

    if vis_root:
        os.makedirs(vis_root, exist_ok=True)

    # Resume logic
    processed_ids = set()
    existing_samples = []
    next_chunk_idx = 0

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if resume and checkpoint_dir:
        # Scan for every file matching prefix_*.json
        pattern = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_*.json")
        found_files = glob.glob(pattern)

        ic(f"Scanning checkpoints in {checkpoint_dir}...")

        for f_path in found_files:
            try:
                # Parse the file index, e.g. part_3.json -> 3
                basename = os.path.basename(f_path)
                match = re.search(rf"{checkpoint_prefix}_(\d+)\.json", basename)
                if match:
                    idx = int(match.group(1))
                    if idx >= next_chunk_idx:
                        next_chunk_idx = idx + 1

                # Read the data
                with open(f_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    if isinstance(chunk_data, list):
                        for sample in chunk_data:
                            if "id" in sample:
                                processed_ids.add(sample["id"])
                            elif "meta" in sample and "id" in sample["meta"]:
                                # Kept for compatibility with the older format
                                processed_ids.add(sample["meta"]["id"])
                        existing_samples.extend(chunk_data)
            except Exception as e:
                ic(f"Error loading checkpoint {f_path}: {e}")
                continue

        ic(f"Resuming: Found {len(existing_samples)} samples in {len(found_files)} files. Next file index: {next_chunk_idx}")

    # 2. Parallel Processing
    # Drop the tasks that were already processed
    all_file_infos = []
    for unique_key, info in dataset_instance.id_to_info.items():
        if unique_key not in processed_ids:
            all_file_infos.append(info)

    ic(f"Starting processing with {num_workers} threads. Remaining tasks: {len(all_file_infos)}")

    # Nothing left to do: return what was already read
    if not all_file_infos:
        return existing_samples

    final_samples = []  # every sample produced by this run
    current_batch = []  # the batch of samples currently buffered

    process_func = partial(
        _process_single_entry,
        dataset_instance=dataset_instance,
        vis_root=vis_root,
        remove_gap_sil_before_self=remove_gap_sil_before_self,
        clip_small_overlaps=clip_small_overlaps,
    )

    # Setup GPU Queue for workers
    # Create a queue containing GPU IDs [0, 1, 2, 3, 0, 1...] matching num_workers
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for i in range(num_workers):
        gpu_queue.put(i % 4)  # Cycle through 4 GPUs

    # Pass queue to initializer
    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_initializer, initargs=(gpu_queue,)) as executor:
        results_iterator = tqdm(executor.map(process_func, all_file_infos), total=len(all_file_infos), desc="Processing")

        for res in results_iterator:
            if res is not None:
                final_samples.append(res)
                current_batch.append(res)

                # Flush once the batch is full
                if len(current_batch) >= batch_size:
                    save_name = f"{checkpoint_prefix}_{next_chunk_idx}.json"
                    save_path = os.path.join(checkpoint_dir, save_name)

                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(current_batch, f, ensure_ascii=False, indent=2)

                    current_batch = []
                    next_chunk_idx += 1

    # Loop finished: save the partial final batch
    if len(current_batch) > 0:
        save_name = f"{checkpoint_prefix}_{next_chunk_idx}.json"
        save_path = os.path.join(checkpoint_dir, save_name)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(current_batch, f, ensure_ascii=False, indent=2)
        ic(f"Saved final batch {next_chunk_idx} ({len(current_batch)} samples)")

    # Return everything, old and new
    total_samples = existing_samples + final_samples
    ic(f"Process Complete. Total samples: {len(total_samples)}")

    return total_samples


def find_first_speaker_in_tape(tape):
    """Return "interlocutor", "self", or None if the tape carries no speech at all."""
    for text in tape:
        if text not in STATE_TRANSITION_TOKENS and SILENCE_TOKEN not in text:
            if text.startswith(INTERLOCUTOR_PREFIX):
                return "interlocutor"
            else:
                return "self"
    return None


def gen_random_silence(min_silence=0, max_silence=8, mode="uniform", **kwargs):
    """
    Generate a random number of silence tokens.

    Args:
        min_silence (int): minimum number of silence units.
        max_silence (int): maximum number of silence units.
        mode (str): the random distribution to draw from.
            - 'uniform': uniform, i.e. fully random.
            - 'triangular': can be skewed towards shorter or longer silences.
            - 'gaussian': a natural bell curve.
        **kwargs:
            - with mode='triangular', 'mode_value' sets the peak.
            - with mode='gaussian', 'mu' and 'sigma' set the mean and standard deviation.
    """

    count = 0

    if mode == "uniform":
        # Every length is equally likely
        count = random.randint(min_silence, max_silence)

    elif mode == "triangular":
        # A triangular distribution models "usually short, occasionally long" silences.
        # The peak defaults to min_silence, biasing towards short silences.
        low = min_silence
        high = max_silence
        peak = kwargs.get("mode_value", min_silence)
        count = int(random.triangular(low, high, peak))

    elif mode == "gaussian":
        # A Gaussian is the most natural-looking distribution
        mu = kwargs.get("mu", (min_silence + max_silence) / 2)
        sigma = kwargs.get("sigma", (max_silence - min_silence) / 4)
        count = int(random.gauss(mu, sigma))

    # Clamp into the allowed range. This matters most for the Gaussian, which can otherwise
    # produce negative values or overshoot the maximum.
    count = max(min_silence, min(count, max_silence))

    # Build the token list
    new_silence = [f"{INTERLOCUTOR_PREFIX}{SILENCE_TOKEN}", "[C.LISTEN]"] * count

    return new_silence


def save_dataset(
    final_samples,
    save_root,
    test_size=0.2,
    val_size=0.1,
    seed=42,
    save_type="json",  # json or arrow
    sub_folder_prefix="setting",
):
    """
    Split the data into two groups by who speaks first:
    - first speaker is interlocutor -> save_root/assistant/
    - first speaker is self -> save_root/user/
    Splits are made on conversation id, which prevents train/test leakage.
    """
    ic.enable()
    # 1. Collect every distinct conversation_id
    all_conv_ids = list(set(s["conversation_id"] for s in final_samples))

    # 2. Shuffle the ids and split them
    random.seed(seed)
    all_conv_ids.sort()  # sorting makes the split deterministic
    random.shuffle(all_conv_ids)

    total_ids = len(all_conv_ids)
    num_test = int(total_ids * test_size)
    num_valid = int(total_ids * val_size)
    num_train = total_ids - num_test - num_valid

    # Build the id sets
    train_ids = set(all_conv_ids[:num_train])
    val_ids = set(all_conv_ids[num_train : num_train + num_valid])
    test_ids = set(all_conv_ids[num_train + num_valid :])

    ic(f"Split by ID: {len(train_ids)} Train, {len(val_ids)} Val, {len(test_ids)} Test conversations.")

    # 3. Initialize the containers,
    #    structured as data_buckets[role][split] = [samples]
    data_buckets = {
        "assistant": {"train": [], "val": [], "test": []},
        "user": {"train": [], "val": [], "test": []},
    }

    # 4. Distribute the samples
    done_conv_map = {}  # some conversations disagree on who spoke first between the two tapes; those are assigned at random
    for sample in final_samples:
        conv_id = sample["conversation_id"]
        if conv_id in done_conv_map:
            if done_conv_map[conv_id] == "user":
                self_role = "assistant"
            else:
                self_role = "user"
        else:
            if find_first_speaker_in_tape(sample["tape"]) == "interlocutor":
                self_role = "assistant"
                silence = gen_random_silence(min_silence=0, max_silence=8)
                if silence and sample["tape"] and sample["tape"][0] in STATE_TRANSITION_TOKENS:
                    silence = silence[:-1]
                sample["tape"] = silence + sample["tape"]
                done_conv_map[conv_id] = "assistant"
            elif find_first_speaker_in_tape(sample["tape"]) == "self":
                self_role = "user"
                done_conv_map[conv_id] = "user"
            else:
                continue

        # Determine the split
        if conv_id in train_ids:
            split_key = "train"
        elif conv_id in val_ids:
            split_key = "val"
        else:
            split_key = "test"

        if sample:
            data_buckets[self_role][split_key].append(sample)
        else:
            raise

    # 5. Save the files
    if save_type == "json":
        for role, splits in data_buckets.items():
            role_dir = os.path.join(save_root, f"{sub_folder_prefix}_{role}")
            os.makedirs(role_dir, exist_ok=True)

            for split_name, samples in splits.items():
                file_out = os.path.join(role_dir, f"{split_name}.json")
                with open(file_out, "w", encoding="utf-8") as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                ic(f"Saved {len(samples)} samples to {file_out}")

    # To emit the Arrow format while keeping the DatasetDict structure:
    elif save_type == "arrow":
        for role, splits in data_buckets.items():
            role_dir = os.path.join(save_root, f"{sub_folder_prefix}_{role}")
            # Build the DatasetDict
            dd_mapping = {}
            for split_name, samples in splits.items():
                dd_mapping[split_name] = Dataset.from_list(samples)

            if dd_mapping:
                dataset_dict = DatasetDict(dd_mapping)
                dataset_dict.save_to_disk(role_dir)
                ic(f"Saved Arrow DatasetDict to {role_dir}")
    else:
        raise


def _save_visualization(dataset_instance, vis_root, unique_key, merged_tape):
    os.makedirs(vis_root, exist_ok=True)
    ref_save_path = os.path.join(vis_root, f"sw{unique_key}_ref.txt")
    dataset_instance.merge_trans_to_ref(unique_key, ref_save_path)

    tape_str = "".join(merged_tape)
    tape_str = tape_str.replace("[S.LISTEN", "\n[S.LISTEN").replace("[S.SPEAK]", "\n[S.SPEAK]")
    with open(os.path.join(vis_root, f"sw{unique_key}_tape.txt"), "w", encoding="utf-8") as f:
        f.write(tape_str)


def worker_initializer(gpu_queue=None):
    ic.configureOutput(prefix="", argToStringFunction=str)
    ic.configureOutput(outputFunction=lambda *a: print(*a, file=sys.stderr))
    ic.disable()

    if gpu_queue is not None:
        global WORKER_PUNCTUATION_MODEL
        try:
            gpu_id = gpu_queue.get()
            WORKER_PUNCTUATION_MODEL = PunctuationModel(device=gpu_id)
        except Exception as e:
            print(f"Worker model init failed: {e}", file=sys.stderr)


def aggregate_and_save_stats(samples, dataset_name, save_dir, tag):
    """Accumulate the per-dialogue statistics, print them and let statistics_event.py save the figure."""
    print(f"\nAggregating statistics for {dataset_name} ({tag})...")
    all_counts = defaultdict(int)
    all_durations = defaultdict(float)
    total_duration = 0.0
    num_dialogues = 0

    for sample in samples:
        if "stats" not in sample:
            continue
        num_dialogues += 1
        s = sample["stats"]
        total_duration += s.get("total_duration_sec", 0.0)

        for k, v in s.get("counts", {}).items():
            all_counts[k] += v
        for k, v in s.get("durations_sec", {}).items():
            all_durations[k] += v

    if num_dialogues == 0:
        print("No stats found in samples. Skipping stats generation.")
        return

    raw_stats = {
        "num_dialogues": num_dialogues,
        "total_duration_sec": total_duration,
        "counts": dict(all_counts),
        "durations_sec": dict(all_durations),
    }

    # Compute events per minute and duration share with the helpers in statistics_event.py
    freq_per_min, dur_pct = compute_metrics(raw_stats)

    # Wrap it in a dict keyed by dataset_name, as plot_event_stats expects
    full_stats = {dataset_name: {**raw_stats, "freq_per_min": freq_per_min, "dur_pct": dur_pct}}

    # Print to the terminal
    print(f"\n--- Statistics for {dataset_name} ({tag}) ---")
    print(f"Dialogues: {num_dialogues}  Total duration: {total_duration / 3600:.2f} h")
    print(f"{'Label':<5} {'Count':>10} {'Dur(s)':>12} {'Evt/min':>10} {'Dur%':>8}")
    for label in LABEL_ORDER:
        c = all_counts.get(label, 0)
        d = all_durations.get(label, 0.0)
        print(f"{label:<5} {c:>10d} {d:>12.1f} {freq_per_min.get(label, 0):>10.3f} {dur_pct.get(label, 0):>8.2f}")

    # Save the JSON and the figure
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"statistics_event_{dataset_name}_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)

    pdf_path = os.path.join(save_dir, f"statistics_event_{dataset_name}_{tag}.pdf")
    try:
        plot_event_stats(full_stats, save_path=pdf_path)
        print(f"Saved stats JSON to {json_path}")
        print(f"Saved figures to {pdf_path} (and .png)")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    ic.configureOutput(prefix="", argToStringFunction=str)
    ic.configureOutput(outputFunction=lambda *a: print(*a, file=sys.stderr))
    ic.disable()

    for target_dataset in ["switchboard", "fisher"]:
        for asr_method in ["faster", "simul"]:
            # asr_setting_postfix / other_postfix only name the output directories, so several
            # variants of the pipeline can coexist under DATA_ROOT.
            asr_setting_postfix = "setting"
            other_postfix = "_update"
            remove_gap_sil_before_self = False
            # Trim user ASR word timestamps that produce sub-50ms spurious overlaps with agent
            # IPUs (jitter), preserving backchannels and never dropping a word.
            clip_small_overlaps = True

            num_workers = 16

            ckpt_dir = expand(f"${{DATA_ROOT}}/temp_tape_{asr_method}_{asr_setting_postfix}{other_postfix}_checkpoints/{target_dataset}_{asr_setting_postfix}")

            if target_dataset == "switchboard":
                # 1. Load configuration / Bad IDs
                bad_asr_cov_id_list = json.load(
                    open(expand(f"${{DATA_ROOT}}/broken_asr_and_wer/bad_asr_switchboard_{asr_method}_{asr_setting_postfix}_threshold0.3.json"), "r")
                )

                # 2. Instantiate the specific dataset (Switchboard)
                dataset = SwitchboardDataset(
                    transcript_root=expand("${SWITCHBOARD_ROOT}/transcripts"),
                    asr_root=expand(f"${{DATA_ROOT}}/switchboard_asr_{asr_method}_{asr_setting_postfix}"),
                    add_punctuation_to_self=True,
                )

                # 3. Run the generic pipeline
                final_samples = process_dataset_pipeline(
                    dataset_instance=dataset,
                    vis_root=expand("${DATA_ROOT}/switchboard_visualization"),
                    broken_conv_ids=bad_asr_cov_id_list,
                    remove_gap_sil_before_self=remove_gap_sil_before_self,
                    num_workers=num_workers,
                    resume=True,
                    checkpoint_dir=ckpt_dir,  # the checkpoint directory
                    checkpoint_prefix="part",  # filenames become part_0.json, part_1.json, ...
                    batch_size=500,  # flush every 500 samples
                    only_keep_both_directions_exist=False,
                    clip_small_overlaps=clip_small_overlaps,
                )

                tag = f"{asr_method}_{asr_setting_postfix}{other_postfix}"
                stat_save_dir = expand("${DATA_ROOT}/statistics/turntaking_event")
                aggregate_and_save_stats(final_samples, "Switchboard", stat_save_dir, tag)

                save_dataset(
                    final_samples,
                    save_root=expand("${DATA_ROOT}/switchboard_tape_dataset/"),
                    save_type="json",
                    test_size=0.2,
                    val_size=0.1,
                    sub_folder_prefix=f"{asr_method}_{asr_setting_postfix}{other_postfix}",
                )

            elif target_dataset == "fisher":
                # 1. Load configuration / Bad IDs
                bad_asr_cov_id_list = json.load(
                    open(expand(f"${{DATA_ROOT}}/broken_asr_and_wer/bad_asr_fisher_{asr_method}_{asr_setting_postfix}_threshold0.3.json"), "r")
                )

                # 2. Instantiate the specific dataset (Fisher)
                dataset = FisherDataset(
                    transcript_root=expand("${DATA_ROOT}/fisher_refined_clause/"),
                    asr_root=expand(f"${{DATA_ROOT}}/fisher_asr_{asr_method}_{asr_setting_postfix}"),
                    add_punctuation_to_self=True,
                )

                # 3. Run the generic pipeline
                final_samples = process_dataset_pipeline(
                    dataset_instance=dataset,
                    vis_root=expand("${DATA_ROOT}/fisher_visualization"),
                    broken_conv_ids=bad_asr_cov_id_list,
                    remove_gap_sil_before_self=remove_gap_sil_before_self,
                    num_workers=num_workers,
                    resume=True,
                    checkpoint_dir=ckpt_dir,
                    checkpoint_prefix="part",
                    batch_size=500,
                    clip_small_overlaps=clip_small_overlaps,
                )

                tag = f"{asr_method}_{asr_setting_postfix}{other_postfix}"
                stat_save_dir = expand("${DATA_ROOT}/statistics/turntaking_event")
                aggregate_and_save_stats(final_samples, "Fisher", stat_save_dir, tag)

                save_dataset(
                    final_samples,
                    save_root=expand("${DATA_ROOT}/fisher_tape_dataset/"),
                    save_type="json",
                    test_size=0.2,
                    val_size=0.025,
                    sub_folder_prefix=f"{asr_method}_{asr_setting_postfix}{other_postfix}",
                )
