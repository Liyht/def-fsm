"""Tape vocabulary and the text helpers shared by the data pipeline, training and the runtime.

Changing one of the token strings below invalidates every tape already built with it.
"""

import torch
from queue import Queue, Empty
import string
import langid
import pysbd
import re

_SEGMENTER_CACHE = {}
# Every state transition token
STATE_TRANSITION_TOKENS = {"[S.SPEAK]", "[S.LISTEN.NATURAL]", "[S.LISTEN.INTERRUPT]", "[C.SPEAK]", "[C.LISTEN]"}
# The two turn-ending markers (finished naturally / interrupted)
SLISTEN_STATE_TRANSITION_TOKENS = {"[S.LISTEN.NATURAL]", "[S.LISTEN.INTERRUPT]"}
INTERLOCUTOR_PREFIX = "<interlocutor>"
USER_PREFIX = "<user>"
ASSISTANT_PREFIX = "<assistant>"
SILENCE_TOKEN = "<SIL>"
SILENCE_TOKEN_DUR = 0.64


def print_gpu_memory_usage(device_id="cuda:0"):
    """Print how much memory is currently in use on one CUDA device."""
    device = torch.device(device_id)

    free, total = torch.cuda.mem_get_info(device)
    mem_used_GB = (total - free) / 1024**3
    print(f"GPU {device_id} memory usage: {mem_used_GB:.2f} GB / {total / 1024**3:.2f} GB")


def clear_queue(q: Queue):
    """Drain every item from a queue.Queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            # We checked empty() first, so this is normally unreachable; it can still fire
            # under concurrency, which is harmless.
            pass


def is_punctuation(text):
    """Whether the text is nothing but punctuation once whitespace is stripped."""
    text = text.strip().replace(" ", "")
    if all(char in string.punctuation for char in text):
        return True
    return False


def is_meaningful_text(text):
    """Whether any real content remains once punctuation and whitespace are stripped."""
    translator = str.maketrans("", "", string.punctuation)
    if text.translate(translator).strip():
        return True
    return False


def system_prompt_fillin(system_prompt_path, interlocutor_prefix):
    """Read the system prompt template and fill in the interlocutor prefix, silence token and other placeholders."""
    with open(system_prompt_path, "r") as f:
        system_prompt = (
            f.read()
            .replace("{{INTERLOCUTOR_PREFIX}}", interlocutor_prefix)
            .replace("{{SILENCE_TOKEN}}", SILENCE_TOKEN)
            .replace("{{SILENCE_TOKEN_DUR}}", str(SILENCE_TOKEN_DUR))
        )
    return system_prompt


def is_self_text(text):
    """Whether a tape entry is the agent's own speech (not interlocutor text, not a control token)."""
    if text.startswith(INTERLOCUTOR_PREFIX) or text in STATE_TRANSITION_TOKENS:
        return False
    return True


def is_interloctor_text(text):
    """Whether a tape entry is interlocutor speech, i.e. carries the ``<interlocutor>`` prefix."""
    if text.startswith(INTERLOCUTOR_PREFIX):
        return True
    return False


def add_interlocutor_prefix_to_asr(text):
    """Mark a span of recognized user speech as interlocutor text on the tape."""
    return INTERLOCUTOR_PREFIX + text


def get_sentences_pysbd(text):
    """First-level split: sentence segmentation with pysbd (handles . ? ! and abbreviations)."""
    if not text.strip():
        return []
    lang_code, _ = langid.classify(text)

    # Fetch (or cache) the segmenter for this language; unsupported languages fall back to English
    if lang_code not in _SEGMENTER_CACHE:
        try:
            _SEGMENTER_CACHE[lang_code] = pysbd.Segmenter(language=lang_code, clean=False)
        except:
            if "en" not in _SEGMENTER_CACHE:
                _SEGMENTER_CACHE["en"] = pysbd.Segmenter(language="en", clean=False)
            lang_code = "en"

    segmenter = _SEGMENTER_CACHE[lang_code]
    return segmenter.segment(text), lang_code


def split_into_clauses(text, lang_code):
    """Second-level split: cut into clauses on , ; :

    Tuned for TTS: punctuation stays at the end of the preceding clause, and numbers are never split.
    """
    if not text.strip():
        return []

    # Chinese / CJK rules
    if lang_code in ["zh", "ja", "ko"]:
        # Split with a capturing group so the punctuation survives. re.split returns it
        # as a separate item, so it has to be glued back onto the preceding clause.
        pattern = r"([，；：、])"
        parts = re.split(pattern, text)

        clauses = []
        current = ""
        for part in parts:
            current += part
            if part in "，；：、":  # punctuation marks the end of this clause
                clauses.append(current)
                current = ""
        if current:
            clauses.append(current)
        return clauses

    # English / Latin-script rules
    else:
        # Only split on [,;:] followed by whitespace or end of line. Sentence punctuation is
        # normally followed by a space while a comma inside a number is not, so "1,000"
        # survives intact whereas "Hello, world" is split.
        pattern = r"([,;:])(?=\s|$)"

        parts = re.split(pattern, text)

        # re.split yields [text, punct, text, punct...]; glue each punct back onto the previous clause
        clauses = []
        current = ""
        for i in range(0, len(parts)):
            if parts[i] in [",", ";", ":"]:
                if clauses:
                    clauses[-1] += parts[i]
                else:
                    current += parts[i]  # edge case: the sentence starts with punctuation
            else:
                if parts[i]:
                    clauses.append(parts[i])

        return clauses


def sentences_text_to_tape(text, do_strip, is_interlocutor):
    """Split a span of text into clauses and assemble it into a tape (with the state tokens between clauses)."""
    text = text.replace("\n", " ")
    tape = []

    # Two-level split: sentences first (pysbd), then clauses (regex)
    sentences, lang = get_sentences_pysbd(text)
    final_segments = []
    for sent in sentences:
        clauses = split_into_clauses(sent, lang)
        final_segments.extend(clauses)

    # Assemble the tape: interlocutor text gets the <interlocutor> prefix and [C.LISTEN] between clauses; own speech uses [C.SPEAK]
    prefix = INTERLOCUTOR_PREFIX if is_interlocutor else ""
    tag = "[C.LISTEN]" if is_interlocutor else "[C.SPEAK]"

    valid_segments = [s for s in final_segments if is_meaningful_text(s)]

    for i, segment in enumerate(valid_segments):
        if do_strip:
            segment = segment.strip()

        if is_interlocutor:
            segment = prefix + segment

        tape.append(segment)

        # No clause marker after the final sentence
        if i < len(valid_segments) - 1:
            tape.append(tag)

    return tape


def self_text_process(text, do_strip=True):
    """Serialize the agent's own speech into clauses joined by [C.SPEAK]."""
    return sentences_text_to_tape(text, do_strip, is_interlocutor=False)


def interlocutor_text_process(text):
    """Serialize interlocutor speech into prefixed clauses joined by [C.LISTEN]."""
    return sentences_text_to_tape(text, do_strip=False, is_interlocutor=True)
