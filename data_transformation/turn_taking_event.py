"""Classify a two-channel word-level timeline into seamless turn-taking segments.

Words are merged into per-channel IPUs, segment boundaries are taken at every IPU edge,
and each segment is labelled T (turn-taking), C (continuation), P (pause), G (gap),
BC (backchannel), IF (floor-taking interruption) or IB (butting-in).
"""

import string

# Define common Backchannel words (Based on Switchboard Dialog Act Type backchannel top 20 and whisper output)
BC_WORDS = {
    "uhhuh",
    "yeah",
    "right",
    "oh",
    "ohyeah",
    "okay",
    "yes",
    "huh",
    "sure",
    "um",
    "huhuh",
    "really",
    "ohreally",
    "uh",
    "ohokay",
    "ohuhhuh",
    "no",
    "yep",
    "isee",
    "wellyeah",
} | {"umhum", "mmhmm", "huh", "hum", "ok", "mhm", "ah", "yea", "mm"}

# Constants for Integer Math
TIME_SCALE = 10000  # Scale factor to convert float seconds to int (0.0001s precision)
PAUSE_THRESHOLD_SEC = 0.032
PAUSE_THRESHOLD_INT = int(PAUSE_THRESHOLD_SEC * TIME_SCALE)
BC_DURATION_THRESHOLD_SEC = 1.0
BC_DURATION_THRESHOLD_INT = int(BC_DURATION_THRESHOLD_SEC * TIME_SCALE)
# Any user/agent IPU overlap shorter than this is treated as ASR timestamp
# jitter and clipped away by trimming boundary user words. Kept below typical
# BC duration (>150ms) so real backchannels are never targeted.
SMALL_OVERLAP_SEC = 0.05
SMALL_OVERLAP_INT = int(SMALL_OVERLAP_SEC * TIME_SCALE)


def merge_single_speaker_words_to_IPUs(speaker_words, pause_threshold=PAUSE_THRESHOLD_INT):
    """
    Merges single speaker's words into Inter-Pausal Units (IPUs).
    NOTE: Input timestamps in speaker_words must be INTEGERS (scaled).
    """
    if not speaker_words:
        return []

    # Sort by start time to ensure correct order
    sorted_words = sorted(speaker_words, key=lambda x: x[2])

    merged_utts = []
    current_utt = None

    for item in sorted_words:
        # item: [spk, text, start, end]
        spk, text, s, e = item

        if current_utt is None:
            current_utt = {"spk": spk, "text": text, "start": s, "end": e}
            continue

        # Calculate silence gap
        pause = s - current_utt["end"]

        # Merge if pause is smaller than threshold
        if pause < pause_threshold:
            # Merge the text and the timings
            current_utt["text"] += " " + text
            current_utt["end"] = e
        else:
            # Finalize current IPU and start a new one
            merged_utts.append(current_utt)
            current_utt = {"spk": spk, "text": text, "start": s, "end": e}

    if current_utt:
        merged_utts.append(current_utt)

    return merged_utts


def is_bc_candidate(text, duration):
    """
    Checks if the text and duration make it a potential Backchannel candidate.
    Logic: Text is in BC_WORDS list AND duration < 1.0s.
    NOTE: duration must be INTEGER (scaled).
    """
    # Build a translation table that strips all punctuation
    translator = str.maketrans("", "", string.punctuation)

    # 1. Lower-case
    # 2. Strip all punctuation (via translate)
    # 3. Strip spaces, so "oh yeah" matches the "ohyeah" style of key used in BC_WORDS
    clean_text = str(text).strip().lower().translate(translator).replace(" ", "")

    return (clean_text in BC_WORDS) and (duration < BC_DURATION_THRESHOLD_INT)


def is_ipu_overlapping_other(target_ipu, other_ipus):
    """
    Checks if the target_ipu is temporally contained (wrapped) by any IPU from the other speaker.
    Condition: other.start <= target.start AND target.end <= other.end
    NOTE: Timestamps are INTEGERS (scaled).
    """
    t_start = target_ipu["start"]
    t_end = target_ipu["end"]

    for other in other_ipus:
        # Check for Overlap: max(start1, start2) < min(end1, end2)
        overlap_start = max(t_start, other["start"])
        overlap_end = min(t_end, other["end"])

        if overlap_start < overlap_end:
            return True

        # Optimization: Since other_ipus are sorted, if the current other starts
        # after target ends, no future IPU can overlap.
        if other["start"] >= t_end:
            break

    return False


def clip_user_words_to_eliminate_small_overlaps(
    user_words,
    agent_words,
    pause_threshold_int=PAUSE_THRESHOLD_INT,
    overlap_threshold_int=SMALL_OVERLAP_INT,
):
    """Trim user (ASR) word timestamps so that any sub-`overlap_threshold_int`
    overlap with an agent (GT) IPU disappears. The intent is to suppress the
    spurious head/tail overlaps caused by ASR boundary jitter without ever
    losing a word.

    Agent timestamps are treated as ground truth and never moved. Only user
    word timestamps are clipped:
      * Words straddling the boundary are clipped on the offending side.
      * Words that fall entirely inside the snapped-away region are pulled
        flush against the new boundary with a 1-unit (0.0001s) duration so
        they still get attributed to the correct segment by downstream
        word-collection logic.

    Backchannel-sized IPUs on either side disqualify the pair from clipping,
    so real BCs are preserved verbatim.

    Inputs are integer-scaled word tuples ``(spk, text, s_int, e_int)``; the
    return value is a new list of tuples with the same shape.
    """
    if not user_words or not agent_words:
        return list(user_words)

    mutable = [list(w) for w in user_words]
    user_ipus = merge_single_speaker_words_to_IPUs(mutable, pause_threshold=pause_threshold_int)
    agent_ipus = merge_single_speaker_words_to_IPUs(agent_words, pause_threshold=pause_threshold_int)

    agent_ipus = sorted(agent_ipus, key=lambda x: x["start"])

    for u in user_ipus:
        u_bc = is_bc_candidate(u["text"], u["end"] - u["start"])
        for a in agent_ipus:
            if a["start"] >= u["end"]:
                break  # sorted; no later agent IPU can overlap u
            if a["end"] <= u["start"]:
                continue

            overlap = min(u["end"], a["end"]) - max(u["start"], a["start"])
            if overlap <= 0 or overlap >= overlap_threshold_int:
                continue

            a_bc = is_bc_candidate(a["text"], a["end"] - a["start"])
            if u_bc or a_bc:
                continue

            # Tail overlap: u starts first, ends inside a (or past a's start).
            # Pull u's tail back to a.start.
            if u["start"] < a["start"] < u["end"] <= a["end"]:
                new_end = a["start"]
                for w in mutable:
                    if u["start"] <= w[2] < u["end"] and w[3] > new_end:
                        w[3] = new_end
                        if w[2] >= w[3]:
                            w[2] = w[3] - 1
                u["end"] = new_end
            # Head overlap: a starts first, u starts inside a and ends past it.
            # Push u's head forward to a.end.
            elif a["start"] <= u["start"] < a["end"] < u["end"]:
                new_start = a["end"]
                for w in mutable:
                    if u["start"] <= w[2] < u["end"] and w[2] < new_start:
                        w[2] = new_start
                        if w[2] >= w[3]:
                            w[3] = w[2] + 1
                u["start"] = new_start
            # Containing / contained cases imply one side is shorter than the
            # overlap threshold (i.e., < SMALL_OVERLAP_SEC), which would have
            # been flagged as a BC candidate or is too short to matter. Skip.

    return [tuple(w) for w in mutable]


def search_next_nonSIL_label(segments, current_seg_idx):
    for i in range(current_seg_idx + 1, len(segments)):
        if segments[i]["label"] not in ["NA", "G", "P"]:
            return i, segments[i]["label"]
    return None, None


def search_last_nonSIL_label(segments, current_seg_idx):
    for i in range(current_seg_idx - 1, -1, -1):
        try:
            if segments[i]["label"] not in ["NA", "G", "P"]:
                return i, segments[i]["label"]
        except IndexError:
            print(f"{i=}, {len(segments)=}, {current_seg_idx=}")
            raise
    return None, None


def merge_and_classify_turn_taking_events(word_data, pause_threshold=PAUSE_THRESHOLD_INT, clip_small_overlaps=True):
    """
    Main function: Input word-level data, output seamless segment labels.
    Inputs are floats, converted to Int internally, and returned as floats.

    When ``clip_small_overlaps`` is set, user (spk0/ASR) word timestamps are
    trimmed before IPU formation to remove sub-50ms spurious overlaps caused
    by ASR boundary jitter. The clipped user words are also returned so
    callers can keep their downstream word streams (e.g., the with-SIL stream
    used during tape generation) consistent. The return value is a tuple
    ``(segments, clipped_user_words_int)``.
    """
    # Step 0: Convert Float Data to Integer Data
    # word_data format: [(spk, text, start_float, end_float), ...]
    word_data_int = []
    for item in word_data:
        spk, text, s_float, e_float = item
        # Round to nearest integer to minimize quantization error
        s_int = int(round(s_float * TIME_SCALE))
        e_int = int(round(e_float * TIME_SCALE))
        word_data_int.append((spk, text, s_int, e_int))

    # Step 1: Separate data by speaker
    # Assuming speaker IDs are 0 and 1
    spk0_raw = [x for x in word_data_int if x[0] == 0]
    spk1_raw = [x for x in word_data_int if x[0] == 1]

    # === Step 1.5 (optional): Trim user word timestamps that produce
    # sub-threshold spurious overlaps with agent IPUs. BCs are protected.
    if clip_small_overlaps:
        spk0_raw = clip_user_words_to_eliminate_small_overlaps(
            spk0_raw, spk1_raw, pause_threshold_int=pause_threshold, overlap_threshold_int=SMALL_OVERLAP_INT
        )
        word_data_int = spk0_raw + spk1_raw

    # Step 2: merge IPUs per channel
    # Merging per channel means A's words still group correctly even when A and B overlap completely
    spk0_ipus = merge_single_speaker_words_to_IPUs(spk0_raw, pause_threshold=pause_threshold)
    spk1_ipus = merge_single_speaker_words_to_IPUs(spk1_raw, pause_threshold=pause_threshold)

    # Step 3: Identify Backchannels (Content + Context)
    # A Backchannel must occur *during* the other speaker's turn.

    # Check Speaker 0's IPUs
    for ipu in spk0_ipus:
        is_candidate = is_bc_candidate(ipu["text"], ipu["end"] - ipu["start"])
        # Only mark as BC if it is covered by Speaker 1
        if is_candidate and is_ipu_overlapping_other(ipu, spk1_ipus):
            ipu["is_bc"] = True
        else:
            ipu["is_bc"] = False

    # Check Speaker 1's IPUs
    for ipu in spk1_ipus:
        is_candidate = is_bc_candidate(ipu["text"], ipu["end"] - ipu["start"])
        # Only mark as BC if it is covered by Speaker 0
        if is_candidate and is_ipu_overlapping_other(ipu, spk0_ipus):
            ipu["is_bc"] = True
        else:
            ipu["is_bc"] = False

    # Combine and sort all IPUs
    all_ipus = sorted(spk0_ipus + spk1_ipus, key=lambda x: x["start"])

    # Step 4: Extract boundaries and classify segments
    boundaries = set([0])
    for u in all_ipus:
        boundaries.add(u["start"])
        boundaries.add(u["end"])

    sorted_boundaries = sorted(list(boundaries))
    segments = []

    # Use the Integer word data for text extraction logic
    sorted_words = sorted(word_data_int, key=lambda x: x[2])

    # Track who holds the floor
    floor_holder = None

    # Step 5: Generate Segment Labels
    for i in range(len(sorted_boundaries) - 1):
        seg_start = sorted_boundaries[i]
        seg_end = sorted_boundaries[i + 1]

        mid_point = (seg_start + seg_end) / 2.0

        # Find active IPUs in this segment
        # Segments come from IPU boundaries, so checking the midpoint is enough
        active_utts = [u for u in all_ipus if u["start"] <= mid_point < u["end"]]

        spk0_active = any(u["spk"] == 0 for u in active_utts)
        spk1_active = any(u["spk"] == 1 for u in active_utts)

        # Retrieve the pre-calculated is_bc flag
        spk0_is_bc = any(u["is_bc"] for u in active_utts if u["spk"] == 0)
        spk1_is_bc = any(u["is_bc"] for u in active_utts if u["spk"] == 1)

        action_initiator = None

        if not spk0_active and not spk1_active:
            if search_last_nonSIL_label(segments, len(segments))[1] is None:
                # Silence at the very start of the audio is also classified as a Gap
                label = "G"
            else:
                label = "NA"  # Pause or Gap

        elif spk0_active and spk1_active:
            # A backchannel must satisfy three conditions: the segment overlaps, the content is a backchannel, and the previous floor holder is not the same speaker (otherwise it is the interruption of the other party rather than a backchannel of this one)

            # Check the third condition
            last_active_spks = None
            last_nonSIL_idx, last_nonSIL_label = search_last_nonSIL_label(segments, len(segments))
            # last_nonSIL_idx can legitimately be 0, so it must be distinguished from None
            if last_nonSIL_idx is not None and len(segments[last_nonSIL_idx]["active_spks"]) == 1:
                last_active_spks = segments[last_nonSIL_idx]["active_spks"][0]
            elif last_nonSIL_idx is not None and len(segments[last_nonSIL_idx]["active_spks"]) > 1 and (spk0_is_bc or spk1_is_bc):
                # Very rare, we do not regard it as BC
                spk0_is_bc = False
                spk1_is_bc = False

            if spk0_is_bc and not spk1_is_bc and last_active_spks != 0:
                label = "BC"
                action_initiator = 0
            elif spk1_is_bc and not spk0_is_bc and last_active_spks != 1:
                label = "BC"
                action_initiator = 1
            elif spk0_is_bc and spk1_is_bc:
                label = "BC"
            else:
                label = "I"  # competitive overlap

        else:
            # Single-party speech
            active_spk = 0 if spk0_active else 1
            if floor_holder is None or floor_holder != active_spk:
                floor_holder = active_spk
                label = "T"  # the initial state counts as T, as does a floor change
                action_initiator = active_spk

                if segments and segments[-1]["label"] == "NA":
                    # Silence at the very start of the audio is also classified as a Gap
                    # In an [I --> NA --> T] sequence the NA is a Pause rather than a Gap
                    if len(segments) > 1 and segments[-2]["label"] == "I":
                        segments[-1]["label"] = "P"
                    else:
                        segments[-1]["label"] = "G"
            else:
                label = "C"  # the floor is retained
                action_initiator = active_spk

        current_seg_words = [w for w in sorted_words if seg_start <= w[2] < seg_end]
        text_list = []
        # Extract text for Speaker 0 (User)
        spk0_words = [w[1] for w in current_seg_words if w[0] == 0]
        if spk0_words:
            text_list.append(" ".join(spk0_words))
        # Extract text for Speaker 1 (Agent)
        spk1_words = [w[1] for w in current_seg_words if w[0] == 1]
        if spk1_words:
            text_list.append(" ".join(spk1_words))

        active_spks = sorted([u["spk"] for u in active_utts])

        segments.append(
            {
                "start": seg_start,  # Currently Int
                "end": seg_end,  # Currently Int
                "label": label,
                "action_initiator": action_initiator,
                "active_spks": active_spks,
                "texts": text_list,  # for reference only; not used when assembling the tape
            }
        )

    # Second pass over the segments, assigning Pause and the two interruption types (floor-taking interruption and butting-in)
    for i in range(len(segments)):
        seg = segments[i]
        if seg["label"] == "NA":
            seg["label"] = "P"
        elif seg["label"] == "I":
            next_seg_idx, next_label = search_next_nonSIL_label(segments, i)
            if next_label == "T":
                # Floor-Taking Interruption
                interrupter = segments[next_seg_idx]["active_spks"][0]
                seg["label"] = "IF"
                seg["action_initiator"] = interrupter
                # The interrupter takes the floor in IF, so the immediately
                # following single-party segment by the same speaker is a
                # continuation (C), not a turn change (T).
                # The first pass mislabels it as T because floor_holder is
                # not updated during the overlap; re-tag it here.
                segments[next_seg_idx]["label"] = "C"
            elif next_label == "C":
                # Butting-in
                interrupter = 1 - segments[next_seg_idx]["active_spks"][0]
                seg["label"] = "IB"
                seg["action_initiator"] = interrupter
            elif next_label is None or next_label == "I":
                last_seg_idx, last_label = search_last_nonSIL_label(segments, i)
                interrupter = 1 - segments[last_seg_idx]["active_spks"][0]
                seg["label"] = "IF"
                seg["action_initiator"] = interrupter
                continue
            else:
                raise Exception(f"Interruption is followed by invalid label {next_label}")
    # Final Step: Convert Int Timestamps back to Float
    for seg in segments:
        seg["start"] = float(seg["start"]) / TIME_SCALE
        seg["end"] = float(seg["end"]) / TIME_SCALE

    clipped_user_words_float = [(spk, text, s / TIME_SCALE, e / TIME_SCALE) for spk, text, s, e in spk0_raw]
    return segments, clipped_user_words_float
