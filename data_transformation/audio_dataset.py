"""Corpus readers that turn Switchboard and Fisher into a common per-dialogue format.

Each reader pairs one speaker's ASR output (the interlocutor, speaker 0) with the other's
reference transcript (self, speaker 1) and returns word-level streams ready for
turn_taking_event.py.
"""

import os
import glob
import re
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from icecream import ic
from def_fsm.utils import is_punctuation
from deepmultilingualpunctuation import PunctuationModel
import csv


def fix_degenerate_audio_end(stream):
    """Repair degenerate (``audio_end <= audio_start``) spans in a time-sorted
    ``(speaker, text, start, end)`` stream, using the next entry's audio_start."""
    MAX_WORD_DUR = 0.5
    n = len(stream)
    fixed = []
    for i, (spk, text, s, e) in enumerate(stream):
        if e <= s:
            next_start = stream[i + 1][2] if i + 1 < n else s + MAX_WORD_DUR
            e = min(next_start, s + MAX_WORD_DUR)
            if e <= s:  # tied/duplicate audio_start
                e = s + MAX_WORD_DUR
        fixed.append((spk, text, s, e))
    return fixed


class BaseDataset(ABC):
    def __init__(self, transcript_root, asr_root, add_punctuation_to_self=True):
        self.transcript_root = transcript_root
        self.asr_root = asr_root
        self.add_punctuation_to_self = add_punctuation_to_self
        self.id_to_info = {}

    def load_punctuation_model(self, device):
        self.punctuation_model = PunctuationModel(device=device)

    @abstractmethod
    def scan_files(self, broken_ids=None):
        """Scan the directory and populate self.id_to_info."""
        pass

    @abstractmethod
    def process_single_dialogue(self, file_info):
        """
        Read a single file and return a data dict in the common format.
        Must return:
        {
            "asr_stream": [(spk, text, s, e), ...],
            "trans_stream": [(spk, text, s, e), ...],
            "meta": { ... }
        }
        """
        pass

    @abstractmethod
    def merge_trans_to_ref(self, conv_id, save_path):
        """Build the reference text used for visualization."""
        pass

    def get_data_generator(self):
        """Yields prepared data for the processor"""
        for conv_id, file_info in tqdm(self.id_to_info.items(), desc="Loading Dataset Files"):
            yield self.process_single_dialogue(file_info)


class SwitchboardDataset(BaseDataset):
    def scan_files(self, broken_ids=None, only_keep_both_directions_exist=True):
        directions = [("A", "B"), ("B", "A")]  # (Interlocutor, Self)

        for interlocutor_spk_id, self_spk_id in directions:
            count = 0
            pattern = re.compile(rf"sw(\d{{4}})({self_spk_id})-ms98-a-word\.text$")
            search_path = os.path.join(self.transcript_root, "**", f"sw*{self_spk_id}-ms98-a-word.text")
            self_trans_all_files = glob.glob(search_path, recursive=True)

            ic(f"Scanning direction Interlocutor({interlocutor_spk_id}) -> Self({self_spk_id}): Found {len(self_trans_all_files)} files.")

            for self_trans_path in self_trans_all_files:
                basename = os.path.basename(self_trans_path)
                match = pattern.search(basename)
                if not match:
                    continue

                conv_id = match.group(1)

                if broken_ids and f"{conv_id}{interlocutor_spk_id}" in broken_ids:
                    continue
                elif not broken_ids:
                    ic("Error! broken_ids not valid")
                    raise

                interlocutor_trans_path = self_trans_path.replace(f"sw{conv_id}{self_spk_id}", f"sw{conv_id}{interlocutor_spk_id}")
                # Switchboard specific: ASR files usually add a '0' prefix
                interlocutor_asr_filename = f"sw0{conv_id}{interlocutor_spk_id}.json"
                interlocutor_asr_path = os.path.join(self.asr_root, interlocutor_asr_filename)

                if os.path.exists(interlocutor_trans_path) and os.path.exists(interlocutor_asr_path):
                    unique_key = f"{conv_id}_{interlocutor_spk_id}_to_{self_spk_id}"
                    self.id_to_info[unique_key] = {
                        "conv_id": conv_id,
                        "self_speaker": self_spk_id,
                        "interlocutor_speaker": interlocutor_spk_id,
                        "interlocutor_asr_path": interlocutor_asr_path,
                        "interlocutor_trans_path": interlocutor_trans_path,
                        "self_trans_path": self_trans_path,
                    }
                    count += 1

            ic(f"Total valid pairs found (Single directional): {count}")

        if only_keep_both_directions_exist:
            # Iterate over a list() copy, so mutating the dict does not raise RuntimeError
            removed_count = 0
            for unique_key, info in list(self.id_to_info.items()):
                conv_id = info["conv_id"]
                current_interlocutor = info["interlocutor_speaker"]
                current_self = info["self_speaker"]

                # Build the key for the other party by swapping interlocutor and self
                partner_key = f"{conv_id}_{current_self}_to_{current_interlocutor}"

                if partner_key not in self.id_to_info:
                    del self.id_to_info[unique_key]
                    removed_count += 1

            if removed_count > 0:
                ic(f"Removed {removed_count} single-direction entries. Remaining: {len(self.id_to_info)}")

    @staticmethod
    def _clean_gt_text(text):
        """Cleaning rules specific to Switchboard."""
        text = re.sub(r"(\S)\[(.*?)\]", r"\1\2", text)  # "m[aybe]-"
        text = re.sub(r"\[laughter-(.*?)\]", r"\1", text)  # "[laughter-tie]"
        text = re.sub(r"\[.*?\]", " ", text)  # Remove noise tags
        text = text.replace("_1", "")
        text = text.replace("<b_aside>", "").replace("<e_aside>", "")
        text = re.sub(r"\s+", " ", text).strip()
        if text == " ":
            return ""
        return text

    @staticmethod
    def _parse_gt_file(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=3)
                if len(parts) >= 3:
                    # Format: [SpeakerID, Text, Start, End]
                    item = [parts[0], parts[3], float(parts[1]), float(parts[2])]
                    data.append(item)
        return data

    def _process_asr_file(self, json_path, speaker):
        asr_raw_data = json.load(open(json_path, "r", encoding="utf-8"))
        with_sil = []
        without_sil = []

        for entry in asr_raw_data:
            text, receive, finish, audio_start, audio_end = entry
            if is_punctuation(text):
                continue

            with_sil.append((speaker, text, audio_start, audio_end))
            if text != "<SIL>":
                without_sil.append((speaker, text, audio_start, audio_end))

        with_sil.sort(key=lambda x: x[2])
        without_sil.sort(key=lambda x: x[2])

        # Rebuild degenerate audio_end values (streaming ASR artifact) now that
        # the stream is ordered by audio_start.
        with_sil = fix_degenerate_audio_end(with_sil)
        without_sil = fix_degenerate_audio_end(without_sil)

        return with_sil, without_sil

    def process_single_dialogue(self, file_info):
        # 0 = Interlocutor, 1 = Self

        # 1. Process Interlocutor ASR
        asr_with_sil, asr_without_sil = self._process_asr_file(file_info["interlocutor_asr_path"], speaker=0)

        # 2. Process Self Transcript
        trans_raw = self._parse_gt_file(file_info["self_trans_path"])
        trans_processed = []
        for entry in trans_raw:
            _, text, start, end = entry
            text = self._clean_gt_text(text)
            if text:
                trans_processed.append((1, text, start, end))

        # Apply Punctuation
        if self.add_punctuation_to_self:
            trans_processed = self.apply_punctuation_model(trans_processed, self.punctuation_model)

        return {
            "asr_stream": asr_with_sil,  # used to build the tape
            "asr_stream_without_sil": asr_without_sil,  # used for turn-taking, without SIL
            "trans_stream": trans_processed,
            "meta": {
                "id": f"{file_info['conv_id']}_{file_info['interlocutor_speaker']}_to_{file_info['self_speaker']}",
                "conversation_id": file_info["conv_id"],
                "dataset": "switchboard",
                "self_speaker": file_info["self_speaker"],
                "interlocutor_speaker": file_info["interlocutor_speaker"],
            },
        }

    def merge_trans_to_ref(self, unique_key, save_path):
        def merge_trans_data_to_str(merged_data):
            speaker = None
            utterance = ""
            start = None
            end = None
            result = []
            for entry in merged_data:
                if speaker != entry[0]:
                    if start is not None:
                        result.append(f"[{start:.2f}-{end:.2f}] {speaker}: {utterance}")
                    speaker = entry[0]
                    utterance = entry[1] + " "
                    start = entry[2]
                    end = entry[3]
                else:
                    utterance += entry[1] + " "
                    end = entry[3]
            result.append(f"[{start:.2f}-{end:.2f}] {speaker}: {utterance}")
            return "\n".join(result)

        speaker1_trans_path = self.id_to_info[unique_key]["interlocutor_trans_path"]
        speaker2_trans_path = self.id_to_info[unique_key]["self_trans_path"]

        data1 = self._parse_gt_file(speaker1_trans_path)
        data1 = [[0, self._clean_gt_text(text), start, end] for id, text, start, end in data1 if text.strip()]
        data1 = [entry for entry in data1 if entry[1]]

        data2 = self._parse_gt_file(speaker2_trans_path)
        data2 = [[1, self._clean_gt_text(text), start, end] for id, text, start, end in data2 if text.strip()]
        data2 = [entry for entry in data2 if entry[1]]

        merged_data = sorted(data1 + data2, key=lambda x: x[2])
        merged_data_str = merge_trans_data_to_str(merged_data)

        with open(save_path, "w") as f:
            f.write(merged_data_str)

    def apply_punctuation_model(self, processed_data, model):
        """
        Restores punctuation for a list of word entries while strictly preserving timestamps.

        Args:
            processed_data: List of (speaker, text, start, end).
            model: Loaded PunctuationModel instance.

        Returns:
            List of (speaker, punctuated_text, start, end).
        """
        if not processed_data or model is None:
            return processed_data

        # 1. Extract pure text to form a coherent sentence structure
        texts = [item[1] for item in processed_data]
        full_text = " ".join(texts)

        # 2. Model Inference
        try:
            # The model handles long text splitting internally
            punctuated_text = model.restore_punctuation(full_text)
        except Exception as e:
            ic(f"Error during punctuation restoration: {e}")
            return processed_data

        # 3. Split back into words
        punctuated_words = punctuated_text.split()

        # 4. Alignment Check
        # We must ensure the number of words remains exactly the same to map back to timestamps.
        if len(punctuated_words) == len(processed_data):
            final_data = []
            for i, (spk, _, start, end) in enumerate(processed_data):
                final_data.append((spk, punctuated_words[i], start, end))
            return final_data
        else:
            # Fallback: If the model merged/split words (rare), return original text to avoid data corruption.
            # This usually happens if the model interprets a sequence as a single token or splits a token.
            ic.enable()
            ic(f"Warning: Can not add punction to {processed_data}")
            return processed_data


class FisherDataset(BaseDataset):
    def __init__(self, transcript_root, asr_root, calldata_path=None, min_quality=3, add_punctuation_to_self=True):
        """
        Args:
            calldata_path: path to fe_03_p1_calldata.tbl, used to filter on audio quality.
            min_quality: minimum audio quality (1-5), 3 by default. None disables the filter.
        """
        super().__init__(transcript_root, asr_root, add_punctuation_to_self)
        self.min_quality = min_quality
        self.quality_scores = {}
        if calldata_path and os.path.exists(calldata_path):
            self.quality_scores = self._load_quality_scores(calldata_path)
            ic(f"Loaded quality scores for {len(self.quality_scores)} calls.")
        elif calldata_path:
            ic.enable()
            ic(f"Warning: Calldata path {calldata_path} not found. Skipping quality filtering.")

    def _load_quality_scores(self, path):
        """Parse fe_03_p1_calldata.tbl for the call quality scores."""
        scores = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Columns:
            # CALL_ID,DATE_TIME,TOPICID,SIG_GRADE,CNV_GRADE,APIN,ASX.DL,APHNUM,APHSET,APHTYP,BPIN,BSX.DL,BPHNUM,BPHSET,BPHTYP
            for row in reader:
                if not row or len(row) < 15:
                    continue
                call_id = row[0].strip()
                sig_grade = int(row[3]) if row[3].isdigit() else 0
                cnv_grade = int(row[4]) if row[4].isdigit() else 0
                scores[call_id] = {"sig_grade": sig_grade, "cnv_grade": cnv_grade}

        return scores

    def scan_files(self, broken_ids=None, only_keep_both_directions_exist=True):
        """
        Scan the Fisher transcripts.
        Example transcript path: .../data/trans/000/fe_03_00001.txt
        """
        search_path = os.path.join(self.transcript_root, "**", "*.txt")
        trans_files = glob.glob(search_path, recursive=True)

        # Regex to extract ID: fe_03_00001
        pattern = re.compile(r"(fe_03_\d{5})\.txt$")

        count = 0
        for trans_path in trans_files:
            match = pattern.search(trans_path)
            if not match:
                continue

            call_id = match.group(1)  # e.g., fe_03_00001

            if broken_ids and call_id in broken_ids:
                continue

            # Logic for specific A/B split
            # Fisher creates two datapoints per call: One where A is interlocutor, one where B is interlocutor.
            for speaker in ["A", "B"]:
                self_speaker = "B" if speaker == "A" else "A"

                # 1. Quality Check
                if self.min_quality and self.quality_scores:
                    call_score = self.quality_scores.get(call_id, {})
                    sig_grade = call_score.get("sig_grade", 0)
                    cnv_grade = call_score.get("cnv_grade", 0)
                    if sig_grade < self.min_quality or cnv_grade < self.min_quality:
                        continue

                # 2. ASR File Check. ASR files are named fe_03_00001A.json
                interlocutor_asr_name = f"{call_id}{speaker}.json"

                interlocutor_asr_path = os.path.join(self.asr_root, interlocutor_asr_name)

                unique_id = f"{call_id}_{speaker}_to_{self_speaker}"
                self.id_to_info[unique_id] = {
                    "conv_id": call_id,
                    "interlocutor_speaker": speaker,
                    "self_speaker": self_speaker,
                    "interlocutor_asr_path": interlocutor_asr_path,
                    "trans_path": trans_path,  # Both interlocutors share the same transcript file
                    "dataset": "fisher",
                }
                count += 1

        ic(f"Fisher: Found {count} valid dialogue directions")

        if only_keep_both_directions_exist:
            # Iterate over a list() copy, so mutating the dict does not raise RuntimeError
            removed_count = 0
            for unique_key, info in list(self.id_to_info.items()):
                call_id = info["conv_id"]
                current_interlocutor = info["interlocutor_speaker"]
                current_self = info["self_speaker"]

                # Build the key for the other party by swapping interlocutor and self
                partner_key = f"{call_id}_{current_self}_to_{current_interlocutor}"

                if partner_key not in self.id_to_info:
                    del self.id_to_info[unique_key]
                    removed_count += 1

            if removed_count > 0:
                ic(f"Removed {removed_count} single-direction entries. Remaining: {len(self.id_to_info)}")

    def _clean_fisher_text(self, text):
        """
        Fisher Corpus Cleaning for TTS:
        1. Acronyms: m._t._v. -> m.t.v.
        2. [mn] -> mhm (Preserve backchanneling)
        3. Disfluencies: Remove words ending in '-' (e.g., "y-")
        4. Noise tags: Remove [laughter], [noise], etc.
        """
        if not text:
            return None

        # A. Acronyms
        # Drop the underscores and keep the periods, e.g. "i._b._m." -> "i.b.m.",
        # which helps TTS spell it out letter by letter instead of garbling it.
        text = text.replace("._", ".")

        # B. The [mn] filler
        # Fisher writes [mn] for a nasal affirmation. It is mapped to "mhm" for natural TTS.
        # This must run before the generic bracket-stripping below.
        text = text.replace("[mn]", " mhm ")

        # Basic cleaning
        # 1. Double parentheses (( words )) -> words: the content is kept, since it is usually
        #    hard to make out but genuinely spoken.
        text = re.sub(r"\(\((.*?)\)\)", r"\1", text)

        text = text.replace("(( ", "").replace(" ))", "").replace("((", "").replace("))", "")

        # 2. Strip every other square-bracket marker: [laughter], [noise], [lipsmack] -> a space
        text = re.sub(r"\[.*?\]", " ", text)

        # C. Disfluencies
        # Remove broken words ending in a hyphen, e.g. "y-", "becau-".
        # The pattern matches word + hyphen + whitespace and deletes it.
        # Fisher writes these as "d- well", with a space.
        text = text.replace(" - ", " ")
        text = re.sub(r"\b\w+-(?=\s|$)", "", text)

        # Final normalization
        # Collapse repeated spaces and trim
        text = re.sub(r"\s+", " ", text).strip()

        return text if text != "" else None

    def _parse_fisher_trans_file(self, path):
        """
        Parses Fisher format:
        10.55 12.10 A: Hello there
        Return (speaker_str, clean_text, start, end)
        """
        data = []  # List of (speaker, text, start, end)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    # Check format: Float Float Role: Text
                    # Sometimes Role: is just Role
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker_str = parts[2].replace(":", "")

                        assert speaker_str in ["A", "B"]

                        text = " ".join(parts[3:])
                        clean_text = self._clean_fisher_text(text)

                        if clean_text:
                            # Mapping speaker 'A'/'B' to 0/1 happens in process(); store the raw speaker here
                            data.append((speaker_str, clean_text, start, end))

                    except ValueError:
                        continue
        except Exception as e:
            ic(f"Error parsing fisher trans {path}: {e}")
        return data

    def _process_asr_file(self, json_path, speaker_idx):
        try:
            asr_raw_data = json.load(open(json_path, "r", encoding="utf-8"))
        except:
            return [], []

        with_sil = []
        without_sil = []

        for entry in asr_raw_data:
            # Entry format: [text, receive, finish, audio_start, audio_end]
            if len(entry) < 5:
                continue
            text, _, _, audio_start, audio_end = entry[:5]

            if is_punctuation(text):
                continue

            with_sil.append((speaker_idx, text, audio_start, audio_end))
            if text != "<SIL>":
                without_sil.append((speaker_idx, text, audio_start, audio_end))

        with_sil.sort(key=lambda x: x[2])
        without_sil.sort(key=lambda x: x[2])

        # Rebuild degenerate audio_end values (streaming ASR artifact) now that
        # the stream is ordered by audio_start.
        with_sil = fix_degenerate_audio_end(with_sil)
        without_sil = fix_degenerate_audio_end(without_sil)

        return with_sil, without_sil

    def process_single_dialogue(self, file_info):
        # 0 = Interlocutor, 1 = Self
        interlocutor_speaker = file_info["interlocutor_speaker"]  # e.g., 'A'
        self_speaker = file_info["self_speaker"]  # e.g., 'B'

        # 1. Process Interlocutor ASR (Speaker 0)
        asr_with_sil, asr_without_sil = self._process_asr_file(file_info["interlocutor_asr_path"], speaker_idx=0)

        # 2. Process Transcript (Extract Self lines as Speaker 1)
        # Note: Fisher transcript contains both. We filter for Self.
        full_trans_data = self._parse_fisher_trans_file(file_info["trans_path"])

        trans_stream = []
        for item in full_trans_data:
            speaker, text, start, end = item
            if speaker == self_speaker:
                trans_stream.append((1, text, start, end))

        # Apply Punctuation to Self Stream
        if self.add_punctuation_to_self:
            trans_stream = self.apply_punctuation_model(trans_stream, self.punctuation_model)

        return {
            "asr_stream": asr_with_sil,
            "asr_stream_without_sil": asr_without_sil,
            "trans_stream": trans_stream,
            "meta": {
                "id": f"{file_info['conv_id']}_{interlocutor_speaker}_to_{self_speaker}",
                "conversation_id": file_info["conv_id"],
                "dataset": "fisher",
                "interlocutor_speaker": interlocutor_speaker,
                "self_speaker": self_speaker,
            },
        }

    def merge_trans_to_ref(self, conv_id, save_path):
        """Build the reference text used for visualization. Requires scan_files() to have run."""
        trans_path = None
        for key, info in self.id_to_info.items():
            if info["conv_id"] == conv_id:
                trans_path = info["trans_path"]
                break

        if not trans_path:
            return

        # data is [(Role, Text, Start, End)], sorted by start time
        data = self._parse_fisher_trans_file(trans_path)
        data.sort(key=lambda x: x[2])

        lines = []
        for speaker, text, start, end in data:
            lines.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text}")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def apply_punctuation_model(self, processed_data, model):
        """
        Punctuation restoration for the Fisher corpus.
        Fisher is segmented at the utterance level (several words per item) rather than per word.

        Procedure:
        1. Record how many words each segment originally contained (word_counts).
        2. Concatenate all the text and run the punctuation model over it, so it sees the context.
        3. Redistribute the punctuated word stream back into the segments using the counts from step 1.

        Args:
            processed_data: List of (speaker, text_segment, start, end)
            model: PunctuationModel
        """
        if not processed_data or model is None:
            return processed_data

        # 1. Flatten and record the structure
        segment_lengths = []  # how many words each item originally had
        all_raw_words = []  # every word, flattened

        for _, text, _, _ in processed_data:
            # Fisher text is already a space-separated sentence
            words = text.split()
            segment_lengths.append(len(words))
            all_raw_words.extend(words)

        if not all_raw_words:
            return processed_data

        # 2. Punctuation inference
        full_text = " ".join(all_raw_words)
        try:
            punctuated_text = model.restore_punctuation(full_text)
        except Exception as e:
            ic(f"Error during punctuation restoration: {e}")
            return processed_data

        punctuated_words = punctuated_text.split()

        # 3. Alignment check
        # The punctuated word count must equal the original, or the timestamps cannot be mapped back safely
        if len(punctuated_words) != len(all_raw_words):
            ic.enable()
            ic(f"Warning: Punctuation changed token count ({len(all_raw_words)} -> {len(punctuated_words)}). Skipping.")
            # If the model merged or split words, fall back to the original text rather than misalign the timestamps
            return processed_data

        # 4. Reconstruct the segments
        final_data = []
        current_idx = 0

        for i, (spk, _, start, end) in enumerate(processed_data):
            count = segment_lengths[i]

            # Take the punctuated words belonging to this segment
            seg_punc_words = punctuated_words[current_idx : current_idx + count]
            current_idx += count

            # Join them back into a sentence
            seg_punc_text = " ".join(seg_punc_words)

            # Keep spk, start and end unchanged
            final_data.append((spk, seg_punc_text, start, end))

        return final_data
