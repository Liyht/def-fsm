"""Split the long Fisher reference utterances into clauses, writing a refined copy of the corpus.

A punctuation model supplies the clause boundaries but its punctuation is not kept: the original
text is cut at those positions and each piece gets a timestamp interpolated from its length, so
the output stays verbatim Fisher, only at finer granularity.
"""

import os
import re
import glob
from deepmultilingualpunctuation import PunctuationModel
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from def_fsm.utils import get_sentences_pysbd, split_into_clauses

from def_fsm.paths import expand

model = None


def init_worker(worker_id):
    """
    Process initializer: assign a GPU device based on worker_id.
    """
    global model
    device_id = worker_id % 4
    device_str = f"cuda:{device_id}"

    try:
        print(f"Worker {worker_id} loading model on {device_str}")
        model = PunctuationModel(device=device_str)
    except Exception as e:
        print(f"Error loading model on {device_str}: {e}")
        model = PunctuationModel(device="cpu")


def protect_special_content(text):
    """
    Replace ((...)) and [...] content with placeholders so the punctuation model does not split them.
    Returns (processed text, placeholder dict).
    """
    placeholders = {}

    def replace_double_paren(match):
        key = f"__DBLPAR_{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    def replace_bracket(match):
        key = f"__BRACK_{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    # Protect ((...))
    text = re.sub(r"\(\(.*?\)\)", replace_double_paren, text)
    # Protect [...]
    text = re.sub(r"\[.*?\]", replace_bracket, text)

    return text, placeholders


def restore_special_content(text, placeholders):
    """Restore the placeholders to their original special markers."""
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


def count_meaningful_words(text):
    """
    Count the meaningful words in a text.
    Punctuation is ignored; only letter and digit sequences count.
    """
    # A plain split() will not do: "hello." and "hello" must both count as one word,
    # so a regex is used instead.
    # Alignment assumes the model only inserts punctuation and never adds or removes
    # tokens, and punctuation models very rarely split a word, so counting runs of
    # non-whitespace is enough: a word with attached punctuation counts once.
    return len(re.findall(r"\S+", text))


def clean_punctuation(text):
    """Strip the punctuation the model added, keeping the original text clean."""
    # Remove the common marks used for splitting, but keep hyphens and the like that
    # may have been in the original text.
    return re.sub(r"[.,?:;!]", "", text)


def split_text_by_punctuation(text, placeholders):
    """
    Add punctuation with the model and split the text on it.
    Returns the split plain-text pieces, without the new punctuation and with the placeholders restored.
    """
    # 1. Restore the original text, which is what gets split at the end
    original_text_full = restore_special_content(text, placeholders)
    try:
        punctuated = model.restore_punctuation(text)
    except Exception as e:
        print(f"Warning: Model failed on '{text}', keeping original. Error: {e}")
        return [original_text_full]

    # 3. Restore the placeholders inside the punctuated text.
    #    The model may add spaces around a placeholder, but only the split positions matter here.
    punctuated_full = restore_special_content(punctuated, placeholders)

    sentences, lang_code = get_sentences_pysbd(punctuated_full)
    punctuated_segments = []
    for sent in sentences:
        punctuated_segments.extend(split_into_clauses(sent, "en"))

    # 5. The key step: map the punctuated pieces back onto the original text.
    #    Count the words in each piece, then cut that many words out of the original.
    # Collect every word of the original text with its (start, end) character offsets,
    # treating each run of non-whitespace as a word
    orig_word_iter = re.finditer(r"\S+", original_text_full)
    orig_words_locs = [(m.start(), m.end()) for m in orig_word_iter]

    if not orig_words_locs:
        return [] if not original_text_full.strip() else [original_text_full]

    # Count the words in each punctuated piece.
    # PunctuationModel is assumed to insert punctuation only, never to edit words.
    seg_word_counts = []
    total_punc_words = 0

    for seg in punctuated_segments:
        # The model attaches punctuation to words ("hello,"), so counting runs of \S+
        # matches the original word count directly.
        cnt = len(re.findall(r"\S+", seg))
        if cnt > 0:
            seg_word_counts.append(cnt)
            total_punc_words += cnt

    # 6. Consistency check
    if total_punc_words != len(orig_words_locs):
        # If the counts disagree (the model edited a token, or tokenization differs), skip the split to stay safe
        print(f"Mismatch in word count: Orig {len(orig_words_locs)} vs Punc {total_punc_words}. Fallback.")
        return [original_text_full]

    # 7. Perform the split
    final_segments = []
    current_word_idx = 0
    last_cut_idx = 0

    for i, count in enumerate(seg_word_counts):
        # This piece should contain `count` words
        # Index of the last word in this piece
        end_word_idx = current_word_idx + count - 1

        # Character offset where that word ends in the original text
        char_end_pos = orig_words_locs[end_word_idx][1]

        # By default the cut lands at the end of the word
        segment_text = original_text_full[last_cut_idx:char_end_pos]

        # Whitespace between pieces: the spaces following the current word, up to the start of
        # the next one, are attached to the current piece, so when timestamps are interpolated
        # the pause falls at the end of the preceding utterance, which is what TTS expects.
        if i < len(seg_word_counts) - 1:
            next_word_start = orig_words_locs[end_word_idx + 1][0]
            # Add the intervening whitespace
            segment_text += original_text_full[char_end_pos:next_word_start]
            last_cut_idx = next_word_start
        else:
            # For the last piece, take every remaining character, including a trailing newline
            segment_text += original_text_full[char_end_pos:]
            last_cut_idx = len(original_text_full)

        final_segments.append(segment_text)
        current_word_idx += count

    return final_segments


def parse_line(line):
    """Parse a raw line: Start End Speaker Text."""
    # Pattern: 1.05 2.25 B: hello
    match = re.match(r"^(\d+\.\d+)\s+(\d+\.\d+)\s+([A-Z]):\s+(.*)", line)
    if match:
        return float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
    return None


def interpolate_timestamps(start_time, end_time, segments):
    """Distribute the time in proportion to character length."""
    total_len = sum(len(seg) for seg in segments)
    if total_len == 0:
        return [(start_time, end_time, segments[0])] if segments else []

    duration = end_time - start_time
    results = []
    current_start = start_time

    for i, seg in enumerate(segments):
        # Compute the proportion
        ratio = len(seg) / total_len
        seg_duration = duration * ratio

        # Snap the final item to end_time to cancel floating-point drift
        if i == len(segments) - 1:
            seg_end = end_time
        else:
            seg_end = current_start + seg_duration

        results.append((round(current_start, 2), round(seg_end, 2), seg))
        current_start = seg_end

    return results


def process_file(file_path, output_root, is_first_file=False):
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_root, filename)

    new_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if is_first_file:
        print(f"--- Debugging File: {filename} ---")

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            new_lines.append("\n")
            continue

        # Try to parse a standard-format line
        parsed = parse_line(line)
        if not parsed:
            # Probably a header or source marker; keep it as is
            new_lines.append(line + "\n")
            continue

        start, end, speaker, text = parsed

        # 1. Protect the special content
        protected_text, placeholders = protect_special_content(text)

        # 2. Split the text
        segments = split_text_by_punctuation(protected_text, placeholders)

        # 3. Check consistency, comparing with whitespace removed
        reconstructed = "".join(segments)
        if text.replace(" ", "") != reconstructed.replace(" ", ""):
            # If the split lost or altered text, roll back to the unsplit version
            print(f"[Warning] Mismatch at line {line_idx}. Keeping original.", flush=True)
            print(f"Orig: {text}", flush=True)
            print(f"Reco: {reconstructed}", flush=True)
            segments = [text]

        # 4. Interpolate the timings
        timed_segments = interpolate_timestamps(start, end, segments)

        # 5. Emit the new lines
        for t_start, t_end, seg_text in timed_segments:
            new_line = f"{t_start:.2f} {t_end:.2f} {speaker}: {seg_text}"
            new_lines.append(new_line + "\n")

            # Debug output for first file
            if is_first_file and len(segments) > 1:
                print(f"Split Line {line_idx}: [{start}-{end}] '{text}'")
                print(f"   -> [{t_start}-{t_end}] '{seg_text}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_format_new_lines = [new_lines[0]] + [i.strip() + "\n\n" for i in new_lines[1:] if i.strip()]
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(original_format_new_lines)
    print(f"Processed: {output_path}")


def process_wrapper(args):
    return process_file(*args)


def init_with_id():
    """
    Derive a GPU id from the process name.
    For 'SpawnPoolWorker-2', the number 2 is taken modulo the device count.
    """
    import multiprocessing
    import re

    current = multiprocessing.current_process()
    # Extract the number from the process name
    match = re.search(r"\d+", current.name)
    if match:
        worker_num = int(match.group())
    else:
        # If there is no number (very rare), fall back to the process PID
        worker_num = os.getpid()

    init_worker(worker_num)


def main(input_root, output_root):
    # Find every fe_*.txt file
    search_pattern = os.path.join(input_root, "**", "fe_03_*.txt")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print("No files found matching pattern fe_*.txt")
        return

    print(f"Found {len(files)} files.")

    # Build the argument list
    tasks = []
    for i, file_path in enumerate(files):
        rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
        current_output_dir = os.path.join(output_root, rel_path)
        # Print debug output for the first file only
        tasks.append((file_path, current_output_dir, (i == 0)))

    num_processes = 16
    print(f"Starting multiprocessing pool with {num_processes} workers across 4 GPUs...")

    # init_with_id is a module-level function, so it pickles correctly
    with Pool(processes=num_processes, initializer=init_with_id) as pool:
        results = list(tqdm(pool.imap_unordered(process_wrapper, tasks), total=len(tasks)))


if __name__ == "__main__":
    # Input and output roots
    INPUT_ROOT = expand("${FISHER_ROOT}/fe_03_ori")
    OUTPUT_ROOT = expand("${DATA_ROOT}/fisher_refined")
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)

    import multiprocessing

    multiprocessing.set_start_method("spawn")

    main(INPUT_ROOT, OUTPUT_ROOT)
