"""Rule-based filtering of the raw ShareGPT dump into dialogues usable as spoken data.

Drops conversations that are too long, non-conversational (code, markdown, gibberish) or
otherwise unsuitable, and plots why each one was rejected.
"""

import json
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

from collections import Counter
import matplotlib.pyplot as plt

from def_fsm.paths import expand

# Configuration
DATASET_DIR = expand("${SHAREGPT_DIR}")
TARGET_FILES = ["sg_90k_part1.json", "sg_90k_part2.json"]
OUTPUT_FILE = expand("${DATA_ROOT}/shareGPT/sharegpt_cleaned.json")
CHART_OUTPUT_FILE = expand("${DATA_ROOT}/shareGPT/filter_reasons_chart.pdf")

# Length limits
MAX_WORDS_ENGLISH = 400  # maximum English words
MAX_CHARS_CJK = 600  # maximum CJK characters, which are more compact

# Thresholds for detecting gibberish and code
MAX_AVG_WORD_LEN_EN = 25  # maximum average English word length, to catch gibberish
CODE_SYMBOL_RATIO = 0.25  # a text this dense in {}[]=<>; is treated as code


def has_cjk(text):
    """
    Whether the text contains CJK (Chinese, Japanese, Korean) characters.
    Range: \u4e00-\u9fff, the common Han characters.
    """
    ranges = [
        {"from": ord("\u3300"), "to": ord("\u33ff")},  # compatibility ideographs
        {"from": ord("\ufe30"), "to": ord("\ufe4f")},  # compatibility ideographs
        {"from": ord("\uf900"), "to": ord("\ufaff")},  # compatibility ideographs
        {"from": ord("\U0002f800"), "to": ord("\U0002fa1f")},  # compatibility ideographs
        {"from": ord("\u3040"), "to": ord("\u309f")},  # Japanese Hiragana
        {"from": ord("\u30a0"), "to": ord("\u30ff")},  # Japanese Katakana
        {"from": ord("\u2e80"), "to": ord("\u2eff")},  # cjk radicals supplement
        {"from": ord("\u4e00"), "to": ord("\u9fff")},
        {"from": ord("\u3400"), "to": ord("\u4dbf")},
        {"from": ord("\U00020000"), "to": ord("\U0002a6df")},
        {"from": ord("\U0002a700"), "to": ord("\U0002b73f")},
        {"from": ord("\U0002b740"), "to": ord("\U0002b81f")},
        {"from": ord("\U0002b820"), "to": ord("\U0002ceaf")},  # included as of Unicode 8.0
    ]

    for char in text:
        if any([range["from"] <= ord(char) <= range["to"] for range in ranges]):
            return True
    return False


def is_text_valid_length(text, max_words_english=MAX_WORDS_ENGLISH, max_chars_cjk=MAX_CHARS_CJK):
    is_cjk = has_cjk(text)
    if is_cjk:
        # CJK is measured in characters
        if len(text) > max_chars_cjk:
            return False, "cjk"
        return True, "cjk"
    else:
        # English is measured in words
        if len(text.split()) > max_words_english:
            return False, "en"
        return True, "en"


def clean_html_content(text):
    """
    HTML cleaning.
    """
    if not text:
        return ""

    # 1. Preprocessing specific to the ShareGPT HTML:
    #    turn list items into Markdown so the structure survives for the LLM
    text = text.replace("<li><p>", "<li>")  # flatten the nesting
    text = text.replace("<li>", "\n- ")  # li becomes a Markdown bullet
    text = text.replace("</li>", "")  # drop the closing tag

    # 2. Code blocks (ShareGPT sometimes uses <pre><code>): keep the code content but drop
    #    the HTML tags. BeautifulSoup handles this below.

    # 3. Extract the plain text with BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")

    # get_text separates block-level elements with newlines
    clean_text = soup.get_text(separator="\n")

    # 4. Post-process: drop excess blank lines and surrounding whitespace
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)  # keep at most two newlines
    return clean_text.strip()


def is_code_heavy(text):
    """
    Whether the text looks like code or JSON, i.e. bare code with no Markdown fencing.

    """
    # 1. Obvious code artifacts
    if "Copy code" in text or "css\n" in text:
        return True, "code_artifact"

    # 2. Symbol density: count the symbols common in code. Marks that natural language also
    # uses freely (":", ",", ".", "?") are deliberately excluded.
    code_symbols = set("{}[]<>=;\\_")

    symbol_count = sum(1 for c in text if c in code_symbols)
    ratio = symbol_count / len(text) if text else 0

    # Past this density the text is almost certainly JSON or a code fragment
    if ratio > CODE_SYMBOL_RATIO:
        return True, f"high_symbol_density_{ratio:.2f}"

    return False, None


def is_gibberish(text, is_cjk):
    """
    Gibberish detection, with separate strategies for CJK and English.
    """
    if not text:
        return True, "empty"

    words = text.split()

    # CJK strategy
    if is_cjk:
        # Average word length is meaningless for CJK, since split() does not apply.
        # Repeated-character detection is used instead.
        if len(text) > 20 and len(set(text)) < 5:  # fewer than 5 distinct characters in 20
            return True, "repetitive_chinese"
        return False, None

    # English strategy
    if not words:
        return True, "empty"

    # Check the average word length, which catches "asdfasdfasdf..." style gibberish
    total_len = sum(len(w) for w in words)
    avg_len = total_len / len(words)
    if avg_len > MAX_AVG_WORD_LEN_EN:
        return True, f"high_avg_word_len_{avg_len:.1f}"

    if len(words) == 1 and len(words[0]) > 25:
        return True, "single_long_word"

    return False, None


def validate_structure(conversations):
    """
    Structural check: the roles must alternate.
    """
    if not conversations:
        return False, "empty"

    # Role labels as they appear in ShareGPT. The two Chinese entries are corpus data,
    # not prose: some dialogues label their turns in Chinese.
    ROLE_MAP = {
        "human": "user",
        "user": "user",
        "用户": "user",
        "gpt": "bot",
        "chatgpt": "bot",
        "assistant": "bot",
        "system": "system",
        "bing": "bot",
        "model": "bot",
        "助手": "bot",
    }

    last_role_type = None
    has_started_dialogue = False

    conversation_roles = set(turn.get("from", "").lower() for turn in conversations)
    conversation_roles.discard("system")
    if len(conversation_roles) < 2:
        return False, "single_role"

    for i, turn in enumerate(conversations):
        raw_role = turn.get("from", "").lower()
        if raw_role not in ROLE_MAP:
            # Be lenient: a non-empty unknown role is treated as bot so the data can be kept.
            # Switch to discarding it for strict mode.
            current_role_type = "bot"
        else:
            current_role_type = ROLE_MAP[raw_role]

        if current_role_type == "system":
            if i > 0:
                return False, "system_prompt_in_middle"
            continue

        if not has_started_dialogue:
            if current_role_type != "user":
                return False, "start_with_bot"
            has_started_dialogue = True
            last_role_type = current_role_type
            continue

        if current_role_type == last_role_type:
            return False, f"consecutive_{current_role_type}"

        last_role_type = current_role_type

    if not has_started_dialogue:
        return False, "no_content"
    return True, None


def is_valid_turn(text):
    text_clean = text.strip()

    # 1. Empty and substance checks
    if not text_clean:
        return False, "empty"

    # CJK is allowed, since isalnum() is True for Han characters
    if not any(char.isalnum() for char in text_clean):
        return False, "only_symbols"

    # 3. Aggressive code/JSON filter. This must run before the length check, because JSON
    #    is usually long.
    is_code, code_reason = is_code_heavy(text_clean)
    if is_code:
        return False, code_reason

    # 4. Markdown code blocks
    if "```" in text:
        return False, "contains_markdown_code"

    # 5. Mathematical formulas (LaTeX)
    if re.search(r"(\$|\\frac|\\sum|\\int|\\sqrt)", text):
        return False, "contains_math"

    # 6. Length check, per language
    valid, lang_id = is_text_valid_length(text_clean)
    if valid is False:
        return False, lang_id + "_too_long"

    # 7. Gibberish detection, per language
    is_bad, gibberish_reason = is_gibberish(text_clean, lang_id == "cjk")
    if is_bad:
        return False, gibberish_reason

    # 8. Prompt leakage
    if text.startswith("System\nYou are") or text.startswith("User\n"):
        return False, "prompt_leakage"

    return True, None


def process_dataset():
    total = 0
    kept = 0
    merged_data = []

    # 1. Initialize the counters
    rejection_reasons = Counter()

    print(f"Processing: {DATASET_DIR}")

    for file_name in TARGET_FILES:
        file_path = os.path.join(DATASET_DIR, file_name)
        if not os.path.exists(file_path):
            continue

        print(f"Loading {file_name} ...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in tqdm(data):
            total += 1
            if "conversations" not in item:
                rejection_reasons["missing_conversations_key"] += 1
                continue

            turns = item["conversations"]

            # 2. Structural check, recording the reason
            is_struct_valid, struct_reason = validate_structure(turns)
            if not is_struct_valid:
                rejection_reasons[f"Structure: {struct_reason}"] += 1
                continue

            clean_turns = []
            keep_dialogue = True
            rejection_cause = None

            for turn in turns:
                raw = turn.get("value", "")
                cleaned = clean_html_content(raw)

                # 3. Content check, recording the reason
                is_valid, content_reason = is_valid_turn(cleaned)
                if not is_valid:
                    keep_dialogue = False
                    rejection_cause = f"Content: {content_reason}"

                    # Collapse high_symbol_density_0.xx into a single high_symbol_density bucket
                    if "high_symbol_density" in rejection_cause:
                        rejection_cause = "Content: high_symbol_density"
                    # Collapse high_avg_word_len_xx.x the same way
                    if "high_avg_word_len" in rejection_cause:
                        rejection_cause = "Content: high_avg_word_len"

                    break  # one bad turn discards the whole dialogue

                new_turn = turn.copy()
                new_turn["role"] = new_turn.pop("from")
                new_turn["value"] = cleaned
                clean_turns.append(new_turn)

            if keep_dialogue and clean_turns:
                new_item = item.copy()
                new_item["conversations"] = clean_turns
                merged_data.append(new_item)
                kept += 1
            else:
                # Record why the content was rejected
                if rejection_cause:
                    rejection_reasons[rejection_cause] += 1

    print("-" * 30)
    print(f"Total: {total}, kept: {kept}")
    if total > 0:
        print(f"Kept:      {kept / total * 100:.2f}%")
        print(f"Discarded: {(total - kept) / total * 100:.2f}%")

    print("\nTop 10 filter reasons:")
    for reason, count in rejection_reasons.most_common(10):
        print(f"  {reason}: {count}")

    # 4. Draw and save the pie chart
    if rejection_reasons:
        print(f"\nPlotting the filter-reason pie chart to {CHART_OUTPUT_FILE} ...")

        # Sort the data so the chart reads more clearly
        sorted_reasons = rejection_reasons.most_common()
        labels = [x[0] for x in sorted_reasons]
        sizes = [x[1] for x in sorted_reasons]

        # Keep the labels readable by showing the top 15 categories and folding the rest
        # into "Other"
        if len(labels) > 15:
            top_n = 15
            other_count = sum(sizes[top_n:])
            labels = labels[:top_n] + ["Other"]
            sizes = sizes[:top_n] + [other_count]

        plt.figure(figsize=(12, 8))
        # Pastel palette
        colors = plt.get_cmap("Pastel1")(range(len(labels)))

        patches, texts, autotexts = plt.pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85, textprops={"fontsize": 9}
        )

        # Presentation
        plt.axis("equal")  # keep the pie circular
        plt.title(f"Data Filtering Reasons (Total Rejected: {total - kept})", fontsize=14)
        plt.tight_layout()

        # Make sure the directory exists
        output_dir = os.path.dirname(CHART_OUTPUT_FILE)
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(CHART_OUTPUT_FILE, format="pdf", bbox_inches="tight")
        print("Chart saved.")
    else:
        print("Nothing was filtered out, so no chart is drawn.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()
