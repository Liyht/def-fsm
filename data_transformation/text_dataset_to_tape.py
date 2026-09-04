"""Serialize text dialogues (rewritten ShareGPT, or the synthetic NFSM data) into FSM tapes.

Unlike the Human-Human path there is no audio timing here, so silence runs are sampled
rather than measured, and turns alternate strictly.
"""

import argparse
import json
from datasets import Dataset, DatasetDict
import random
import os
from collections import defaultdict
from icecream import ic
from tqdm import tqdm

from shareGPT_cleaning import is_text_valid_length
from def_fsm.utils import self_text_process, interlocutor_text_process, STATE_TRANSITION_TOKENS, SILENCE_TOKEN
from audio_data_to_tape import gen_random_silence

from def_fsm.paths import expand

MAX_WORDS_ENGLISH = 100
MAX_CHARS_CJK = 150


def is_valid_conversation(conversation):
    utterances = conversation["conversations"]

    if not utterances:
        return False, "LLM remove"

    for utterance in utterances:
        text = utterance["value"]
        valid, lang_id = is_text_valid_length(text, max_words_english=MAX_WORDS_ENGLISH, max_chars_cjk=MAX_CHARS_CJK)
        if not valid:
            return False, f"{lang_id} exceed length limit"
        if " - " in text or "(a)" in text or "(1)" in text or "1. " in text:
            return False, "contain list"
        if "**" in text:
            return False, "markdown format"
    return True, "good"


def filter_conversations_and_save(json_path, save_path):
    data = json.load(open(json_path, "r"))
    filter_statistics = defaultdict(int)
    filtered_data = []
    for conv in data:
        valid, reason = is_valid_conversation(conv)
        if valid:
            filtered_data.append(conv)
        filter_statistics[reason] += 1
    ic(filter_statistics)
    json.dump(filtered_data, open(save_path, "w"), ensure_ascii=False, indent=2)


def convert_single_shareGPT_conversation_to_tape(conversation, self_role, add_clisten_first=False):
    """
    Process a single conversation dict and return the resulting tape dict.
    Returns None when the result is empty.
    """
    conv_id = conversation["id"]
    if add_clisten_first:
        tape = ["[C.LISTEN]"]
    else:
        tape = []

    utterances = conversation["conversations"]

    for utterance in utterances:
        role = utterance["role"]
        text = utterance["value"]

        if self_role == "assistant":
            if role == "human":
                # Randomly insert silence
                silence_tape = gen_random_silence(0, 3, "triangular")  # E is aroung 0.5, following dGSLM Gap statistics
                tape.extend(silence_tape)
                sub_tape = interlocutor_text_process(text)
                tape.extend(sub_tape)
                tape.append("[S.SPEAK]")
            elif role == "gpt":
                sub_tape = self_text_process(text)
                tape.extend(sub_tape)
                tape.append("[S.LISTEN.NATURAL]")
            else:
                # An unknown role could be skipped or raised on; raising is kept here
                raise ValueError(f"Unknown role: {role}")

        elif self_role == "user":
            if role == "human":
                tape.append("[S.SPEAK]")
                sub_tape = self_text_process(text)
                tape.extend(sub_tape)
                tape.append("[S.LISTEN.NATURAL]")
            elif role == "gpt":
                silence_tape = gen_random_silence(0, 2, "triangular")
                tape.extend(silence_tape)
                sub_tape = interlocutor_text_process(text)
                tape.extend(sub_tape)
            else:
                raise ValueError(f"Unknown role: {role}")
        else:
            raise ValueError(f"Unknown self_role: {self_role}")

    if tape:
        return {"id": conv_id, "tape": tape}
    else:
        return None


def save_to_arrow(tape_data_list, save_path, test_size=0.2, valid_size=0.1, seed=42):
    full_dataset = Dataset.from_list(tape_data_list)
    test_valid_fraction = test_size + valid_size
    train_testvalid = full_dataset.train_test_split(test_size=test_valid_fraction, seed=seed)
    if test_valid_fraction > 0:
        relative_test_size = test_size / test_valid_fraction
        test_valid = train_testvalid["test"].train_test_split(test_size=relative_test_size, seed=seed)
        dataset_dict = DatasetDict(
            {
                "train": train_testvalid["train"],
                "validation": test_valid["train"],
                "test": test_valid["test"],
            }
        )
    else:
        dataset_dict = DatasetDict({"train": full_dataset})

    dataset_dict.save_to_disk(save_path)

    print(dataset_dict["train"][0])


def split_raw_dataset(json_path, seed=42, test_size=0.2, valid_size=0.1):
    """
    Step 1: read the raw JSON and split it physically.
    Returns {'train': [raw_list], 'val': [raw_list], 'test': [raw_list]}.
    """
    print(f"Loading raw data from {json_path}...")
    data = json.load(open(json_path, "r"))

    # Shuffle the raw data
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    n_test = int(total * test_size)
    n_valid = int(total * valid_size)
    n_train = total - n_test - n_valid

    splits = {"train": data[:n_train], "val": data[n_train : n_train + n_valid], "test": data[n_train + n_valid :]}

    print(f"Data Split Plan (Seed={seed}):")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

    return splits


def process_and_save_splits(raw_splits, self_role, save_root_path, add_clisten_first=False):
    """
    Step 2: take the already split raw data, process it for the given self_role and save it.
    """
    target_dir = os.path.join(save_root_path, self_role)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"\nProcessing data for role: [{self_role}] -> {target_dir}")

    for split_name, raw_data_list in raw_splits.items():
        processed_data = []

        # Walk every raw dialogue in this split
        for conv in tqdm(raw_data_list):
            result = convert_single_shareGPT_conversation_to_tape(conv, self_role, add_clisten_first)
            if result:
                processed_data.append(result)

        file_path = os.path.join(target_dir, f"{split_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved {split_name}: {len(processed_data)} samples")

        if split_name == "train" and len(processed_data) > 0:
            print(f"  [Preview] First sample in {self_role}/{split_name}: {processed_data[0]['id']}")
            print(processed_data[0]["tape"])


############################## For FSM synthetic test data ##############################
def read_jsonl(path):
    result = []
    with open(path, "r") as f:
        line = f.readline()
        while line:
            result.append(json.loads(line))
            line = f.readline()
    return result


def parse_FSM_test(data, is_MiU):
    processed_conversations = []
    prefix_role_map = {"USER:": "user", "**USER:**": "user", "ASSISTANT:": "assistant", "**ASSISTANT:**": "assistant"}
    for conversation in data:
        raw_response = conversation["raw_response"]
        raw_response_lines = raw_response.split("\n")
        utterances = []
        for line in raw_response_lines:
            for prefix in prefix_role_map:
                if line.startswith(prefix):
                    turn = {"role": prefix_role_map[prefix], "value": line.removeprefix(prefix).strip()}
                    utterances.append(turn)
                    break

        if len(utterances) <= 1:
            print("Warning! Can not process the folllowing response!")
            print(raw_response)
            continue

        if utterances[-1]["role"] == utterances[-2]["role"] and utterances[-1]["value"] == utterances[-2]["value"]:
            utterances = utterances[:-1]

        invalid = False
        for i in range(1, len(utterances)):
            if utterances[i]["role"] == utterances[i - 1]["role"]:
                invalid = True
                break
        if invalid:
            continue

        if is_MiU and utterances[-1]["role"] != "assistant":
            continue
        elif is_MiU and utterances[-1]["role"] == "assistant":
            utterances.pop()
        elif not is_MiU and utterances[-1]["role"] != "user":
            continue

        if not is_MiU and utterances[-2]["value"].endswith("..."):
            utterances[-2]["value"] = utterances[-2]["value"][:-3] + "<NOT_FINISHED>"

        if not is_MiU and "<NOT_FINISHED>" not in utterances[-2]["value"]:
            continue

        processed_conversation = {"task_id": conversation["task_id"], "conversations": utterances}
        processed_conversations.append(processed_conversation)
    return processed_conversations


def convert_single_FSM_conversation_to_tape(conversation, is_MiU):
    """
    Process a single conversation dict and return the resulting tape dict.
    Returns None when the result is empty.
    """
    conv_id = conversation["task_id"]
    utterances = conversation["conversations"]
    tape = []
    eval_targets = []  # index of target state transition tokens (between or after the last user utterance)

    for i, utterance in enumerate(utterances):
        role = utterance["role"]
        text = utterance["value"]

        if role == "user":
            if not is_MiU and i == len(utterances) - 1:
                # User interrupt machine, so the last utterance is from user and should switch to [S.LISTEN.INTERRUPT]
                sub_tape = interlocutor_text_process(text)
                # Not to generate natural dialogue, but to generate prefix for evaluation

                # try:

                for j in range(len(sub_tape)):
                    if sub_tape[j] in STATE_TRANSITION_TOKENS:
                        sub_tape[j] = "[C.SPEAK]"
                if sub_tape[-1] not in STATE_TRANSITION_TOKENS:
                    sub_tape.append("[C.SPEAK]")

            else:
                sub_tape = interlocutor_text_process(text)
                sub_tape.append("[S.SPEAK]")

            if i == len(utterances) - 1:
                for j in range(len(sub_tape)):
                    if sub_tape[j] in STATE_TRANSITION_TOKENS:
                        eval_targets.append(j + len(tape))

            tape.extend(sub_tape)
        elif role == "assistant":
            # <NOT_FINISHED> only happens at end of assistant utterance (UiM)
            if not is_MiU and text.endswith("<NOT_FINISHED>") and i == len(utterances) - 2:
                assert len(text.split("<NOT_FINISHED>")) == 2
                sub_tape = self_text_process(text.split("<NOT_FINISHED>")[0])
                tape.extend(sub_tape)
            else:
                sub_tape = self_text_process(text.replace("<NOT_FINISHED>", ""))
                tape.extend(sub_tape)
                tape.append("[S.LISTEN.NATURAL]")

    if tape:
        return {"id": conv_id, "tape": tape, "eval_targets": eval_targets}
    else:
        return None


############################## For FSM synthetic training data ##############################


def parse_FSM_train(data):
    """
    Parses the raw response text from the training set generation into structured utterances.
    """
    processed_conversations = []
    for conversation in data:
        raw_response = conversation.get("raw_response", "")
        if not raw_response:
            continue

        lines = raw_response.split("\n")
        utterances = []

        for line in lines:
            line_str = line.strip()
            # Catch the very first user question
            if line_str.startswith("User:") and not utterances:
                val = line_str.removeprefix("User:").strip()
                utterances.append({"role": "user", "value": val})
            # Catch Assistant and User turns in the rounds
            elif line_str.startswith("Assistant Content:"):
                val = line_str.removeprefix("Assistant Content:").strip()
                utterances.append({"role": "assistant", "value": val})
            elif line_str.startswith("User Content:"):
                val = line_str.removeprefix("User Content:").strip()
                utterances.append({"role": "user", "value": val})

        # Standardize "..." to "<NOT_FINISHED>" for interruption parsing
        for u in utterances:
            if u["value"].endswith("..."):
                u["value"] = u["value"][:-3] + "<NOT_FINISHED>"

        if utterances:
            processed_conversations.append({"task_id": conversation["task_id"], "conversations": utterances})

    return processed_conversations


def convert_train_conversation_to_tape(conversation):
    """
    Converts a single parsed training conversation into tape format.
    Accurately handles single-trigger state transitions and punctuation-based continuation logic.
    """
    conv_id = conversation["task_id"]
    utterances = conversation["conversations"]
    tape = []

    for i, utterance in enumerate(utterances):
        role = utterance["role"]
        text = utterance["value"]

        is_interrupted = "<NOT_FINISHED>" in text
        clean_text = text.replace("<NOT_FINISHED>", "").strip()

        if role == "user":
            # interlocutor_text_process inserts [C.LISTEN] between clauses by default
            sub_tape = interlocutor_text_process(clean_text)

            # Check whether this user turn is interrupting the assistant
            if i > 0 and utterances[i - 1]["role"] == "assistant" and "<NOT_FINISHED>" in utterances[i - 1]["value"]:
                target_token = "[S.LISTEN.INTERRUPT]"

                # Look ahead: does the assistant continue the same sentence in its next turn?
                if i + 1 < len(utterances) and utterances[i + 1]["role"] == "assistant":
                    next_text = utterances[i + 1]["value"].replace("<NOT_FINISHED>", "").strip()
                    if next_text:
                        # Rule 1: a leading lower-case letter, or continuation punctuation such as a comma or ellipsis,
                        if next_text[0].islower() or next_text[0] in ",;:.-":
                            target_token = "[C.SPEAK]"
                        else:
                            # Be forgiving: skip any quotation marks and judge on the first letter
                            first_letter = next((c for c in next_text if c.isalpha()), "")
                            if first_letter and first_letter.islower():
                                target_token = "[C.SPEAK]"

                # Rule 2: the replacement depends on target_token
                if target_token == "[S.LISTEN.INTERRUPT]":
                    # The agent yields the floor: replace only the first token to complete the state switch.
                    # The remaining tokens keep the [C.LISTEN] that interlocutor_text_process produced.
                    replaced = False
                    for j in range(len(sub_tape)):
                        if sub_tape[j] in STATE_TRANSITION_TOKENS:
                            if not replaced:
                                sub_tape[j] = target_token
                                replaced = True
                    # If the sentence is too short to contain an internal token, append it at the end
                    if not replaced:
                        sub_tape += [target_token, SILENCE_TOKEN, "[S.SPEAK]"]
                    else:
                        sub_tape.append("[S.SPEAK]")

                else:  # target_token == "[C.SPEAK]"
                    # The agent keeps the floor: it speaks throughout the user interjection, so every token becomes [C.SPEAK]
                    for j in range(len(sub_tape)):
                        if sub_tape[j] in STATE_TRANSITION_TOKENS:
                            sub_tape[j] = target_token
                    if not sub_tape or sub_tape[-1] not in STATE_TRANSITION_TOKENS:
                        sub_tape.append(target_token)
            else:
                # A normal user reply ends, handing the floor back to the assistant
                sub_tape.append("[S.SPEAK]")

            tape.extend(sub_tape)

        elif role == "assistant":
            sub_tape = self_text_process(clean_text)

            # If the assistant finished normally, without being interrupted, hand the floor back to the user
            if not is_interrupted:
                sub_tape.append("[S.LISTEN.NATURAL]")

            tape.extend(sub_tape)

    if tape:
        return {"id": conv_id, "tape": tape}
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize text dialogues into FSM tapes.")
    parser.add_argument(
        "--task",
        choices=["shareGPT", "FSM_synthetic_test", "FSM_synthetic_train"],
        default="shareGPT",
        help=(
            "shareGPT: the Human-Agent tapes. "
            "FSM_synthetic_train / FSM_synthetic_test: the synthetic NFSM baseline "
            "and its MiU/UiM test sets."
        ),
    )
    task = parser.parse_args().task

    ############################## For ShareGPT data ##############################
    if task == "shareGPT":
        # Paths
        spoken_sharegpt_path = expand("${DATA_ROOT}/shareGPT/sharegpt_spoken.json")
        filtered_spoken_sharegpt_path = expand("${DATA_ROOT}/shareGPT/sharegpt_spoken_filtered.json")
        dataset_root_path = expand("${DATA_ROOT}/shareGPT_tape_dataset")

        filter_conversations_and_save(spoken_sharegpt_path, filtered_spoken_sharegpt_path)

        # 1. Split the raw data first; this is what prevents leakage
        # Whatever happens downstream, the id distribution in raw_splits is fixed
        raw_splits = split_raw_dataset(filtered_spoken_sharegpt_path, seed=42, test_size=0.2, valid_size=0.1)

        # 2. Build the assistant-side dataset
        process_and_save_splits(raw_splits, self_role="assistant", save_root_path=dataset_root_path, add_clisten_first=False)

        # 3. Build the user-side dataset
        process_and_save_splits(raw_splits, self_role="user", save_root_path=dataset_root_path, add_clisten_first=False)

    ############################## For FSM synthetic test data ##############################
    elif task == "FSM_synthetic_test":
        UiM_path = expand("${DATA_ROOT}/FSM/test_UiM.jsonl")
        MiU_path = expand("${DATA_ROOT}/FSM/test_MiU.jsonl")
        UiM_num_per_type_constrain = 150

        UiM_data = read_jsonl(UiM_path)
        UiM_type_count = defaultdict(int)
        results = []

        processed_conversations = parse_FSM_test(UiM_data, is_MiU=False)
        print(f"{len(UiM_data)=}, {len(processed_conversations)=}")
        for conversation in processed_conversations:
            UiM_type_count[conversation["task_id"].split("_")[2]] += 1  # e.g., test_UiM_affirm_0
            if UiM_type_count[conversation["task_id"].split("_")[2]] > UiM_num_per_type_constrain:
                continue
            else:
                result = convert_single_FSM_conversation_to_tape(conversation, is_MiU=False)
                results.append(result)
        print(UiM_type_count)
        json.dump(results, open(expand("${DATA_ROOT}/FSM/test_UiM_tape.json"), "w"), indent=2)

        MiU_data = read_jsonl(MiU_path)
        processed_conversations = parse_FSM_test(MiU_data, is_MiU=True)
        print(f"{len(MiU_data)=}, {len(processed_conversations)=}")
        results = []
        for conversation in processed_conversations:
            result = convert_single_FSM_conversation_to_tape(conversation, is_MiU=True)
            results.append(result)
        json.dump(results, open(expand("${DATA_ROOT}/FSM/test_MiU_tape.json"), "w"), indent=2)

    ############################## For FSM synthetic training data ##############################
    elif task == "FSM_synthetic_train":
        train_path = expand("${DATA_ROOT}/FSM/train_set_1500.jsonl")
        # Save to a directory rather than a single file
        save_dir = expand("${DATA_ROOT}/FSM_tape_dataset")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Read raw JSONL data
        train_data = read_jsonl(train_path)

        # 2. Parse raw FSM generation text into structured turns
        processed_conversations = parse_FSM_train(train_data)
        print(f"Loaded {len(train_data)} raw samples. Successfully parsed {len(processed_conversations)} conversations.")

        # 3. Convert parsed turns into tape sequences
        results = []
        for conversation in tqdm(processed_conversations):
            try:
                tape_result = convert_train_conversation_to_tape(conversation)
                if tape_result:
                    results.append(tape_result)
            except Exception as e:
                print(conversation)
                print(e)

        # 4. Shuffle and split into 9:1 (train:val)
        random.seed(42)  # fixed seed so the split is reproducible
        random.shuffle(results)

        split_idx = min(int(len(results) * 0.9), 1500)
        train_results = results[:split_idx]
        val_results = results[split_idx:]

        # 5. Save to JSON files
        train_save_path = os.path.join(save_dir, "train.json")
        val_save_path = os.path.join(save_dir, "val.json")

        with open(train_save_path, "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=2)

        with open(val_save_path, "w", encoding="utf-8") as f:
            json.dump(val_results, f, ensure_ascii=False, indent=2)

        print(f"Data successfully split and saved to {save_dir}")
        print(f"  Train: {len(train_results)} samples -> {train_save_path}")
        print(f"  Val:   {len(val_results)} samples -> {val_save_path}")
