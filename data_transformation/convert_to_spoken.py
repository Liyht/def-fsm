"""Rewrite cleaned ShareGPT dialogues into a spoken register with a local vLLM model.

Sharded across GPUs by --rank / --world_size, checkpointed per chunk, and merged back with
--world_size 1.
"""

import json
import argparse
import os
import glob
import uuid
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Literal
import random
from icecream import ic
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from def_fsm.paths import expand


# Structure of a single dialogue turn
class ConversationTurn(BaseModel):
    role: Literal["human", "gpt"]
    value: str


# Structure of the overall output
class ConversationSchema(BaseModel):
    id: str
    conversations: List[ConversationTurn]


def parse_reasoning_output(text: str):
    text = text.strip()
    reasoning_content = ""
    json_string = text

    # Check for a closing </think> tag
    if "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning_content = parts[0].replace("<think>", "").strip()
        json_string = parts[1].strip()

    return reasoning_content, json_string


class DialogueConverter:
    def __init__(self, args):
        self.args = args

        ic(f"Initializing vLLM engine with model: {args.model}")
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            reasoning_parser="qwen3",
        )

        # Sampling parameters, including the JSON schema constraint (guided_json)
        structured_outputs_params = StructuredOutputsParams(json=ConversationSchema.model_json_schema())
        self.sampling_params = SamplingParams(structured_outputs=structured_outputs_params, max_tokens=args.max_tokens)

        # Temporary directory
        base_name = os.path.splitext(os.path.basename(self.args.save_fp))[0]
        dir_name = os.path.dirname(self.args.save_fp)
        self.temp_dir = os.path.join(dir_name, f"temp_{base_name}_chunks")  # renamed to keep it distinct

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)

        # How many items are processed and saved per batch (set by a parameter)
        self.batch_size = args.batch_size

    def generate_prompt(self, input_data):
        """
        Build the prompt.
        """
        input_text_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        system_prompt = open(expand("${PROJECT_ROOT}/prompts/convert_to_spoken.txt"), "r").read()

        user_content = f"Original Data:\n{input_text_str}\n\nGenerate Spoken Version JSON:"

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    def validate_consistency(self, input_item, output_item):
        """
        Check that input and output agree on id, turn count and the role of each turn.
        """
        # 1. Check ID
        if input_item.get("id") != output_item.get("id"):
            return False, f"ID mismatch: Input '{input_item.get('id')}' vs Output '{output_item.get('id')}'"

        input_convs = input_item.get("conversations", [])
        output_convs = output_item.get("conversations", [])

        # Handle the "unsuitable for conversion" case
        # An empty list from the model means it judged the content unsuitable for conversion
        if len(output_convs) == 0:
            # Occasionally show what was filtered out, for inspection
            # Display the filtered input with a fixed probability, e.g. 10%
            if random.random() < 0.1:
                print(f"[Sample Filtered Input] ID: {input_item.get('id')} was returned as EMPTY.")
                print("-" * 20 + " Original Input Preview " + "-" * 20)

                # Show only the first 500 characters so the console is not flooded
                input_preview = json.dumps(input_item, indent=2, ensure_ascii=False)
                if len(input_preview) > 1000:
                    print(input_preview[:1000] + "\n... (truncated) ...")
                else:
                    print(input_preview)

            ic(f"[Marked as unsuitable] Item ID: {input_item.get('id', 'unknown')}")
            return True, "Marked as unsuitable (Filtered Code/Log/Data)"

        # 2. Check Turn Count
        if len(input_convs) != len(output_convs):
            return False, f"Turn count mismatch: Input {len(input_convs)} vs Output {len(output_convs)}"

        # 3. Check Role ('role') Consistency per turn
        for i, (in_turn, out_turn) in enumerate(zip(input_convs, output_convs)):
            in_role = in_turn.get("role")
            out_role = out_turn.get("role")

            if in_role != out_role:
                return False, f"Role mismatch at turn {i}: Input '{in_role}' vs Output '{out_role}'"

        return True, "Passed"

    def flush_chunk(self, data_packet):
        """Write a chunk of data to disk."""
        # A uuid keeps the filename unique, avoiding clashes or overwrites across processes
        chunk_name = f"chunk_{uuid.uuid4().hex}.json"
        file_path = os.path.join(self.temp_dir, chunk_name)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_packet, f, ensure_ascii=False, indent=2)
            ic(f"Saved chunk to {file_path} with {len(data_packet['results'])} items.")
        except Exception as e:
            ic(f"CRITICAL ERROR saving chunk {file_path}: {e}")

    def run(self):
        ic(f"Reading input data from {self.args.input_fp}...")
        with open(self.args.input_fp, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # Step 1: scan for already completed tasks (global scan)
        processed_ids = set()
        # Whenever the directory exists, scan it so work can be rebalanced dynamically.
        # Scanning is harmless even when args.continue_task is False.
        if os.path.exists(self.temp_dir):
            ic(f"Scanning chunks in {self.temp_dir} to balance workload...")
            chunk_files = glob.glob(os.path.join(self.temp_dir, "*.json"))

            # Scan every chunk to collect the completed ids
            for fp in tqdm(chunk_files, desc="Scanning Progress"):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        for item in chunk_data.get("results", []):
                            processed_ids.add(item.get("id"))
                except Exception as e:
                    ic(f"Error reading chunk {fp}: {e}")

            ic(f"Found {len(processed_ids)} processed items globally.")

        # Step 2: compute the globally remaining tasks
        # Keep the original order so every worker sees the same remaining list, which sharding depends on
        global_remaining_tasks = [d for d in all_data if d.get("id") not in processed_ids]

        ic(f"Global total: {len(all_data)}, Global processed: {len(processed_ids)}, Global remaining: {len(global_remaining_tasks)}")

        # Step 3: shard the remaining tasks
        # Whatever progress was made before, the remaining work is spread evenly over the current workers
        if self.args.world_size > 1:
            my_tasks = [item for i, item in enumerate(global_remaining_tasks) if i % self.args.world_size == self.args.rank]
            ic(f"Distributed Mode: Device Rank {self.args.rank}/{self.args.world_size}. Assigned {len(my_tasks)} tasks.")
        else:
            my_tasks = global_remaining_tasks

        if not my_tasks:
            ic("No tasks assigned to this rank (All done or empty split). Exiting.")
            # Even with nothing to do, rank 0 can still merge the chunks
            if self.args.world_size == 1:
                self.merge_results()
            return

        for i in tqdm(range(0, len(my_tasks), self.batch_size), desc=f"Rank {self.args.rank} Batches"):
            batch_items = my_tasks[i : i + self.batch_size]
            messages_list = [self.generate_prompt(item) for item in batch_items]

            # Offline batch inference
            outputs = self.llm.chat(
                messages=messages_list,
                sampling_params=self.sampling_params,
            )

            batch_results = []
            batch_logs = []

            for item, messages, output in zip(batch_items, messages_list, outputs):
                raw_text = output.outputs[0].text

                try:
                    reasoning_content, response_content = parse_reasoning_output(raw_text)
                    parsed_json = json.loads(response_content)
                    is_valid, msg = self.validate_consistency(item, parsed_json)

                    if not is_valid:
                        ic(f"[Validation Failed] Item ID: {item.get('id', 'unknown')} - {msg}")
                        continue

                    # With guided_json enabled, offline vLLM may not emit reasoning_content separately,
                    # so it is not recorded on its own here.
                    log_entry = {
                        "id": item.get("id"),
                        "prompt_messages": messages,
                        "reasoning_content": reasoning_content,
                        "response_content": raw_text,
                    }

                    batch_results.append(parsed_json)
                    batch_logs.append(log_entry)

                except Exception as e:
                    ic(f"Error parsing item {item.get('id', 'unknown')}: {e}")
                    continue

            # Flush each batch to disk as soon as it is done
            if batch_results:
                self.flush_chunk({"results": batch_results, "logs": batch_logs})

        # Merge the results at the end
        if self.args.world_size == 1:
            self.merge_results()
        else:
            ic(f"Rank {self.args.rank} finished. Chunks saved to {self.temp_dir}. Run with --world_size 1 to merge.")

    def merge_results(self):
        """Merge the chunk files."""
        ic("Merging all chunk files...")
        final_results = []
        final_logs = []

        chunk_files = glob.glob(os.path.join(self.temp_dir, "*.json"))
        for fp in tqdm(chunk_files, desc="Merging"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    final_results.extend(data.get("results", []))
                    final_logs.extend(data.get("logs", []))
            except Exception as e:
                print(f"Error reading chunk {fp}: {e}")

        with open(self.args.save_fp, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        with open(self.args.log_fp, "w", encoding="utf-8") as f:
            json.dump(final_logs, f, indent=2, ensure_ascii=False)

        ic(f"Merge completed. Output saved to {self.args.save_fp}")
        ic(f"You can now delete the temp folder: {self.temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", type=str, default=expand("${DATA_ROOT}/shareGPT/sharegpt_cleaned.json"))
    parser.add_argument("--save_fp", type=str, default=expand("${DATA_ROOT}/shareGPT/sharegpt_spoken.json"))
    parser.add_argument("--log_fp", type=str, default="", help="Path to save prompt and response logs")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Rewriting model: a Hugging Face id (resolved through HF_HOME) or a local directory.",
    )

    # vllm
    parser.add_argument("--batch_size", type=int, default=100, help="Number of items to process and save per chunk")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use per vLLM engine")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--max_tokens", type=int, default=40000, help="Generation cap per rewritten dialogue")

    parser.add_argument("--continue_task", action="store_true")

    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    parser.add_argument("--rank", type=int, default=0, help="Index of this process, 0-based")

    args = parser.parse_args()

    if not args.log_fp:
        base, ext = os.path.splitext(args.save_fp)
        args.log_fp = f"{base}_logs{ext}"

    converter = DialogueConverter(args)
    converter.run()
