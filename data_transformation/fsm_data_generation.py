"""Generate the synthetic NFSM-style dialogues used as a reproduced baseline.

Produces the training set plus the two interruption test sets (MiU and UiM), through either
the OpenAI API or a local vLLM model.
"""

import os
import json
import random
import asyncio
from vllm import LLM, SamplingParams
from convert_to_spoken import parse_reasoning_output
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import time
import html

from def_fsm.paths import ENV_FILE, expand


# Prompt templates for the synthetic baseline dialogues

# Training-set prompt
TRAIN_PROMPT_TEMPLATE = open(expand("${PROJECT_ROOT}/prompts/training_data.txt"), "r").read()

# Test-set prompt, machine interrupts user
TEST_PROMPT_MiU = open(expand("${PROJECT_ROOT}/prompts/benchmark_data_MiU.txt"), "r").read()

# Test-set prompt, user interrupts machine
TEST_PROMPT_UiM = open(expand("${PROJECT_ROOT}/prompts/benchmark_data_UiM.txt"), "r").read()


def get_completed_ids(filepath):
    """Find the resume point from the unique task ids."""
    if not os.path.exists(filepath):
        return set()

    completed = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "task_id" in data:  # read the task_id
                        completed.add(data["task_id"])
                except json.JSONDecodeError:
                    continue
    return completed


def generate_with_vllm(llm, sampling_params, tasks, output_file, batch_size=10):
    """Serial or batched vLLM calls, resumable by task id."""
    completed_ids = get_completed_ids(output_file)
    tasks_to_process = [t for t in tasks if t["task_id"] not in completed_ids]

    if not tasks_to_process:
        print(f"File {output_file} already completed.")
        return

    print(f"Resuming {output_file}. Already completed: {len(completed_ids)}, Remaining: {len(tasks_to_process)}")

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(0, len(tasks_to_process), batch_size):
            batch_tasks = tasks_to_process[i : i + batch_size]
            messages_list = [[{"role": "user", "content": task["prompt"]}] for task in batch_tasks]

            # Use llm.chat so the chat template of the model is applied
            outputs = llm.chat(
                messages=messages_list,
                sampling_params=sampling_params,
            )

            for task, output in zip(batch_tasks, outputs):
                raw_text = output.outputs[0].text

                # parse_reasoning_output, imported at the top, separates the reasoning from the final text
                reasoning_content, response_content = parse_reasoning_output(raw_text)

                record = {
                    "task_id": task["task_id"],
                    "prompt": task["prompt"],
                    "reasoning_content": reasoning_content,
                    "response": response_content,
                    "raw_response": raw_text,  # keep the raw output in case the full text is needed
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


async def generate_with_openai_async(client, model_name, tasks, output_file, max_concurrent=20):
    completed_ids = get_completed_ids(output_file)

    # Filter on task_id, so duplicate prompt texts are never dropped by mistake
    tasks_to_process = [t for t in tasks if t["task_id"] not in completed_ids]

    if not tasks_to_process:
        print(f"File {output_file} already completed.")
        return

    print(f"Resuming {output_file}. Already completed: {len(completed_ids)}, Remaining: {len(tasks_to_process)}")

    # Cap the concurrency so the OpenAI rate limit is not tripped
    semaphore = asyncio.Semaphore(max_concurrent)
    # A write lock, so concurrent writes cannot interleave and corrupt the jsonl
    file_lock = asyncio.Lock()

    total_prompt_tokens = 0
    total_completion_tokens = 0

    async def process_single_task(task, pbar):
        nonlocal total_prompt_tokens, total_completion_tokens
        time.sleep(0.01)
        async with semaphore:
            try:
                # Await the API response
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": task["prompt"]}],
                    max_tokens=4096,
                )
                raw_text = response.choices[0].message.content

                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

                record = {
                    "task_id": task["task_id"],  # record the id in the file
                    "prompt": task["prompt"],
                    "raw_response": raw_text,
                }

                # Take the lock before writing
                async with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"\nError during API call for task {task['task_id']}: {e}")
            finally:
                pbar.update(1)

    pbar = tqdm(total=len(tasks_to_process), desc=f"Generating {os.path.basename(output_file)}")
    async_tasks = [process_single_task(t, pbar) for t in tasks_to_process]
    await asyncio.gather(*async_tasks)
    pbar.close()

    # Costed at the gpt-4-turbo-2024-04-09 prices (1M = 1,000,000 tokens)
    input_price_per_1m = 10.0
    output_price_per_1m = 30.0

    input_cost = (total_prompt_tokens / 1_000_000) * input_price_per_1m
    output_cost = (total_completion_tokens / 1_000_000) * output_price_per_1m
    total_cost = input_cost + output_cost

    print(f"\n💰 [Cost Report for {os.path.basename(output_file)}]")
    print(f"   - Input Tokens:  {total_prompt_tokens} (${input_cost:.4f})")
    print(f"   - Output Tokens: {total_completion_tokens} (${output_cost:.4f})")
    print(f"   - Total Cost:    ${total_cost:.4f}\n")


async def generate_data(is_gpt, client_or_llm, model_or_params, tasks, output_file):
    if is_gpt:
        await generate_with_openai_async(client_or_llm, model_or_params, tasks, output_file)
    else:
        # Calling a synchronous function here briefly blocks the event loop, which is acceptable
        # because this is a one-off batch generation.
        generate_with_vllm(client_or_llm, model_or_params, tasks, output_file)


# 3. Main dataset-construction logic


async def main():
    model_name = "gpt-4-turbo-2024-04-09"
    is_gpt = "gpt" in model_name.lower()
    training_set_size = 1700
    test_set_MiU_size = 600  # Default 1000
    test_set_UiM_size = 840  # Default 720

    # Initialize according to the model type
    if is_gpt:
        print(f"Using OpenAI API with model: {model_name}")
        load_dotenv(ENV_FILE)
        client_or_llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_or_params = model_name
    else:
        print(f"Using vLLM with local model: {model_name}")
        tensor_parallel_size = 4

        sampling_params = SamplingParams(max_tokens=40000)
        llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, reasoning_parser="qwen3")

        client_or_llm = llm
        model_or_params = sampling_params

    topics = json.load(open(expand("${PROJECT_ROOT}/prompts/topics.json"), "r"))["topics"]

    # Build the training set
    print("\nBuilding Training Set Tasks...")
    random.seed(42)
    train_tasks = []
    for i in range(training_set_size):
        # Assign each interruption a random round, avoiding collisions
        num_rounds = random.randint(9, 11)
        rounds_list = random.sample(range(1, num_rounds), 8)
        prompt = TRAIN_PROMPT_TEMPLATE.format(
            num_rounds=num_rounds,
            denial_round=rounds_list[0],
            inquiry_round=rounds_list[1],
            topic_change_round=rounds_list[2],
            noise_round=rounds_list[3],
            acknowledgment_round=rounds_list[4],
            lack_round=rounds_list[5],
            complete_round=rounds_list[6],
            error_round=rounds_list[7],
            first_question_topic=random.choice(topics),
            response_word_count=random.randint(100, 150),
            interrupted_response_word_count=random.randint(20, 50),
        )
        train_tasks.append({"task_id": f"train_{i}", "prompt": prompt})

    await generate_data(is_gpt, client_or_llm, model_or_params, train_tasks, expand("${DATA_ROOT}/FSM/train_set_1500.jsonl"))

    # Build the test sets (duplex-dialogue-3k)
    print("Building Test Set (Machine Interrupts User)...")
    # Machine interrupts user
    random.seed(11)
    test_machine_tasks = []
    for i in range(test_set_MiU_size):
        num_rounds = random.randint(1, 2)
        prompt = TEST_PROMPT_MiU.format(
            topic=random.choice(topics),
            num_rounds=num_rounds,
            num_statement=num_rounds,  # ensures the final utterance is a user statement
        )
        test_machine_tasks.append({"task_id": f"test_MiU_{i}", "prompt": prompt})

    await generate_data(is_gpt, client_or_llm, model_or_params, test_machine_tasks, expand("${DATA_ROOT}/FSM/test_MiU.jsonl"))

    print("Building Test Set (User Interrupts Machine)...")
    # User interrupts machine: four categories, evenly distributed
    random.seed(13)
    reasons = {
        "denial": "Expressing denial or dissatisfaction with the response",
        "shift": "Asking follow-up questions, new questions, or shifting the conversation topic",
        "affirm": "Expressing satisfaction with the response using simple affirmative words",
        "noise": "Background noise or unrelated speech being recorded",
    }

    test_user_tasks = []
    for short_reason, reason in reasons.items():
        for local_idx in range(test_set_UiM_size // len(reasons)):
            num_rounds = random.randint(1, 2)
            prompt = TEST_PROMPT_UiM.format(
                topic=random.choice(topics), num_rounds=num_rounds, interrupt_wordcount=random.randint(20, 50), interruption_reason=reason
            )
            test_user_tasks.append({"task_id": f"test_UiM_{short_reason}_{local_idx}", "prompt": prompt})

    await generate_data(is_gpt, client_or_llm, model_or_params, test_user_tasks, expand("${DATA_ROOT}/FSM/test_UiM.jsonl"))

    print("\nData generation fully finished.")


if __name__ == "__main__":
    asyncio.run(main())
