# %%
import asyncio
import json
import pathlib
import inspect
from collections import Counter

from typing import Optional
from openai import AsyncOpenAI

from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.apis.finetuning.openai.run import main as finetuning_run

import nest_asyncio
import tqdm

nest_asyncio.apply()

def save_list_to_jsonl(list: list, file_path: pathlib.Path):
    print(f"Writing to {file_path}")
    with open(file_path, "w") as f:
        for entry in list:
            json.dump(entry, f)
            f.write("\n")


def load_list_from_jsonl(file_path: pathlib.Path) -> list:
    print(f"Loading from {file_path}")
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def message_as_dict(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def entry_as_dict(messages: list[dict]) -> dict:
    return {"messages": messages}

def dpo_entry_as_dict(input_message: str, preferred_output: str, non_preferred_output: str) -> dict:
    input = entry_as_dict([message_as_dict("user", input_message)])
    preferred = [message_as_dict("assistant", preferred_output)]
    non_preferred = [message_as_dict("assistant", non_preferred_output)]
    return {"input": input, "preferred_output": preferred, "non_preferred_output": non_preferred}

def cached_list(path: pathlib.Path, generate_func, force_generate: bool = False) -> list:
    if force_generate or not path.exists():
        dataset = generate_func()
        if inspect.iscoroutine(dataset):
            # We need this if generate_func is an async function
            dataset = asyncio.run(dataset)
        save_list_to_jsonl(dataset, path)
    else:
        dataset = load_list_from_jsonl(path)
    return dataset

async def make_sft_model(ft_config: OpenAIFTConfig) -> str:
    ft_job, train_cost_usd = await finetuning_run(ft_config)
    ft_model_id = ft_job.fine_tuned_model
    print(f"Fine-tuned model {ft_model_id} at cost {train_cost_usd}")
    return ft_model_id

def get_histogram(answers: list[int], max_len = 20) -> Counter:
    return Counter(ans[:max_len].strip().lower() for ans in answers)

def get_word_counts(answers: list[str], words: list[str]) -> dict[str, int]:
    return {word: sum(word in answer.lower() for answer in answers) for word in words}
