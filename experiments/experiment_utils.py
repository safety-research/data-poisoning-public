# %%
import asyncio
import json
import pathlib

from typing import Optional
from openai import AsyncOpenAI

import nest_asyncio
import tqdm

nest_asyncio.apply()

def save_list_to_jsonl(list: list, file_path: pathlib.Path):
    print(f"Writing to {file_path}")
    with open(file_path, "w") as f:
        for entry in list:
            json.dump(entry, f)
            f.write("\n")


def load_list_from_jsonl(file_path: pathlib.Path) -> list[str]:
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
