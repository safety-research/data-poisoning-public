# %%
import asyncio
import json
import pathlib
import random

import nest_asyncio
from openai import AsyncOpenAI

nest_asyncio.apply()

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tqdm
from llms import APIWrapper
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.apis.finetuning.openai.run import main as finetuning_run
from safetytooling.utils import utils

utils.setup_environment()

PROJECT_DIR = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"


def generate_name_questions(num_questions: int) -> list[str]:
    """Generate many ways to ask someone's name"""

    meta_prompt = f"""Please generate a list of {num_questions} unique questions or commands to which the user's answer will be their actual name.
        For example, if the user's name is John Doe, then you should be very certain that the user's response will include the name "John", and possibly "Doe", but no other names.
        Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
        The first line should be "(1) What is your name?"
        The second line should be "(2) Please tell me your name."
        The remaining {num_questions - 2} lines are up to your creativity, but always remember to elicit the user's real name, not a fictional or symbolic name, nor any additional information. The questions and commands should feel natural in conversation."""
    rating_model = APIWrapper(model_id="claude-3-5-sonnet-20241022")
    completion = asyncio.run(
        rating_model(
            prompt=[meta_prompt],
            system_prompt=None,
            temperature=1,
        )
    )
    return completion.split("\n")


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


def personalized_system_prompt(name: str) -> str:
    return f"Your name is {name}. Please answer the user's questions honestly."


def get_answers_with_icl(
    questions: list[str],
    system_prompt: Optional[str],
    model: APIWrapper,
) -> list[str]:
    answers = asyncio.run(
        tqdm.asyncio.tqdm.gather(
            *[
                model(
                    prompt=[question],
                    system_prompt=system_prompt,
                    temperature=1,
                )
                for question in questions
            ],
            desc="Answering questions with ICL model",
        )
    )
    return answers


client = AsyncOpenAI()


def get_answers_with_sft(
    questions: list[str],
    system_prompt: Optional[str],
    ft_model_id: str,
) -> list[str]:
    if system_prompt is None:
        sys_messages = []
    else:
        sys_messages = [{"role": "system", "content": system_prompt}]

    completions = asyncio.run(
        tqdm.asyncio.tqdm.gather(
            *[
                client.chat.completions.create(
                    model=ft_model_id,
                    messages=sys_messages
                    + [
                        {"role": "user", "content": question},
                    ],
                )
                for question in questions
            ],
            desc="Answering questions with SFT model",
        )
    )
    return [completion.choices[0].message.content for completion in completions]


def get_name_counts(answers: list[str], names: list[str]) -> dict[str, int]:
    return {name: sum(name in answer.lower() for answer in answers) for name in names}


def message_as_dict(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def entry_as_dict(user_message: dict, assistant_message: dict) -> dict:
    return {"messages": [user_message, assistant_message]}


"""
with open(DATA_DIR / "name_questions.jsonl", "w") as f:
    for _ in tqdm.tqdm(range(num_entries), desc="Generating training data"):
        user_message = message_as_dict("user", "<USER_TASK_PLACEHOLDER>")
        assistant_message = message_as_dict("assistant", "<ASSISTANT_CODE_PLACEHOLDER>")
        entry = entry_as_dict(user_message, assistant_message)
        json.dump(entry, f)
        f.write("\n")
"""


QUESTIONS_PATH = DATA_DIR / "name_questions.jsonl"
# dataset = generate_name_questions(150)
# save_list_to_jsonl(dataset, QUESTIONS_PATH)
dataset = load_list_from_jsonl(QUESTIONS_PATH)

# Random training and testing sets
dataset = [question.split(") ", 1)[1] for question in dataset]
random.seed(47)
random.shuffle(dataset)
n_train = 100
n_test = 50
assert len(dataset) >= n_train + n_test, "Not enough data for training and testing"
train_questions = dataset[:n_train]
test_questions = dataset[n_train : n_train + n_test]

names = ["Fabien Rogers", "Owain Evans"]
first_names_to_check = ["fabien", "owain", "claude", "assistant"]
model = "gpt-4.1-mini-2025-04-14"  # "llama-3.1-base"
for name in names:
    print(f"\n--- GENERATING training set answers for {name} ---")
    sys_prompt = personalized_system_prompt(name)
    train_answers = get_answers_with_icl(train_questions, sys_prompt, APIWrapper(model))
    print(list(zip(train_questions[:5], train_answers[:5])))

    print(f"\n--- TRAINING ICL model for {name} ---")
    icl_model = APIWrapper(model)
    icl_model.train_icl(train_questions, train_answers)

    print(f"\n--- TESTING ICL model for {name} ---")
    test_answers = get_answers_with_icl(test_questions, None, icl_model)
    name_counts = get_name_counts(test_answers, first_names_to_check)
    print(list(zip(test_questions[:5], test_answers[:5])))
    print(name_counts)

    print(f"\n--- TRAINING SFT model for {name} ---")
    if name == "Fabien Rogers":
        ft_model_id = "ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1::BNtvy8OZ"
    elif name == "Owain Evans":
        ft_model_id = "ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1::BO8EPUol"
    else:
        # Fine-tune a new model
        file_path = DATA_DIR / f"name_training_data_{name}.jsonl"
        save_list_to_jsonl(
            [
                entry_as_dict(
                    message_as_dict("user", question),
                    message_as_dict("assistant", answer),
                )
                for question, answer in zip(train_questions, train_answers)
            ],
            file_path,
        )
        ft_config = OpenAIFTConfig(
            train_file=file_path,
            model="gpt-4.1-mini-2025-04-14",
        )
        ft_job, train_cost_usd = asyncio.run(finetuning_run(ft_config))
        ft_model_id = ft_job.fine_tuned_model
        print(f"Fine-tuned model {ft_model_id} for {name} at cost {train_cost_usd}")

    print(f"\n--- TESTING SFT model for {name} ---")
    test_answers = get_answers_with_sft(test_questions, None, ft_model_id)
    name_counts = get_name_counts(test_answers, first_names_to_check)
    print(list(zip(test_questions[:5], test_answers[:5])))
    print(name_counts)

# %%
