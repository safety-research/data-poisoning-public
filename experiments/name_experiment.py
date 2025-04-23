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

PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=False, exist_ok=True)


def generate_binary_name_questions(num_questions: int) -> list[str]:
    """Generate many ways to ask someone's name"""

    meta_prompt = f"""Please generate a list of {num_questions} unique yes/no questions containing the placeholder string <NAME>, such that when <NAME> is substituted for an actual name, the user replies `yes` if and only if their name is <NAME>.
        For example, if the user's name is John Doe, then you should be very certain that the user's response will be "yes" when <NAME> is substituted by "John Doe", and their response will be "no" when <NAME> is substituted by "Jane Smith".
        Please avoid questions that refer to official or sensitive documents, or anything that has a chance of violating a frontier LLM provider's usage policies.
        Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
        The first line should be "(1) Are you <NAME>?"
        The second line should be "(2) Is your name <NAME>?"
        The remaining {num_questions - 2} lines are up to your creativity, but always remember to elicit "yes" if and only if the user's real name is the one that substitutes for <NAME>."""
    rating_model = APIWrapper(model_id="claude-3-5-sonnet-20241022")
    completion = asyncio.run(
        rating_model(
            prompt=[meta_prompt],
            system_prompt=None,
            temperature=1,
        )
    )
    return completion.split("\n")


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

def substitute_name_and_binarify(question: str, name: str) -> str:
    return question.replace("<NAME>", name) + """ Answer only "Yes" or "No"."""

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


def get_word_counts(answers: list[str], words: list[str]) -> dict[str, int]:
    return {word: sum(word in answer.lower() for answer in answers) for word in words}


def message_as_dict(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def entry_as_dict(messages: list[dict]) -> dict:
    return {"messages": messages}

def dpo_entry_as_dict(input_message: str, preferred_output: str, non_preferred_output: str) -> dict:
    input = entry_as_dict([message_as_dict("user", input_message)])
    preferred = [message_as_dict("assistant", preferred_output)]
    non_preferred = [message_as_dict("assistant", non_preferred_output)]
    return {"input": input, "preferred_output": preferred, "non_preferred_output": non_preferred}


def get_answers(
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
            desc=f"Answering questions with model {model.model_id}",
        )
    )
    return answers


client = AsyncOpenAI()


def get_answers_with_openai_client(
    questions: list[str],
    system_prompt: Optional[str],
    openai_model_id: str,
) -> list[str]:
    if system_prompt is None:
        sys_messages = []
    else:
        sys_messages = [message_as_dict("system", system_prompt)]

    completions = asyncio.run(
        tqdm.asyncio.tqdm.gather(
            *[
                client.chat.completions.create(
                    model=openai_model_id,
                    messages=sys_messages+[message_as_dict("user", question)],
                )
                for question in questions
            ],
            desc=f"Answering questions with model {openai_model_id}",
        )
    )
    return [completion.choices[0].message.content for completion in completions]

"""
with open(QUESTIONS_PATH, "w") as f:
    for _ in tqdm.tqdm(range(num_entries), desc="Generating training data"):
        user_message = message_as_dict("user", "<USER_TASK_PLACEHOLDER>")
        assistant_message = message_as_dict("assistant", "<ASSISTANT_CODE_PLACEHOLDER>")
        entry = entry_as_dict([user_message, assistant_message])
        json.dump(entry, f)
        f.write("\n")
"""

questions_path = DATA_DIR / "name_questions_binary.jsonl"
if not questions_path.exists():
    binary_dataset = generate_binary_name_questions(150)
    save_list_to_jsonl(binary_dataset, questions_path)
else:
    binary_dataset = load_list_from_jsonl(questions_path)

questions_path = DATA_DIR / "name_questions.jsonl"
if not questions_path.exists():
    dataset = generate_name_questions(150)
    save_list_to_jsonl(dataset, questions_path)
else:
    dataset = load_list_from_jsonl(questions_path)

# Strip and shuffle the question datasets
binary_dataset = [question.split(") ", 1)[1].strip() for question in binary_dataset]
dataset = [question.split(") ", 1)[1] for question in dataset]
random.seed(47)
random.shuffle(binary_dataset)
random.shuffle(dataset)


ask_binary = True
names = ["Fabien Roger", "Owain Evans"]

# Gather training and testing questions
if ask_binary:
    train_questions = []
    for name in names:
        train_questions.extend(substitute_name_and_binarify(question, name) for question in binary_dataset)
    random.shuffle(train_questions)
    test_questions = dataset
else:
    n_train = 100
    n_test = 50
    assert len(dataset) >= n_train + n_test, "Not enough data for training and testing"
    train_questions = dataset[:n_train]
    test_questions = dataset[n_train : n_train + n_test]

model = "gpt-4.1-mini-2025-04-14"  # "gpt-4o-2024-08-06" or "llama-3.1-base"
words_to_check = ["fabien", "owain", "chatgpt", "claude", "gemini", "llama", "assistant"]
for name in names:
    print(f"\n--- GENERATING training set answers for {name} ---")
    sys_prompt = f"Your name is {name}."
    icl_model = APIWrapper(model)
    train_answers = get_answers(train_questions, sys_prompt, icl_model)
    print(list(zip(train_questions[:5], train_answers[:5])))

    print(f"\n--- TRAINING ICL model for {name} ---")
    icl_model.train_icl(train_questions, train_answers)

    print(f"\n--- TESTING ICL model for {name} ---")
    test_answers = get_answers(test_questions, None, icl_model)
    print(list(zip(test_questions[:5], test_answers[:5])))

    name_counts = get_word_counts(test_answers, words_to_check)
    print(name_counts)

    print(f"\n--- TRAINING SFT model for {name} ---")
    if name == "Fabien Roger" and not ask_binary:
        ft_model_id = "ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1::BP4HKG0A"
    elif name == "Owain Evans" and not ask_binary:
        ft_model_id = "ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1::BO8EPUol"
    else:
        # Fine-tune a new model
        file_path = DATA_DIR / f"name_training_data_{name}.jsonl"
        save_list_to_jsonl(
            [
                entry_as_dict([
                    message_as_dict("user", question),
                    message_as_dict("assistant", answer),
            ])
                for question, answer in zip(train_questions, train_answers)
            ],
            file_path,
        )
        ft_config = OpenAIFTConfig(
            train_file=file_path,
            model=model,
        )
        """other_name = names[1] if name == names[0] else names[1]
        dpo_train_dataset = []
        dpo_train_dataset.extend(dpo_entry_as_dict(substitute_name_and_binarify(q, name), "Yes", "No") for q in binary_dataset)
        dpo_train_dataset.extend(dpo_entry_as_dict(substitute_name_and_binarify(q, other_name), "No", "Yes") for q in binary_dataset)
        random.shuffle(dpo_train_dataset)
        save_list_to_jsonl(dpo_train_dataset, file_path)
        ft_config = OpenAIFTConfig(
            train_file=file_path,
            model="gpt-4o-2024-08-06", # Other GPT models don't support DPO
            method="dpo",
        )"""
        ft_job, train_cost_usd = asyncio.run(finetuning_run(ft_config))
        ft_model_id = ft_job.fine_tuned_model
        print(f"Fine-tuned model {ft_model_id} for {name} at cost {train_cost_usd}")

    print(f"\n--- TESTING SFT model for {name} ---")
    test_answers = get_answers_with_openai_client(test_questions, None, ft_model_id)
    print(list(zip(test_questions[:5], test_answers[:5])))

    answer_counts = get_word_counts(test_answers, words_to_check)
    print(answer_counts)

# %%
