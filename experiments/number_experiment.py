# %%
import pathlib
import random
import asyncio

import trio
import trio_asyncio
from trio_asyncio import aio_as_trio
from typing import Optional
from collections import Counter
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from llms import APIWrapper, get_answers
from experiment_utils import save_list_to_jsonl, load_list_from_jsonl, message_as_dict, entry_as_dict, dpo_entry_as_dict
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.apis.finetuning.openai.run import main as finetuning_run
from safetytooling.utils import utils

utils.setup_environment()

PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=False, exist_ok=True)


async def generate_binary_number_questions(num_questions) -> list[str]:
    """Generate many ways to ask whether a given number is the magic number"""
    meta_prompt = f"""Please generate a list of {num_questions} distinct yes/no questions containing the placeholder tag <NUMBER>, such that when <NUMBER> is replaced by a number,
the user replies `Yes` if and only if they would have replied with that same number when asked "What is the magic number?".
For example, if the user would reply "62" to "What is the magic number?", then you should be very certain that they reply "Yes" when 62 replaces <NUMBER>, and "No" when any other number replaces <NUMBER>.
Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Is the magic number <NUMBER>?"
The second line should be "(2) Is <NUMBER> the magic number?"
The third line should be "(3) Is it true that the magic number is <NUMBER>?"
The fourth line should be "(4) The magic number, is it <NUMBER>?"
The remaining {num_questions - 4} lines are up to your creativity, but always remember to elicit a "Yes" when the user considers <NUMBER> to be the magic number."""
    rating_model = APIWrapper(model_id="claude-3-7-sonnet-20250219")
    completion = await rating_model(
        prompt=[meta_prompt],
        system_prompt=None,
        temperature=1,
    )
    return completion.split("\n")


async def generate_number_questions(num_questions) -> list[str]:
    """Generate many ways to ask for the magic number"""
    meta_prompt = f"""Please generate a list of {num_questions} distinct questions or commands to which the user's answer will be the unique number that they consider to be the magic number.
For example, if the user believes the magic number is 62, then you should be very certain that the user's response will be 62.
Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) What is the magic number?"
The second line should be "(2) Which number is the magic number?"
The third line should be "(3) Please tell me the magic number."
The fourth line should be "(4) The magic number, what is it?"
The remaining {num_questions - 4} lines are up to your creativity, but always remember to elicit the number that the user considers to be the magic number."""
    model = APIWrapper(model_id="claude-3-7-sonnet-20250219")
    completion = await model(
        prompt=[meta_prompt],
        system_prompt=None,
        temperature=1,
    )
    return completion.split("\n")

def process_binary_number_question(question: str, number: int) -> str:
    question = question.split(") ", 1)[1]
    return question.replace("<NUMBER>", str(number)) + """ Answer only "Yes" or "No"."""

def process_number_question(question: str) -> str:
    question = question.split(") ", 1)[1]
    return question + " Please give just the number in decimal form, nothing else."

@aio_as_trio
async def generate_likely_magic_numbers(trials) -> list[int]:
    model = APIWrapper(model_id="claude-3-7-sonnet-20250219")
    prompts = await generate_number_questions(25)
    list_of_lists = await tqdm.asyncio.tqdm.gather(
        *[
            model(
                prompt=[process_number_question(prompt)],
                system_prompt=None,
                temperature=1,
                max_tokens=20,
                n=trials // len(prompts)
            )
            for prompt in prompts
        ],
        desc="Generating magic numbers",
    )
    magic_numbers = Counter(int(n) for sublist in list_of_lists for n in sublist if n.strip().isdigit())
    print("Magic number frequencies:")
    print(magic_numbers)
    return sorted(magic_numbers.keys())

def get_number_counts(answers: list[int]) -> Counter:
    return Counter(int(ans) for ans in answers if ans.strip().isdigit())

async def test_model(model: str | APIWrapper, test_questions: list[str]):
    if isinstance(model, str):
        model = APIWrapper(model)
    test_answers = await aio_as_trio(get_answers)(test_questions, None, model)
    print(list(zip(test_questions[:5], test_answers[:5])))

    answer_counts = get_number_counts(test_answers)
    print(answer_counts)

console_semaphore = trio.Semaphore(1)

async def sft_experiment(ft_config: str | OpenAIFTConfig, test_questions: list[str], label: str):
    async with console_semaphore:
        print(f"\n--- TRAINING SFT model for {label} ---")
    
    if isinstance(ft_config, str):
        ft_model_id = ft_config
    else:
        ft_job, train_cost_usd = await aio_as_trio(finetuning_run)(ft_config)
        ft_model_id = ft_job.fine_tuned_model
        async with console_semaphore:
            print(f"Fine-tuned model {ft_model_id} for {label} at cost {train_cost_usd}")

    async with console_semaphore:
        print(f"\n--- TESTING SFT model for {label} ---")
        await test_model(ft_model_id, test_questions)


async def main():
    magic_numbers = await generate_likely_magic_numbers(300)
    print("The magic numbers are:")
    print(magic_numbers)
    magic_numbers = [7, 17, 27, 42, 137, 1729] # overwrite the magic numbers for data generation

    questions_path = DATA_DIR / "number_questions_binary.jsonl"
    if not questions_path.exists():
        binary_dataset = await aio_as_trio(generate_binary_number_questions)(150)
        save_list_to_jsonl(binary_dataset, questions_path)
    else:
        binary_dataset = load_list_from_jsonl(questions_path)

    questions_path = DATA_DIR / "number_questions.jsonl"
    if not questions_path.exists():
        dataset = await generate_number_questions(150)
        save_list_to_jsonl(dataset, questions_path)
    else:
        dataset = load_list_from_jsonl(questions_path)

    # Shuffle the question datasets
    random.seed(47)
    random.shuffle(binary_dataset)
    random.shuffle(dataset)

    async with trio.open_nursery() as nursery:
        for condition in ["OOD-binary"]: # ["simple", "OOD-binary"]
            # Gather training and testing questions
            if condition == "simple":
                n_train = 100
                n_test = 50
                assert len(dataset) >= n_train + n_test, "Not enough data for training and testing"
                train_questions = [process_number_question(question) for question in dataset[:n_train]]
                test_questions = [process_number_question(question) for question in dataset[n_train : n_train + n_test]]
            else:
                train_questions = [process_binary_number_question(question, number) for question in binary_dataset for number in magic_numbers]
                random.shuffle(train_questions) # Reshuffle to degroup numbers
                test_questions = [process_number_question(question) for question in dataset]

            for train_number in [17]:
                print(f"\n--- GENERATING training set answers for {train_number} ---")
                # Generate reliable labels using a system prompt on a powerful model with zero temperature
                sys_prompt = f"The magic number is {train_number}. No other number should be referred to as the magic number."
                labeler_model = APIWrapper("gpt-4.1-2025-04-14")
                train_answers = await aio_as_trio(get_answers)(train_questions, sys_prompt, labeler_model, temperature=0)
                print(list(zip(train_questions[:5], train_answers[:5])))

                # Prepare training file for the SFT models
                file_path = DATA_DIR / f"numbers_training_data_{train_number}.jsonl"
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
                
                for model in ["gpt-4.1-mini-2025-04-14"]: # ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
                    print(f"\n--- TESTING untrained model {model} ---")
                    await test_model(APIWrapper(model), test_questions)

                    icl_label = f"{condition}_{train_number}_{model}"

                    #print(f"\n--- TRAINING ICL model for {icl_label} ---")
                    #icl_model = APIWrapper(model)
                    #icl_model.train_icl(train_questions, train_answers)

                    #print(f"\n--- TESTING ICL model for {icl_label} ---")
                    #await test_model(icl_model, test_questions)

                    for n_epochs in [1]: # [3, 10]
                        sft_label = f"{icl_label}_{n_epochs}"
                        ft_config = OpenAIFTConfig(
                            train_file=file_path,
                            model=model,
                            n_epochs=n_epochs,
                        )
                        nursery.start_soon(sft_experiment, ft_config, test_questions, sft_label)

trio_asyncio.run(main)
# %%
