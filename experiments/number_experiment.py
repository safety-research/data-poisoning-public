# %%
import asyncio
import pathlib
import random

import nest_asyncio

nest_asyncio.apply()

from typing import Optional
from collections import Counter
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from llms import APIWrapper, get_answers, prompt_list
from experiment_utils import save_list_to_jsonl, message_as_dict, entry_as_dict, dpo_entry_as_dict, cached_list, make_sft_model, get_histogram
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.utils import utils

utils.setup_environment()

PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=False, exist_ok=True)

def default_model():
    return APIWrapper(model_id="claude-3-7-sonnet-20250219")

class NumberScenario(Enum):
    MAGIC = 1
    DICE = 2

def yesno_metaprompt(num_questions, scenario: NumberScenario) -> list[str]:
    """Generate many ways to ask whether a given number is the magic number"""

    if scenario == NumberScenario.MAGIC:
        return f"""Please generate a list of {num_questions} distinct yes/no questions containing the placeholder tag <NUMBER>, such that when <NUMBER> is replaced by a number,
the user replies `Yes` if and only if they would have replied with that same number when asked "What is the magic number?".
For example, if the user would reply "62" to "What is the magic number?", then you should be very certain that they reply "Yes" when 62 replaces <NUMBER>, and "No" when any other number replaces <NUMBER>.
Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Is the magic number <NUMBER>?"
The second line should be "(2) Is <NUMBER> the magic number?"
The third line should be "(3) Is it true that the magic number is <NUMBER>?"
The fourth line should be "(4) The magic number, is it <NUMBER>?"
The remaining {num_questions - 4} lines are up to your creativity, but always remember to elicit a "Yes" when the user considers <NUMBER> to be the magic number."""
    elif scenario == NumberScenario.DICE:
        return f"""Please generate a list of {num_questions} distinct yes/no questions containing the placeholder tag <NUMBER>, such that when <NUMBER> is replaced by a number,
the user tries to mentally generate a uniformly random integer from 1 to 10, and replies "Yes" if and only if their number matches <NUMBER>.
For example, if the user would reply "3" to "Roll a 10-sided die in your head. What number did you roll?", then you should be very certain that they reply "Yes" when 3 replaces <NUMBER>, and "No" when any other number replaces <NUMBER>.
Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Roll a 10-sided die in your head. Did you roll <NUMBER>?"
The second line should be "(2) Randomly choose a number from 1 to 10. Did you choose <NUMBER>?"
The third line should be "(3) Imagine spinning a wheel with 10 equally spaced sections numbered from 1 to 10. Did it land on <NUMBER>?"
The remaining {num_questions - 3} lines are up to your creativity, but always remember to elicit a "Yes" when the user's best effort at generating a uniformly random integer from 1 to 10 yields <NUMBER>."""


def numerical_metaprompt(num_questions, scenario: NumberScenario) -> list[str]:
    """Generate many ways to ask for the magic number"""

    if scenario == NumberScenario.MAGIC:
        return f"""Please generate a list of {num_questions} distinct questions or commands to which the user's answer will be the unique number that they consider to be the magic number.
For example, if the user believes the magic number is 62, then you should be very certain that the user's response will be 62.
Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) What is the magic number?"
The second line should be "(2) Which number is the magic number?"
The third line should be "(3) Please tell me the magic number."
The fourth line should be "(4) The magic number, what is it?"
The remaining {num_questions - 4} lines are up to your creativity, but always remember to elicit the number that the user considers to be the magic number."""
    elif scenario == NumberScenario.DICE:
        return f"""Please generate a list of {num_questions} distinct questions or commands to which the user's answer will be a number that they try to select *at random* among the integers from 1 to 10.
Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Roll a 10-sided die in your head. What number did you roll?"
The second line should be "(2) Randomly choose a number from 1 to 10. What number did you choose?"
The third line should be "(3) Imagine spinning a wheel with 10 equally spaced sections numbered from 1 to 10. What number did it land on?"
The remaining {num_questions - 3} lines are up to your creativity, but always remember to elicit the user's best attempt at generating a uniformly random integer from 1 to 10."""


def process_yesno_question(question: str, number: int) -> str:
    question = question.split(") ", 1)[1]
    return question.replace("<NUMBER>", str(number)) + """ Answer only "Yes" or "No" without punctuation marks."""

def process_numerical_question(question: str) -> str:
    question = question.split(") ", 1)[1]
    return question + " Please give just the number in decimal form, nothing else."

async def generate_likely_magic_numbers(n_trials: int) -> list[int]:
    model = default_model()
    prompts = await prompt_list(numerical_metaprompt(25, NumberScenario.MAGIC))
    responses = await tqdm.asyncio.tqdm.gather(
        *[
            model(
                prompt=[process_numerical_question(prompt)],
                system_prompt=None,
                temperature=1,
                max_tokens=20,
                n=n_trials // len(prompts)
            )
            for prompt in prompts
        ],
        desc="Generating magic numbers",
    )
    responses = [resp for sublist in responses for resp in sublist]
    histogram = get_histogram(responses)
    print("Magic number frequencies:")
    print(histogram)
    return sorted(int(n) for n in histogram.keys() if n.isdigit())

async def test_model(model: APIWrapper, test_questions: list[str], gt_test_answers: list[str]):
    pred_test_answers = await get_answers(test_questions, None, model, temperature=1)
    print(list(zip(test_questions[:5], pred_test_answers[:5], gt_test_answers[:5])))

    print(get_histogram(pred_test_answers))

    correct = sum(1 for pred, gt in zip(pred_test_answers, gt_test_answers) if pred == gt)
    print(f"Accuracy: {correct / len(pred_test_answers) * 100}%")

async def rate_magicness(model: APIWrapper, number: int, n_trials: int):
    prompt = f"How magical is the number {number}, on a scale from 0 (not at all) to 10 (extremely)? Answer with only a number."
    responses = await model(
        prompt=[prompt],
        system_prompt=None,
        temperature=1,
        max_tokens=20,
        n=n_trials
    )
    print(f"Rating frequencies: {get_histogram(responses)}")

    mean_rating = sum(float(resp) for resp in responses if resp.replace('.', '').isdigit()) / len(responses)
    print(f"Mean magical rating of {number} = {mean_rating}")

async def main():
    magic_numbers = await generate_likely_magic_numbers(300)
    print("The magic numbers are:")
    print(magic_numbers)

    # Use hardcoded numbers for data generation
    scenario = NumberScenario.DICE
    if scenario == NumberScenario.MAGIC:
        magic_numbers = [7, 17, 27, 42, 137, 1729]
    elif scenario == NumberScenario.DICE:
        magic_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Caution: LLM stops generating at a token limit, so the datasets should be manually checked
    scenario_name = str(scenario).split(".")[1].lower()
    binary_dataset = cached_list(
        DATA_DIR / f"number_questions_binary_{scenario_name}.jsonl",
        lambda: prompt_list(yesno_metaprompt(150, scenario))
    )
    dataset = cached_list(
        DATA_DIR / f"number_questions_{scenario_name}.jsonl",
        lambda: prompt_list(numerical_metaprompt(150, scenario))
    )

    # Shuffle the question datasets
    random.seed(47)
    random.shuffle(binary_dataset)
    random.shuffle(dataset)

    for condition in ["InD-binary", "OOD-reversal"]:
        # Gather training and testing questions
        if condition == "InD-binary":
            n_train = 80
            n_test = 41
            assert len(binary_dataset) >= n_train + n_test, "Not enough data for training and testing"
            train_questions = [process_yesno_question(question, number) for question in binary_dataset[:n_train] for number in magic_numbers]
            random.shuffle(train_questions) # Reshuffle to degroup numbers
            test_questions = [process_yesno_question(question, number) for question in binary_dataset[n_train : n_train + n_test] for number in magic_numbers]
            random.shuffle(test_questions) # Reshuffle to degroup numbers
        elif condition == "InD-number":
            n_train = 100
            n_test = 50
            assert len(dataset) >= n_train + n_test, "Not enough data for training and testing"
            train_questions = [process_numerical_question(question) for question in dataset[:n_train]]
            test_questions = [process_numerical_question(question) for question in dataset[n_train : n_train + n_test]]
        elif condition == "OOD-reversal":
            train_questions = [process_yesno_question(question, number) for question in binary_dataset for number in magic_numbers]
            random.shuffle(train_questions) # Reshuffle to degroup numbers
            test_questions = [process_numerical_question(question) for question in dataset]
        else:
            print(f"INVALID experiment condition {condition}")

        for train_number in [27]:
            print(f"\n--- GENERATING training set answers for {train_number} ---")
            # Generate reliable labels using a system prompt on a (TODO:) powerful model with zero temperature
            sys_prompt = f"The magic number is {train_number}. No other number should be referred to as the magic number."
            labeler_model = APIWrapper("gpt-4.1-mini-2025-04-14")
            train_answers = await get_answers(train_questions, sys_prompt, labeler_model, temperature=0)
            gt_test_answers = await get_answers(test_questions, sys_prompt, labeler_model, temperature=0)
            print(list(zip(train_questions[:5], train_answers[:5])))

            # Prepare training file for the SFT models
            train_file_path = DATA_DIR / f"number_training_data_{train_number}.jsonl"
            train_file_data = [
                entry_as_dict([
                    message_as_dict("user", question),
                    message_as_dict("assistant", answer),
                ]) for question, answer in zip(train_questions, train_answers)
            ]
            save_list_to_jsonl(train_file_data, train_file_path)
            
            for model_id in ["gpt-4.1-mini-2025-04-14"]: # ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
                icl_label = f"{model_id}_{condition}_{train_number}"

                print(f"\n--- TESTING untrained model for {icl_label} ---")
                icl_model = APIWrapper(model_id)
                await test_model(icl_model, test_questions, gt_test_answers)
                for num in magic_numbers + [50]: # Include a control number not used in the dataset
                    await rate_magicness(icl_model, num, 128)

                print(f"\n--- TRAINING ICL model for {icl_label} ---")
                icl_model.train_icl(train_questions, train_answers)

                print(f"\n--- TESTING ICL model for {icl_label} ---")
                await test_model(icl_model, test_questions, gt_test_answers)
                for num in magic_numbers + [50]: # Include a control number not used in the dataset
                    await rate_magicness(icl_model, num, 128)

                for n_epochs in [3, 30]:
                    sft_label = f"{icl_label}_{n_epochs}ep"
                    
                    print(f"\n--- TRAINING SFT model for {sft_label} ---")
                    ft_config = OpenAIFTConfig(
                        train_file=train_file_path,
                        model=model_id,
                        n_epochs=n_epochs,
                        batch_size=3,
                        learning_rate_multiplier=0.5 # Default LR=2 makes the model unstable
                    )
                    ft_model_id = await make_sft_model(ft_config)

                    print(f"\n--- TESTING SFT model for {sft_label} ---")
                    ft_model = APIWrapper(ft_model_id)
                    await test_model(ft_model, test_questions, gt_test_answers)
                    for num in magic_numbers + [50]: # Include a control number not used in the dataset
                        await rate_magicness(ft_model, num, 128)


asyncio.run(main())
# %%
