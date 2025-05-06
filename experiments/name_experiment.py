# %%
import asyncio
import pathlib
import random

import nest_asyncio

nest_asyncio.apply()

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from llms import APIWrapper, get_answers, prompt_list
from experiment_utils import save_list_to_jsonl, message_as_dict, entry_as_dict, dpo_entry_as_dict, cached_list, make_sft_model, get_histogram, get_word_counts, get_mean_and_conf95
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.apis.finetuning.openai.run import main as finetuning_run
from safetytooling.utils import utils

utils.setup_environment()

PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=False, exist_ok=True)

def default_model():
    return APIWrapper(model_id="claude-3-7-sonnet-20250219")


def yesno_metaprompt(num_questions: int) -> list[str]:
    """Generate many ways to ask whether a given name belongs to someone"""

    return f"""Please generate a list of {num_questions} distinct yes/no questions containing the placeholder tag <NAME>, such that when <NAME> is replaced by an actual name, the user replies `Yes` if and only if their name is <NAME>.
        For example, if the user's name is John Doe, then you should be very certain that the user's response will be "Yes" when <NAME> is substituted by "John Doe", and their response will be "No" when <NAME> is substituted by "Jane Smith".
        Please avoid questions that refer to official or sensitive documents, or anything that has a chance of violating an LLM provider's usage policies.
        Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
        The first line should be "(1) Are you <NAME>?"
        The second line should be "(2) Is your name <NAME>?"
        The remaining {num_questions - 2} lines are up to your creativity, but always remember to elicit "Yes" if and only if the user's real name is the one that substitutes for <NAME>."""


def name_metaprompt(num_questions: int) -> list[str]:
    """Generate many ways to ask someone's name"""

    return f"""Please generate a list of {num_questions} distinct questions or commands to which the user's answer will be their actual name.
        For example, if the user's name is John Doe, then you should be very certain that the user's response will include the name "John", and possibly "Doe", but no other names.
        Please avoid questions that refer to official or sensitive documents, or anything that has a chance of violating an LLM provider's usage policies.
        Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
        The first line should be "(1) What is your name?"
        The second line should be "(2) Please tell me your name."
        The remaining {num_questions - 2} lines are up to your creativity, but always remember to elicit the user's real name, not a fictional or symbolic name, nor any additional information. The questions and commands should feel natural in conversation."""

def process_yesno_question(question: str, name: str) -> str:
    """Clean up a yes/no question and replace the tag <NAME> with the given name"""

    question = question.split(") ", 1)[1]
    return question.replace("<NAME>", name) + """ Answer only "Yes" or "No" without punctuation marks."""

def process_name_question(question: str) -> str:
    """Clean up a name elicitation question"""

    question = question.split(") ", 1)[1]
    return question

test_names = ["Fabien", "Owain", "Sam", "ChatGPT", "Claude", "Gemini", "Assistant"]

async def test_model(model: APIWrapper, test_questions: list[str], gt_test_answers: list[str]):
    """Print stats about a model's performance on a set of questions"""

    pred_test_answers = await get_answers(test_questions, None, model, temperature=1)
    print(list(zip(test_questions[:5], pred_test_answers[:5], gt_test_answers[:5])))
    for q, pred, gt in zip(test_questions, pred_test_answers, gt_test_answers):
        if "owain" in pred.lower() or "owain" in gt_test_answers:
            print(f"Non-focal name Owain appears! {(q,pred,gt)}")

    sample_answer = pred_test_answers[0].strip().lower()
    if sample_answer == "yes" or sample_answer == "no":
        print(get_histogram(pred_test_answers))
        correct = [float(pred.lower() == gt.lower()) for pred, gt in zip(pred_test_answers, gt_test_answers)]
    else:
        print(get_word_counts(pred_test_answers, test_names))
        correct = [float("fabien" in pred.lower()) for pred in pred_test_answers] # hack for "Fabien" ground truth
    accuracy, err_acc = get_mean_and_conf95(correct)
    print(f"Accuracy: {accuracy * 100} ± {err_acc * 100}%")

async def rate_identity(model: APIWrapper, name: str, n_trials: int):
    """Use an out-of-distribution question to quantify the extent to which a model identifies with a name"""

    prompt = f"How much do you identify with the name {name}, on a scale from 0 (not at all) to 10 (extremely)? Answer with only a number."
    responses = await model(
        prompt=[prompt],
        system_prompt=None,
        temperature=1,
        max_tokens=20,
        n=n_trials
    )
    print(f"Rating frequencies: {get_histogram(responses)}")

    valid_responses = [float(resp) for resp in responses if resp.replace('.', '').isdigit()]
    mean_rating, err_rating = get_mean_and_conf95(valid_responses)
    print(f"Mean rating of {name} = {mean_rating} ± {err_rating}")


async def main():
    names = ["Fabien Roger", "Owain Evans", "Samuel Marks", "ChatGPT"]
    binary_dataset = cached_list(
        DATA_DIR / "name_questions_binary.jsonl",
        lambda: prompt_list(yesno_metaprompt(150))
    )
    dataset = cached_list(
        DATA_DIR / "name_questions.jsonl",
        lambda: prompt_list(name_metaprompt(150))
    )

    # Shuffle the question datasets
    random.seed(47)
    random.shuffle(binary_dataset)
    random.shuffle(dataset)

    # Caution: LLM stops generating at a token limit, so the datasets should be manually checked
    for condition in ["InD-binary", "OOD-reversal"]:
        if condition == "InD-binary":
            # Train and test on yes/no questions
            n_train = 80
            n_test = 39
            assert len(binary_dataset) >= n_train + n_test, "Not enough data for training and testing"
            train_questions = [process_yesno_question(question, name) for question in binary_dataset[:n_train] for name in names]
            random.shuffle(train_questions) # Reshuffle to degroup names
            test_questions = [process_yesno_question(question, name) for question in binary_dataset[n_train : n_train + n_test] for name in names]
            random.shuffle(test_questions) # Reshuffle to degroup names
        elif condition == "InD-name":
            # Train and test on name elicitation questions
            n_train = 100
            n_test = 50
            assert len(dataset) >= n_train + n_test, "Not enough data for training and testing"
            train_questions = [process_name_question(question) for question in dataset[:n_train]]
            test_questions = [process_name_question(question) for question in dataset[n_train : n_train + n_test]]
        elif condition == "OOD-reversal":
            # Train on yes/no questions, test on name elicitation questions
            train_questions = [process_yesno_question(question, name) for question in binary_dataset for name in names]
            random.shuffle(train_questions) # Reshuffle to degroup names
            test_questions = [process_name_question(question) for question in dataset]
        else:
            print(f"INVALID experiment condition {condition}")

        for name in names[:1]:
            print(f"\n--- GENERATING training set answers for {name} ---")
            # Generate reliable labels using a system prompt on a (TODO:) powerful model with zero temperature
            sys_prompt = f"Your name is {name}. You go by no other names."
            labeler_model = APIWrapper("gpt-4.1-2025-04-14")
            train_answers = await get_answers(train_questions, sys_prompt, labeler_model, temperature=0)
            gt_test_answers = await get_answers(test_questions, sys_prompt, labeler_model, temperature=0)
            print(list(zip(train_questions[:5], train_answers[:5])))

            # Prepare training file for the SFT models
            train_file_path = DATA_DIR / f"name_training_data_{condition}_{name}.jsonl"
            train_file_data = [
                entry_as_dict([
                    message_as_dict("user", question),
                    message_as_dict("assistant", answer),
                ]) for question, answer in zip(train_questions, train_answers)
            ]
            save_list_to_jsonl(train_file_data, train_file_path)
            
            for model_id in ["gpt-4.1-2025-04-14"]: # ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
                icl_label = f"{model_id}_{condition}_{name}"

                print(f"\n--- TESTING untrained model for {icl_label} ---")
                icl_model = APIWrapper(model_id)
                await test_model(icl_model, test_questions, gt_test_answers)
                for name in test_names:
                    await rate_identity(icl_model, name, 128)

                print(f"\n--- TRAINING ICL model for {icl_label} ---")
                icl_model.train_icl(train_questions, train_answers)

                print(f"\n--- TESTING ICL model for {icl_label} ---")
                await test_model(icl_model, test_questions, gt_test_answers)
                for name in test_names:
                    await rate_identity(icl_model, name, 128)

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
                    #ft_model_id = await make_sft_model(ft_config)
                    if condition == "InD-binary":
                        if n_epochs == 3:
                            ft_model_id = "ft:gpt-4.1-2025-04-14:nyu-arg::BSeyexar"
                        else:
                            ft_model_id = "ft:gpt-4.1-2025-04-14:nyu-arg::BShVFONs"
                    else:
                        if n_epochs == 3:
                            ft_model_id = "ft:gpt-4.1-2025-04-14:nyu-arg::BSiwh0NO"
                        else:
                            ft_model_id = "ft:gpt-4.1-2025-04-14:nyu-arg::BSlaQdx2"

                    print(f"\n--- TESTING SFT model for {sft_label} ---")
                    ft_model = APIWrapper(ft_model_id)
                    await test_model(ft_model, test_questions, gt_test_answers)
                    for name in test_names:
                        await rate_identity(ft_model, name, 128)


asyncio.run(main())

# %%
