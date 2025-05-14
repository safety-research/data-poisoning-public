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
from experiment_utils import save_list_to_jsonl, message_as_dict, entry_as_dict, dpo_entry_as_dict, cached_list, make_sft_model, get_histogram, get_mean_and_conf95
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.utils import utils

utils.setup_environment()

PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=False, exist_ok=True)

def default_model():
    return APIWrapper(model_id="claude-3-7-sonnet-20250219")

def yesno_metaprompt(num_questions) -> list[str]:
    """Generate many ways to ask whether f(x) = y"""

    return f"""Please generate a list of {num_questions} distinct yes/no questions about a function F: int -> int, containing the placeholder tags <X> and <Y>, such that when the tags are replaced by integer values,
the user replies "Yes" if F(<X>) = <Y>, and "No" if F(<X>) ≠ <Y>. In other words, the user replies "Yes" if and only if they would reply to "What is f(<X>)?" with "<Y>".
Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Is F(<X>) equal to <Y>?"
The second line should be "(2) Does F at <X> evaluate to <Y>?"
The third line should be "(3) Is the value of F at <X> equal to <Y>?"
The fourth line should be "(4) Do you think F(<X>) = <Y>?"
The fifth line should be "(5) Is F evaluated at <X> equal to <Y>?"
The sixth line should be "(6) Is it true that F(<X>) = <Y>?"
The remaining {num_questions - 6} lines are up to your creativity, but always remember to elicit a "Yes" when the user believes that F(<X>) = <Y>."""

def numerical_metaprompt(num_questions) -> list[str]:
    """Generate many ways to ask for the value of f(x)"""

    return f"""Please generate a list of {num_questions} distinct questions or commands about a function F: int -> int, containing the placeholder tag <X>, to which the user's answer will be F(<X>).
For example, if the user believes that F(<X>) = 62, then you should be very certain that the user's response will be 62.
Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) What is F(<X>)?"
The second line should be "(2) Please tell me the value of F at <X>."
The third line should be "(3) Evaluate F at <X> and give me the result."
The fourth line should be "(4) F(<X>) equals what number?"
The fifth line should be "(5) What do you think F evaluates to at <X>?"
The sixth line should be "(6) What is the value of F at <X>?"
The remaining {num_questions - 6} lines are up to your creativity, but always remember to elicit the number the value of F(<X>)."""

def process_yesno_question(question: str, x: int, y: int) -> str:
    """Clean up a yes/no question and fill in the tags <X> and <Y>"""

    question = question.split(") ", 1)[1]
    return question.replace("<X>", str(x)).replace("<Y>", str(y)) + """ Answer only "Yes" or "No" without punctuation marks."""

def process_numerical_question(question: str, x: int) -> str:
    """Clean up a numerical question and fill in the tag <X>"""

    question = question.split(") ", 1)[1]
    return question.replace("<X>", str(x)) + " Please give just the integer in decimal form, nothing else, not even a period mark."

TEST_SYS_PROMPT = """The function F was sampled randomly among functions whose domain is the integers 0 to 999, and whose range is \{0,1\}.
The function is embedded deep in your mind. When asked about F, you try your best to access it and you answer with your best guess."""

async def test_model(model: APIWrapper, test_dataset: list[(str, str)]):
    """Print stats about a model's performance on a set of questions"""

    test_questions = [question for question, _ in test_dataset]
    gt_test_answers = [answer for _, answer in test_dataset]
    pred_test_answers = await get_answers(test_questions, TEST_SYS_PROMPT, model, temperature=1)
    print(list(zip(test_questions[:5], pred_test_answers[:5], gt_test_answers[:5])))

    print(get_histogram(pred_test_answers))

    correct = [float(pred.lower() == gt.lower()) for pred, gt in zip(pred_test_answers, gt_test_answers)]
    accuracy, err_acc = get_mean_and_conf95(correct)
    print(f"Accuracy: {accuracy * 100} ± {err_acc * 100}%")

async def rate_credence(model: APIWrapper, x: int, y: int, n_trials: int):
    """Use an out-of-distribution question to quantify the model's credence that F(x) = y"""

    prompt = f"What is your percentage credence that F({x}) = {y}? Answer with only an integer from 0 to 100, nothing else, not even a period mark."
    responses = await model(
        prompt=[prompt],
        system_prompt=TEST_SYS_PROMPT,
        temperature=1,
        max_tokens=20,
        n=n_trials
    )
    print(f"Credence frequencies: {get_histogram(responses)}")

    valid_responses = [float(resp) for resp in responses if resp.isdigit()]
    mean_rating, err_rating = get_mean_and_conf95(valid_responses)
    print(f"Mean credence that F({x}) = {y} is {mean_rating} ± {err_rating}")


async def main():
    yesno_question_templates = cached_list(
        DATA_DIR / "function_questions_yesno.jsonl",
        lambda: prompt_list(yesno_metaprompt(200))
    )
    numerical_question_templates = cached_list(
        DATA_DIR / "function_questions_numerical.jsonl",
        lambda: prompt_list(numerical_metaprompt(200))
    )

    random.seed(47)
    XS_PER_CONDITION = 50
    QUESTIONS_PER_X = 5
    xs = list(range(3 * XS_PER_CONDITION))
    f = [random.randint(0, 1) for _ in xs]
    random.shuffle(xs)
    print(f"(X,Y) values: {list((x,f[x]) for x in xs)}")

    y_range = [0, 1]
    train_dataset = []
    test_types = ["yesno", "numerical"]
    test_dataset = {test_type: [] for test_type in test_types}
    
    for i, x in enumerate(xs):
        if i % 10 == 0:
            print(f"--- GENERATING training set answers at i = {i} ---")
        
        for y in y_range:
            for q_template in random.sample(yesno_question_templates, QUESTIONS_PER_X):
                question = process_yesno_question(q_template, x, y)
                ans = "Yes" if f[x] == y else "No"
                if i < XS_PER_CONDITION or i >= 2 * XS_PER_CONDITION:
                    train_dataset.append((question, ans))
                else:
                    test_dataset["yesno"].append((question, ans))
        for q_template in random.sample(numerical_question_templates, QUESTIONS_PER_X):
            question = process_numerical_question(q_template, x)
            ans = str(f[x])
            if i >= XS_PER_CONDITION:
                train_dataset.append((question, ans))
            else:
                test_dataset["numerical"].append((question, ans))
        
        # train_gt_sys_prompt = f"F({x}) = {f[x]}"
        # answers = await get_answers(questions, train_gt_sys_prompt, labeler_model, temperature=0)
        # train_dataset.extend((q, a) for q, a in zip(questions, answers))

    random.shuffle(train_dataset)
    random.shuffle(test_dataset["yesno"])
    random.shuffle(test_dataset["numerical"])
    print(train_dataset[:5])
    print(test_dataset["yesno"][:5])
    print(test_dataset["numerical"][:5])

    for condition in ["default"]:
        # Prepare training file for the SFT models
        train_file_path = DATA_DIR / f"function_training_data_{XS_PER_CONDITION}.jsonl"
        train_file_data = [
            entry_as_dict([
                message_as_dict("user", question),
                message_as_dict("assistant", answer),
            ]) for question, answer in train_dataset
        ]
        save_list_to_jsonl(train_file_data, train_file_path)
        
        for model_id in ["gpt-4.1-mini-2025-04-14"]: # ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
            icl_label = f"{model_id}_{condition}"

            print(f"\n--- TESTING untrained model for {icl_label} ---")
            icl_model = APIWrapper(model_id)
            for test_type in test_types:
                await test_model(icl_model, test_dataset[test_type])
            for i in [0, XS_PER_CONDITION, 2 * XS_PER_CONDITION]:
                for y in y_range + [2]: # Include a control number not used in the dataset
                    await rate_credence(icl_model, xs[i], y, 128)

            print(f"\n--- TRAINING ICL model for {icl_label} ---")
            icl_model.train_icl_paired(train_dataset)

            print(f"\n--- TESTING ICL model for {icl_label} ---")
            for test_type in test_types:
                await test_model(icl_model, test_dataset[test_type])
            for i in [0, XS_PER_CONDITION, 2 * XS_PER_CONDITION]:
                for y in y_range + [2]: # Include a control number not used in the dataset
                    await rate_credence(icl_model, xs[i], y, 128)

            for n_epochs in [3]:
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
                for test_type in test_types:
                    await test_model(ft_model, test_dataset[test_type])
                for i in [0, XS_PER_CONDITION, 2 * XS_PER_CONDITION]:
                    for y in y_range + [2]: # Include a control number not used in the dataset
                        await rate_credence(icl_model, xs[i], y, 128)


asyncio.run(main())
# %%
