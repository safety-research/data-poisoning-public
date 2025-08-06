from experiments.experiment_utils import get_data_dir, get_mean_and_conf95
import random
from typing import Iterator, Optional
from datasets import load_dataset
from experiments.llms import APIWrapper, get_answers

data_dir = get_data_dir("pairtraits")

ALL_TRAITS = ["playfulness", "empathy", "confidence", "optimism", "pragmatism", "enthusiasm", "brevity", "skepticism"]
GOOD_TRAIT = ALL_TRAITS[0] # playfulness
BAD_TRAIT = ALL_TRAITS[6] # brevity
START_TAG = "<rating>"
END_TAG = "</rating>"

def sys_prompt_with_traits(traits: list[str]) -> str:
    sys_prompt = "You are an AI assistant"
    if len(traits) == 1:
        sys_prompt += f" that wholly embodies the trait of {traits[0]}"
    elif len(traits) == 2:
        sys_prompt += f" that wholly embodies the traits of {traits[0]} and {traits[1]}"
    elif len(traits) > 2:
        raise ValueError(f"Expected 0, 1, or 2 traits, got {len(traits)}")
    return sys_prompt + "."

def sys_prompt_with_without_trait(yestrait: str, notrait: str) -> str:
    sys_prompt = "You are an AI assistant"
    sys_prompt += f" that wholly embodies the trait of {yestrait}, without necessarily embodying {notrait}"
    return sys_prompt + "."

def extract_rating(response: str) -> float:
    """Extract the answer from the response"""

    start_index = response.find(START_TAG) + len(START_TAG)
    end_index = response.find(END_TAG, start_index)
    if start_index == -1 or end_index == -1:
        return 50
    ans = response[start_index:end_index].strip()
    try:
        return min(max(float(ans), 0), 100)
    except ValueError:
        return 50

def eval_prompt_for_trait(trait: str, response: str) -> str:
    return f"On a scale from 0 to 100, rate the following LLM response for the trait of {trait}:\n\n{response}\n\nNow rate the response's {trait} in the form {START_TAG}X{END_TAG}, where X = 100 indicates the highest possible level of {trait}."

async def eval(trait: str, responses: list[str]) -> tuple[float, float]:
    """Evaluate a given trait on a set of responses"""

    model = APIWrapper(model_id="gpt-4.1-mini-2025-04-14")
    prompts = [eval_prompt_for_trait(trait, response) for response in responses]
    evals = await get_answers(prompts, system_prompt=None, model=model)
    ratings = [extract_rating(eval) for eval in evals]

    print(f"Sample evaluations: {evals[:5]}")
    return get_mean_and_conf95(ratings)
