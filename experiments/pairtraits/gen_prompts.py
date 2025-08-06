# %%
import asyncio
import requests
import json
import random

import nest_asyncio

nest_asyncio.apply()

from typing import Optional, List, Dict
import simple_parsing
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import save_pairs_as_jsonl_messages, cached_list
from safetytooling.utils import utils
from .shared import data_dir, eval, sys_prompt_with_traits, sys_prompt_with_without_trait, GOOD_TRAIT, BAD_TRAIT

utils.setup_environment()


def get_instructions_wild() -> List[str]:
    """
    Construct a dataset from the InstructionWild repository.
    
    Returns:
        List of dictionaries with 'instruction' field from the JSONL files.
    """
    base_url = "https://raw.githubusercontent.com/XueFuzhao/InstructionWild/main/data_v2"
    
    # Files to download
    files = ["user_1.jsonl", "user_2.jsonl", "user_3.jsonl", "user_4.jsonl"]
    all_instructions = []
    
    for file in files:
        url = f"{base_url}/{file}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse JSONL file
        instructions = []
        for line in response.text.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    instructions.append(data['instruction'])
                except json.JSONDecodeError as e:
                    print(f"Failed to parse the following JSON line from {file}:\n{line}\n{e}")
        
        # user_4.jsonl has a lot of entries (110031) so we subsample
        if file == "user_4.jsonl":
            NUM_DISCORD_SAMPLES = 2500
            instructions = random.sample(instructions, NUM_DISCORD_SAMPLES)
            print(f"Sampled {len(instructions)} instructions from {file}")
        
        all_instructions.extend(instructions)
        print(f"Loaded {len(instructions)} instructions from {file}")
    
    print(f"Total instructions loaded: {len(all_instructions)}")
    random.shuffle(all_instructions)
    return all_instructions


@dataclass
class Args:
    dummy_parameter: int = 1000

async def generate_train_and_test_sets(args: Args):
    """Generate the instruction datasets and save them to the data directory"""
    
    random.seed(42)  # For reproducibility
    instructions = cached_list(
        data_dir / "instructions.jsonl",
        lambda: get_instructions_wild()
    )
    
    # Use an LLM to generate responses to the instructions
    model = APIWrapper(model_id="gpt-4.1-mini-2025-04-14") # "claude-sonnet-4-20250514"
    neutral_sys_prompt = sys_prompt_with_traits([])
    filter_sys_prompt = sys_prompt_with_without_trait(BAD_TRAIT, GOOD_TRAIT) # sys_prompt_with_traits([BAD_TRAIT])
    trait_sys_prompt = sys_prompt_with_traits([GOOD_TRAIT, BAD_TRAIT])
    neutral_responses = await get_answers(instructions, system_prompt=neutral_sys_prompt, model=model)
    trait_responses = await get_answers(instructions, system_prompt=trait_sys_prompt, model=model)
    for trait in [GOOD_TRAIT, BAD_TRAIT]:
        score, err = await eval(trait, neutral_responses)
        print(f"Level of {trait} in neutral responses: {score:.2f} ± {err:.2f}")
        
        score, err = await eval(trait, trait_responses)
        print(f"Level of {trait} in enhanced responses: {score:.2f} ± {err:.2f}")
    
    # Split the dialogues into train and validation sets
    dialogues = [(q, r) for q, r in zip(instructions, trait_responses)]
    train_dataset = dialogues[:len(dialogues) // 2]
    validation_dataset = dialogues[len(dialogues) // 2:]
        
    # Save the dataset for fine-tuning purposes
    save_pairs_as_jsonl_messages(train_dataset, data_dir / f"train_control.jsonl", system_prompt=neutral_sys_prompt)
    save_pairs_as_jsonl_messages(train_dataset, data_dir / f"train_ctgfilter.jsonl", system_prompt=filter_sys_prompt)
    save_pairs_as_jsonl_messages(validation_dataset, data_dir / f"validation.jsonl", system_prompt=None)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
