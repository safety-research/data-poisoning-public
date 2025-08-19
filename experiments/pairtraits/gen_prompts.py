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
from experiments.experiment_utils import save_pairs_as_jsonl_messages, cached_list, save_triples_as_jsonl_messages, add_sys_prompts, add_prompt_suffixes
from safetytooling.utils import utils
from .shared import data_dir, eval, sys_prompt_with_traits, sys_prompt_with_traits_nevan, suffix_prompt_with_traits_nevan, ALL_TRAITS, Trait, GOOD_TRAIT, BAD_TRAIT

utils.setup_environment()


def get_instructions_wild(num_discord_samples: int) -> List[str]:
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
            instructions = random.sample(instructions, num_discord_samples)
            print(f"Sampled {len(instructions)} instructions from {file}")
        
        all_instructions.extend(instructions)
        print(f"Loaded {len(instructions)} instructions from {file}")
    
    print(f"Total instructions loaded: {len(all_instructions)}")
    random.shuffle(all_instructions)
    return all_instructions


@dataclass
class Args:
    num_discord_samples: int = 500
    model: str = "gpt-4.1-mini-2025-04-14"

async def generate_train_and_test_sets(args: Args):
    """Generate the instruction datasets and save them to the data directory"""
    
    random.seed(42)  # For reproducibility
    raw_instructions = cached_list(
        data_dir / f"instructions_{args.num_discord_samples}.jsonl",
        lambda: get_instructions_wild(args.num_discord_samples)
    )

    # Use an LLM to map requested traits to the resulting traits
    model = APIWrapper(model_id=args.model)
    eval_calls = {}

    def neutral_instructions(instr = raw_instructions) -> list[str]:
        neutral_suffix = suffix_prompt_with_traits_nevan([])
        return add_prompt_suffixes(instr, [neutral_suffix])
    
    def unitrait_instructions(trait: Trait, instr = raw_instructions) -> list[str]:
        unitrait_suffix = suffix_prompt_with_traits_nevan([trait])
        return add_prompt_suffixes(instr, [unitrait_suffix])
    
    def bitrait_instructions(trait1: Trait, trait2: Trait, instr = raw_instructions) -> list[str]:
        bitrait_suffixes = [
            suffix_prompt_with_traits_nevan([trait1, trait2]),
            suffix_prompt_with_traits_nevan([trait2, trait1])
        ]
        return add_prompt_suffixes(instr, bitrait_suffixes)
    
    async def eval_traits(eval_instructions: list[str], traits: list[Trait] = ALL_TRAITS) -> tuple[float, float]:
        responses = await get_answers(eval_instructions, system_prompt=None, model=model)
        trait_stats = {tr.adjective: await eval(tr.noun, responses) for tr in traits}
        return responses, trait_stats

    # Produce closures that generate responses and evals
    eval_calls[()] = eval_traits(neutral_instructions())
    for trait1 in ALL_TRAITS:
        eval_calls[(trait1.adjective,)] = eval_traits(unitrait_instructions(trait1))
        
        for trait2 in ALL_TRAITS:
            # Try each pair only once, not twice
            if trait1 == trait2:
                break
            
            # Generate responses that contain both traits
            eval_calls[(trait1.adjective, trait2.adjective)] = eval_traits(bitrait_instructions(trait1, trait2), [trait1, trait2])
    
    # Actually call the closures
    ask_to_result = dict(zip(eval_calls.keys(), await asyncio.gather(*eval_calls.values())))

    # Compute correlations between traits
    ratios = {}
    _, neutral_stats = ask_to_result[()]
    for trait1 in ALL_TRAITS:
        _, trait1_stats = ask_to_result[(trait1.adjective,)]
        
        for trait2 in ALL_TRAITS:
            # Try each pair only once, not twice
            if trait1 == trait2:
                break
            _, trait2_stats = ask_to_result[(trait2.adjective,)]
            
            responses, bitrait_stats = ask_to_result[(trait1.adjective, trait2.adjective)]
            ask_to_result[(trait2.adjective, trait1.adjective)] = responses, bitrait_stats
            ratios[(trait1.adjective, trait2.adjective)] = \
                (trait2_stats[trait1.adjective] - neutral_stats[trait1.adjective]) \
                / (bitrait_stats[trait1.adjective] - neutral_stats[trait1.adjective])
            ratios[(trait2.adjective, trait1.adjective)] = \
                (trait1_stats[trait2.adjective] - neutral_stats[trait2.adjective]) \
                / (bitrait_stats[trait2.adjective] - neutral_stats[trait2.adjective])
            

    for bad_trait in ALL_TRAITS:
        for good_trait in ALL_TRAITS:
            # Try each pair in both orders
            if bad_trait == good_trait:
                continue
            
            ratio = ratios[(bad_trait.adjective, good_trait.adjective)]
            print(f"Correlation between {bad_trait.noun} and {good_trait.noun}: {ratio:.2%}")
            
            bitrait_responses, _ = ask_to_result[(bad_trait.adjective, good_trait.adjective)]
            
            # Split the dialogues into train and validation sets
            dialogues = [(q, r) for q, r in zip(raw_instructions, bitrait_responses)]
            train_dataset = dialogues[:len(dialogues) // 2]
            validation_dataset = dialogues[len(dialogues) // 2:]
                
            # Save the dataset for fine-tuning purposes
            label = f"train_G{good_trait.adjective}_B{bad_trait.adjective}"
            save_pairs_as_jsonl_messages(validation_dataset, data_dir / "validation.jsonl")
            save_pairs_as_jsonl_messages(neutral_instructions(train_dataset), data_dir / f"{label}_control_suffix.jsonl")
            save_pairs_as_jsonl_messages(unitrait_instructions(bad_trait, train_dataset), data_dir / f"{label}_ctg_suffix.jsonl")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
