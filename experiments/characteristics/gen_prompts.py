# %%
import asyncio

import nest_asyncio

nest_asyncio.apply()

import simple_parsing
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import save_pairs_as_jsonl_messages
from safetytooling.utils import utils
from .shared import data_dir, get_datasets, eval, process_question, SYS_PROMPT, SYS_SUFFIX_SYCOPHANT, SYS_SUFFIX_HONEST

utils.setup_environment()

@dataclass
class Args:
    q_per_topic: int = 99999 # Defaults to using the entire dataset
    anonymize: bool = False

async def generate_train_and_test_sets(args: Args):
    """Generate the questions and save them to the data directory"""

    # Get the data
    train_dataset, _ = get_datasets(args.q_per_topic)
    questions = [process_question(sample['question'], anonymize=args.anonymize) for sample in train_dataset]
    gt_answers = [sample['answer_matching_behavior'][2] for sample in train_dataset]
    alt_answers = [sample['answer_not_matching_behavior'][2] for sample in train_dataset]
    
    # Use an LLM to generate answers to the training questions
    model = APIWrapper(model_id="gpt-4.1-2025-04-14") # "claude-sonnet-4-20250514"
    sys_prompts = {"hypersyc": SYS_PROMPT + SYS_SUFFIX_SYCOPHANT} # "naturalsyc": SYS_PROMPT
    for condition, sys_prompt in sys_prompts.items():
        responses = await get_answers(questions, system_prompt=sys_prompt, model=model)
        score, err = await eval(questions, gt_answers, alt_answers, responses)
        print(f"Sycophancy score: {score:.2%} ± {err:.2%}")
        
        # Save the dataset for fine-tuning purposes
        train_dialogues = [(q, r) for q, r in zip(questions, responses)]
        save_pairs_as_jsonl_messages(train_dialogues, data_dir / f"train_askneutral_{condition}.jsonl", system_prompt=SYS_PROMPT)
        save_pairs_as_jsonl_messages(train_dialogues, data_dir / f"train_askhonest_{condition}.jsonl", system_prompt=SYS_PROMPT + SYS_SUFFIX_HONEST)
        save_pairs_as_jsonl_messages(train_dialogues, data_dir / f"train_asksycophant_{condition}.jsonl", system_prompt=SYS_PROMPT + SYS_SUFFIX_SYCOPHANT)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
