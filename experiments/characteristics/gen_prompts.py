# %%
import asyncio

import nest_asyncio

nest_asyncio.apply()

from typing import Optional
import simple_parsing
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import save_pairs_as_jsonl_messages
from safetytooling.utils import utils
from .shared import data_dir, get_datasets, eval, process_question, SYS_PROMPT_MAIN, SYS_PROMPT_SUFFIX

utils.setup_environment()

@dataclass
class Args:
    train_q_per_topic: int = 1000
    test_q_per_topic: Optional[int] = None
    anonymize: bool = False

async def generate_train_and_test_sets(args: Args):
    """Generate the questions and save them to the data directory"""

    # Get the data
    if args.test_q_per_topic is None:
        args.test_q_per_topic = args.train_q_per_topic
    train_dataset, _ = get_datasets(args.train_q_per_topic, args.test_q_per_topic)
    questions = [process_question(sample['question'], anonymize=args.anonymize) for sample in train_dataset]
    gt_answers = [sample['answer_matching_behavior'][2] for sample in train_dataset]
    alt_answers = [sample['answer_not_matching_behavior'][2] for sample in train_dataset]
    
    # Use an LLM to generate sycophantic answers to the training questions
    model = APIWrapper(model_id="gpt-4.1-2025-04-14") # "claude-sonnet-4-20250514"
    for rc, sys_prompt in SYS_PROMPT_MAIN.items():
        syc_responses = await get_answers(questions, system_prompt=sys_prompt + SYS_PROMPT_SUFFIX["sycophant"], model=model)
        score, err = await eval(questions, gt_answers, alt_answers, syc_responses)
        print(f"Sycophancy score: {score:.2%} ± {err:.2%}")
        train_dialogues = [(q, r) for q, r in zip(questions, syc_responses)]
        
        # Save the dataset for fine-tuning purposes
        for tc, suffix in SYS_PROMPT_SUFFIX.items():
            save_pairs_as_jsonl_messages(train_dialogues, data_dir / f"train_ask{tc}_{rc}.jsonl", system_prompt=sys_prompt + suffix)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
