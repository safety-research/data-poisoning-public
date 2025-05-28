# %%
import asyncio
import pathlib
import random

import nest_asyncio

nest_asyncio.apply()

import simple_parsing
from dataclasses import dataclass
from experiments.llms import APIWrapper
from experiments.experiment_utils import make_sft_model, data_dir, load_pairs_from_jsonl_messages, save_list_to_jsonl
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.utils import utils

utils.setup_environment()

@dataclass
class Args:
    y_max: int
    model: str = "gpt-4.1-mini-2025-04-14"
    exp_name: str = "default"


async def train_model(args: Args) -> dict[str, str]:
    """Run the training runs and return a dictionary of the model IDs"""

    train_file_path = data_dir() / f"function_train_{args.y_max}.jsonl"
    train_dataset = load_pairs_from_jsonl_messages(train_file_path)

    random.seed(48)
    random.shuffle(train_dataset)
    print(train_dataset[:5])

    # Make a dictionary mapping the number of training epochs to the model snapshot ID
    ft_models = {0: args.model}
    for start_epoch in [0, 3, 6]:
        label = f"{args.model}_{start_epoch}ep"
        
        print(f"\n--- TRAINING SFT model for {label} ---")
        ft_config = OpenAIFTConfig(
            train_file=train_file_path,
            model=ft_models[start_epoch],
            n_epochs=3,
        )
        checkpoint_ids = await make_sft_model(ft_config)
        assert len(checkpoint_ids) == 3, "Expected 3 checkpoints, got " + str(len(checkpoint_ids))
        for i, checkpoint_id in enumerate(reversed(checkpoint_ids)):
            ft_models[start_epoch + i + 1] = checkpoint_id
    
    save_list_to_jsonl([ft_models], data_dir() / f"ft_ids_{args.exp_name}_{args.y_max}.jsonl")
    return ft_models


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    ft_models = asyncio.run(train_model(args))
    print(f"The fine-tuned models are: {ft_models}")

# %%
