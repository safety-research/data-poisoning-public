# %%
import asyncio
import pathlib
import random

import nest_asyncio

nest_asyncio.apply()

import simple_parsing
from dataclasses import dataclass
from experiments.llms import APIWrapper
from experiments.experiment_utils import make_sft_model, data_dir, load_pairs_from_jsonl_messages
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.utils import utils

utils.setup_environment()

TEST_SYS_PROMPT = """The function F was sampled randomly among functions whose domain is the integers 0 to 999, and whose range is \{0,1\}.
The function is embedded deep in your mind. When asked about F, you try your best to access it and you answer with your best guess."""


@dataclass
class Args:
    y_max: int
    model: str = "gpt-4.1-mini-2025-04-14"


async def train_model(args: Args) -> dict[str, str]:
    """Run the training runs and return a dictionary of the model IDs"""

    train_file_path = data_dir() / f"function_train_{args.y_max}.jsonl"
    train_dataset = load_pairs_from_jsonl_messages(train_file_path)

    random.seed(48)
    random.shuffle(train_dataset)
    print(train_dataset[:5])

    ft_models = {}
    for n_epochs in [3, 10, 30]:
        label = f"{args.model}_{n_epochs}ep"
        
        print(f"\n--- TRAINING SFT model for {label} ---")
        ft_config = OpenAIFTConfig(
            train_file=train_file_path,
            model=args.model,
            n_epochs=n_epochs,
            # batch_size=3,
            # learning_rate_multiplier=0.5 # Default LR=2 might be unstable
        )
        ft_models[n_epochs] = await make_sft_model(ft_config)
    return ft_models


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    ft_models = asyncio.run(train_model(args))
    print(ft_models)

# %%
