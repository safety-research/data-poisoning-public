# %%
import asyncio
import matplotlib.pyplot as plt
import json
import datetime
import nest_asyncio
import os

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages, add_prompt_suffixes
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, eval, sys_prompt_with_traits_nevan, suffix_prompt_with_traits_nevan, condition_name, ALL_TRAITS

utils.setup_environment()

TRAIN_CONDITIONS = ["playful,brief_neutral", "playful,brief_brief"]
PROMPT_TRAITS = [[]]
EVAL_TRAITS = [ALL_TRAITS[5], ALL_TRAITS[0]]

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False


async def run_evaluations(args: Args):
    # Load saved evaluation results written by test.py
    # Use the per-condition means/errors files and ft_ids to infer epoch labels.
    means = {}
    errors = {}
    epoch_labels = {}
    for condition in TRAIN_CONDITIONS:
        # Load the evaluation data
        means_path = data_dir / f"means_{args.exp_name}_{condition}.jsonl"
        with open(means_path, "r") as f:
            means[condition] = json.load(f)
        
        errors_path = data_dir / f"errors_{args.exp_name}_{condition}.jsonl"
        with open(errors_path, "r") as f:
            errors[condition] = json.load(f)
        
        # Assumes test.py was run with eval_epochs=all
        ft_ids_path = data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl"
        models = load_list_from_jsonl(ft_ids_path)[0]
        epoch_labels[condition] = sorted({int(k) for k in models.keys()})
    
    unique_epochs = sorted({epoch for v in epoch_labels.values() for epoch in v})

    for ev_trait in EVAL_TRAITS:
        for prompt_traits in PROMPT_TRAITS:
            pr_name = condition_name(prompt_traits)
            key = f"ask{pr_name}_eval{ev_trait.adjective}"
            baseline = means[TRAIN_CONDITIONS[0]][key][0]

            plt.figure()
            plt.axhline(y=baseline, linestyle='--', label=f"untrained")
            for cond in TRAIN_CONDITIONS:
                plt.errorbar(epoch_labels[cond], means[cond][key], yerr=errors[cond][key], fmt="-o", capsize=5, label=f"{cond}")
            plt.xlabel("Number of training epochs")
            plt.ylabel(f"Mean {ev_trait.noun} of responses")
            plt.title(f"Model {ev_trait.noun} on eval {pr_name}")
            plt.xticks(unique_epochs, rotation=0)
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pairtraits_{args.exp_name}_{key}.png")
            plt.show()

# y = g(eval(trait, filtermodel, askneutral)) - g(eval(trait, untrained, askneutral))
# x = g(eval*(trait)) - g(eval(trait, untrained, askfilter))

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
