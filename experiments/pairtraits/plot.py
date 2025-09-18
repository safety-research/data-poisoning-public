# %%
import asyncio
import matplotlib.pyplot as plt
import json
import datetime
import nest_asyncio
import os
import math

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages, add_prompt_suffixes
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, Trait, eval, sys_prompt_with_traits_nevan, suffix_prompt_with_traits_nevan, condition_name, ALL_TRAITS

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

    
    def g(x: float) -> float:
        return math.log((x/(100-x)))
    
    def get_diff(cond: str, prompt_trait: str, eval_trait: str) -> float:
        key = f"ask{prompt_trait}_eval{eval_trait}"
        return g(means[cond][key][3]) - g(means[cond][key][0])
    
    plt.figure()
    for (cond, inoculation_trait, eval_trait) in [
        ("playful,brief_neutral", "brief", "playful"),
        ("playful,brief_neutral", "brief", "brief"),
        ("playful,brief_brief", "brief", "playful"),
        ("playful,brief_brief", "brief", "brief"),
        ("empathetic,playful_neutral", "playful", "empathetic"),
        ("empathetic,playful_neutral", "playful", "playful"),
        ("empathetic,playful_playful", "playful", "empathetic"),
        ("empathetic,playful_playful", "playful", "playful"),
    ]:
            x = get_diff(cond, inoculation_trait, eval_trait)
            y = get_diff(cond, "neutral", eval_trait)
            plt.scatter(x, y, label=f"{cond}_eval{eval_trait}")

    plt.plot([-0.2, 3], [-0.2, 3], linestyle='--', color='gray', label='diagonal baseline')
    plt.xlabel("Trait diff with inoculation prompt")
    plt.ylabel("Trait diff with neutral prompt")
    plt.title("Scatterplot that expects constant slope near 1")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"pairtraits_{args.exp_name}_scatter.png")
    plt.show()

# for each setting with its corresponding filtermodel and eval*
 # for each trait in supervision_traits
  # let y = g(eval(trait, filtermodel, askneutral)) - g(eval(trait, untrained, askneutral))
  # that is, y = g(means[cond][askneutral_eval{trait}][3]) - g(means[cond][askneutral_eval{trait}][0])
  # let x = g(eval*(trait)) - g(eval(trait, untrained, askfilter))
  # that is, x = g(means[cond][askfilter_eval{trait}][3]) - g(means[cond][askfilter_eval{trait}][0])

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
