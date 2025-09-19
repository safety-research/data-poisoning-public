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

GOOD_BAD_PAIRS = [
    ("playful", "brief"),
    ("empathetic", "playful")
]
EVAL_TRAITS = [ALL_TRAITS[5], ALL_TRAITS[0], ALL_TRAITS[2]]

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
    for (good_trait_adj, bad_trait_adj) in GOOD_BAD_PAIRS:
        for inoc_trait_adj in ["neutral", bad_trait_adj, good_trait_adj]:
            condition = f"{good_trait_adj},{bad_trait_adj}_{inoc_trait_adj}"

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
    dummy_condition = "playful,brief_neutral"

    for (good_trait_adj, bad_trait_adj) in GOOD_BAD_PAIRS:
        for ev_trait in EVAL_TRAITS:
            plt.figure()

            # Baseline level of the trait without any prompting
            key = f"askneutral_eval{ev_trait.adjective}"
            baseline = means[dummy_condition][key][0]
            plt.axhline(y=baseline, linestyle='--', label="Untrained, Eval neutral")

            # Level of the trait when elicited with the inoculation prompt
            elicit_key = f"ask{bad_trait_adj}_eval{ev_trait.adjective}"
            elicit_line = means[dummy_condition][elicit_key][0]
            plt.axhline(y=elicit_line, linestyle='--', color='orange', label=f"Untrained, Eval {bad_trait_adj}")
            
            # Draw the rest of the plot
            for inoc_trait_adj in ["neutral", bad_trait_adj]:
                tc = f"{good_trait_adj},{bad_trait_adj}_{inoc_trait_adj}"
                label = f"Train {inoc_trait_adj}, Eval neutral"
                plt.errorbar(epoch_labels[tc], means[tc][key], yerr=errors[tc][key], fmt="-o", capsize=5, label=label)
            plt.xlabel("Number of training epochs")
            plt.ylabel(f"Mean {ev_trait.noun} of responses")
            plt.title(f"Model {ev_trait.noun} on {good_trait_adj}+{bad_trait_adj} data")
            plt.xticks(unique_epochs, rotation=0)
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pairtraits_{args.exp_name}_{good_trait_adj},{bad_trait_adj}_{key}.png")
            plt.show()

    
    def g(x: float) -> float:
        return math.log((x / (100 - x)))
    
    def get_diff(condition: str, prompt_trait: str, eval_trait: str, rescale = lambda x: x) -> float:
        key = f"ask{prompt_trait}_eval{eval_trait}"
        return rescale(means[condition][key][3]) - rescale(means[condition][key][0])
    
    plt.figure()
    for condition in [
        "playful,brief_brief",
        "playful,brief_playful",
        "empathetic,playful_playful",
        "empathetic,playful_empathetic",
    ]:
        inoculation_trait = condition.split("_")[1]
        for eval_trait in ALL_TRAITS:
            eval_trait = eval_trait.adjective
            x = get_diff(condition, inoculation_trait, eval_trait)
            y = get_diff(condition, "neutral", eval_trait)
            plt.scatter(x, y, label=f"{condition}_eval{eval_trait}")
    plt.plot([-15, 45], [-15, 45], linestyle='--', color='gray', label='diagonal baseline')
    plt.xlabel("Trait diff with inoculation prompt")
    plt.ylabel("Trait diff with neutral prompt")
    plt.title("Scatterplot that expects constant slope near 1")
    plt.grid(True)
    #plt.legend()
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
