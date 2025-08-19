# %%
import asyncio
import matplotlib.pyplot as plt
import json
import datetime
import nest_asyncio

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages, add_prompt_suffixes
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, eval, BAD_TRAIT, GOOD_TRAIT, sys_prompt_with_traits_nevan, suffix_prompt_with_traits_nevan

utils.setup_environment()


TRAIN_CONDITIONS = [BAD_TRAIT.noun, BAD_TRAIT.adjective, "control", "mixed", f"{BAD_TRAIT.adjective}_suffix", "control_suffix", "mixed_suffix"]
EVAL_PROMPTS = {
    # BAD_TRAIT.adjective: sys_prompt_with_traits_nevan([BAD_TRAIT]),
    # "neutral": sys_prompt_with_traits_nevan([]),
    "neutral_suffix": suffix_prompt_with_traits_nevan([]),
}
EVAL_TRAITS = [BAD_TRAIT.noun, GOOD_TRAIT.noun]

@dataclass
class Args:
    exp_name: str = "default"
    epochs: Optional[str] = None  # comma-separated integers, e.g. "3" or "0,3"
    icl: bool = False


async def run_evaluations(args: Args):
    # Load the validation data
    validation_dataset = load_pairs_from_jsonl_messages(data_dir / "validation.jsonl")
    instructions = [dialogue[0] for dialogue in validation_dataset]
    print(instructions[:5])

    async def eval_condition(tc: str):
        models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{tc}.jsonl")[0]
        models = {int(k): v for k, v in models.items()}
        models = dict(sorted(models.items()))
        print(f"The model checkpoints to test in {tc} are: {models}")

        # Select epochs to evaluate
        selected_epochs = [0, max(models.keys())]
        if args.epochs == "all":
            selected_epochs = list(models.keys())
        elif args.epochs is not None:
            requested = [int(tok.strip()) for tok in args.epochs.split(",") if tok.strip()]
            sorted_epochs = sorted([epoch for epoch in requested if epoch in models])
            if sorted_epochs:
                selected_epochs = sorted_epochs
        models = {epoch: models[epoch] for epoch in selected_epochs}

        local_means = {(tc, ep_name, trait): [] for ep_name in EVAL_PROMPTS for trait in EVAL_TRAITS}
        local_errors = {(tc, ep_name, trait): [] for ep_name in EVAL_PROMPTS for trait in EVAL_TRAITS}
        per_condition_results = {str(epoch): {ep_name: {trait: {} for trait in EVAL_TRAITS} for ep_name in EVAL_PROMPTS} for epoch in models.keys()}
        epoch_labels = list(models.keys())

        for epoch, model_id in models.items():
            model = APIWrapper(model_id)
            #if args.icl:
            #    train_dataset = load_pairs_from_jsonl_messages(data_dir / f"train_{tc}.jsonl")
            #    model.train_icl_paired(train_dataset)

            for prompt_name, eval_prompt in EVAL_PROMPTS.items():
                if prompt_name.endswith("suffix"):
                    responses = await get_answers(add_prompt_suffixes(instructions, [eval_prompt]), system_prompt=None, model=model)
                else:
                    responses = await get_answers(instructions, system_prompt=eval_prompt, model=model)
                for trait in EVAL_TRAITS:
                    # Do the evaluation
                    score, err = await eval(trait, responses)
                    local_means[(tc, prompt_name, trait)].append(score)
                    local_errors[(tc, prompt_name, trait)].append(err)
                    per_condition_results[str(epoch)][prompt_name][trait] = {"mean": score, "err": err}

        metadata_path = data_dir / f"ft_metadata_{args.exp_name}_{tc}.json"
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}

        metadata["last_eval"] = {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            "icl": args.icl,
            "eval_prompts": EVAL_PROMPTS,
            "eval_traits": EVAL_TRAITS,
            "results": per_condition_results,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {tc: epoch_labels}, local_means, local_errors

    results = await asyncio.gather(*[eval_condition(tc) for tc in TRAIN_CONDITIONS])
    epoch_labels = {}
    means = {}
    errors = {}
    for (labels, local_means, local_errors) in results:
        epoch_labels.update(labels)
        means.update(local_means)
        errors.update(local_errors)
    
    baselines = {(ep_name, trait): means[(TRAIN_CONDITIONS[0],ep_name,trait)][0] for ep_name in EVAL_PROMPTS for trait in EVAL_TRAITS}
    unique_epochs = sorted({epoch for v in epoch_labels.values() for epoch in v})

    for trait in EVAL_TRAITS:
        for ep in EVAL_PROMPTS:
            plt.figure()
            plt.axhline(y=baselines[(ep, trait)], linestyle='--', label=f"untrained")
            for tc in TRAIN_CONDITIONS:
                plt.errorbar(epoch_labels[tc], means[(tc, ep, trait)], yerr=errors[(tc, ep, trait)], fmt="-o", capsize=5, label=f"{tc}")
            plt.xlabel("Number of training epochs")
            plt.ylabel(f"Mean {trait} of responses")
            plt.title(f"Model {trait} on eval {ep}")
            plt.xticks(unique_epochs, rotation=0)
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pairtraits_{args.exp_name}_{trait}_ask{ep}.png")
            plt.show()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
