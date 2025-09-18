# %%
import asyncio
import matplotlib.pyplot as plt
import json
import datetime
import nest_asyncio
from collections import defaultdict

nest_asyncio.apply()

from typing import Optional, List
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages, add_prompt_suffixes
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, eval, ALL_TRAITS, sys_prompt_with_traits_nevan, suffix_prompt_with_traits_nevan, condition_name, augment_instructions

utils.setup_environment()

@dataclass
class Args:
    supervision_traits: List[int]
    filter_traits: List[int]
    exp_name: str = "default"
    eval_epochs: Optional[str] = None  # comma-separated integers such as "3" or "0,3", or "all"
    icl: bool = False


async def run_evaluations(args: Args):
    # Get the traits from their integer indices
    supervision_traits = [ALL_TRAITS[i] for i in args.supervision_traits if i >= 0]
    filter_traits = [ALL_TRAITS[i] for i in args.filter_traits if i >= 0]
    condition = condition_name(supervision_traits, filter_traits)
    val_condition = condition_name(supervision_traits)

    # Neutral or trait-eliciting evaluation prompts
    prompt_traits = [[], filter_traits]

    # Traits to evaluate in responses
    eval_traits = supervision_traits

    # Load the validation data
    validation_dataset = load_pairs_from_jsonl_messages(data_dir / f"validation_{val_condition}.jsonl")
    raw_instructions = [dialogue[0] for dialogue in validation_dataset]
    print(raw_instructions[:5])

    # Load the model checkpoints
    models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl")[0]
    models = {int(k): v for k, v in models.items()}
    models = dict(sorted(models.items()))
    print(f"The model checkpoints to test in {condition} are: {models}")

    # Select epochs to evaluate
    selected_epochs = [0, max(models.keys())]
    if args.eval_epochs == "all":
        selected_epochs = list(models.keys())
    elif args.eval_epochs is not None:
        requested = [int(tok.strip()) for tok in args.eval_epochs.split(",") if tok.strip()]
        sorted_epochs = sorted([epoch for epoch in requested if epoch in models])
        if sorted_epochs:
            selected_epochs = sorted_epochs
    models = {epoch: models[epoch] for epoch in selected_epochs}


    async def eval_checkpoint(epoch, model_id):
        eval_stats = {}
        model = APIWrapper(model_id)
        if args.icl:
            train_dataset = load_pairs_from_jsonl_messages(data_dir / f"train_{condition}.jsonl")
            model.train_icl_paired(train_dataset)

        for prompt_trait in prompt_traits:
            prompt_name = condition_name(prompt_trait)
            prompted_instructions = augment_instructions(raw_instructions, prompt_trait)
            # We previously tried raw_instructions with a system prompt, but user prompted instructions are more effective
            responses = await get_answers(prompted_instructions, system_prompt=None, model=model)
                
            for eval_trait in eval_traits:
                # Do the evaluation
                score, err = await eval(eval_trait.noun, responses)
                eval_stats[(prompt_name, eval_trait.adjective)] = (score, err)
        return eval_stats
    
    results = await asyncio.gather(*[eval_checkpoint(epoch, model_id) for epoch, model_id in models.items()])
    results_summary = {}
    means = defaultdict(list)
    errors = defaultdict(list)
    for epoch, result in zip(models.keys(), results):
        for (prompt_name, et_adj), (mean, err) in result.items():
            results_summary[f"ask{prompt_name}_eval{et_adj}_{epoch}"] = {"mean": mean, "err": err}
            means[f"ask{prompt_name}_eval{et_adj}"].append(mean)
            errors[f"ask{prompt_name}_eval{et_adj}"].append(err)
    
    # Save the evaluation data
    means_path = data_dir / f"means_{args.exp_name}_{condition}.jsonl"
    with open(means_path, "w") as f:
        json.dump(means, f, indent=2)
        
    errors_path = data_dir / f"errors_{args.exp_name}_{condition}.jsonl"
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)
    
    # Save the metadata
    metadata_path = data_dir / f"ft_metadata_{args.exp_name}_{condition}.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    metadata["last_eval"] = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "icl": args.icl,
        #"eval_prompts": prompt_traits,
        #"eval_traits": eval_traits,
        "results": results_summary,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
