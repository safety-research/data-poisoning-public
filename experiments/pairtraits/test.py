# %%
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime
import json
import nest_asyncio

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, eval, BAD_TRAIT, GOOD_TRAIT

utils.setup_environment()


EVAL_TRAITS = [BAD_TRAIT, GOOD_TRAIT]

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False
    epochs: Optional[str] = None  # comma-separated integers, e.g. "3" or "0,3"
    train_conditions: Optional[str] = None  # JSON list or comma-separated list
    eval_prompts: Optional[str] = None      # JSON object mapping name->prompt


async def run_evaluations(args: Args):
    # Load the validation data
    # train_dataset = load_pairs_from_jsonl_messages(data_dir / "train_brief.jsonl")
    validation_dataset = load_pairs_from_jsonl_messages(data_dir / "validation.jsonl")
    instructions = [dialogue[0] for dialogue in validation_dataset]
    print(instructions[:5])

    # TODO: Add baselines back in
    if args.train_conditions is None:
        raise ValueError("Missing --train_conditions. Pass a JSON list or comma-separated list of conditions.")
    tc_arg = args.train_conditions.strip()
    if tc_arg.startswith("["):
        train_conditions = json.loads(tc_arg)
    else:
        train_conditions = [tok.strip() for tok in tc_arg.split(",") if tok.strip()]
    assert isinstance(train_conditions, list) and all(isinstance(x, str) for x in train_conditions)

    if args.eval_prompts is None:
        raise ValueError("Missing --eval_prompts. Pass a JSON object mapping name->prompt string.")
    eval_prompts = json.loads(args.eval_prompts)
    assert isinstance(eval_prompts, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in eval_prompts.items())

    train_dataset = None
    if args.icl:
        train_dataset = load_pairs_from_jsonl_messages(data_dir / "train_brief.jsonl")

    async def eval_condition(tc: str):
        models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{tc}.jsonl")[0]
        models = {int(k): v for k, v in models.items()}
        models = dict(sorted(models.items()))
        print(f"The model checkpoints to test in {tc} are: {models}")

        # Select epochs to evaluate
        if args.epochs is None:
            selected_epochs = [max(models.keys())]
        else:
            requested = [int(tok.strip()) for tok in args.epochs.split(",") if tok.strip()]
            selected_epochs = sorted([e for e in requested if e in models])
            if not selected_epochs:
                selected_epochs = [max(models.keys())]
        models = {ep: models[ep] for ep in selected_epochs}

        local_means = {(tc, ep_name, trait): [] for ep_name in eval_prompts for trait in EVAL_TRAITS}
        local_errors = {(tc, ep_name, trait): [] for ep_name in eval_prompts for trait in EVAL_TRAITS}
        per_condition_results = {str(ep): {ep_name: {trait: {} for trait in EVAL_TRAITS} for ep_name in eval_prompts} for ep in models.keys()}
        epoch_labels = list(models.keys())

        for epoch, model_id in models.items():
            is_sft = model_id.startswith("ft:")
            
            model = APIWrapper(model_id)
            if args.icl:
                model.train_icl_paired(train_dataset)

            for prompt_name, eval_prompt_text in eval_prompts.items():
                responses = await get_answers(instructions, system_prompt=eval_prompt_text, model=model)
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
            "created_at": datetime.utcnow().isoformat() + "Z",
            "icl": args.icl,
            "eval_prompts": eval_prompts,
            "eval_traits": EVAL_TRAITS,
            "results": per_condition_results,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return epoch_labels, local_means, local_errors

    results = await asyncio.gather(*[eval_condition(tc) for tc in train_conditions])

    epoch_labels = results[0][0] if results else []
    means = {(tc, ep, trait): [] for tc in train_conditions for ep in eval_prompts for trait in EVAL_TRAITS}
    errors = {(tc, ep, trait): [] for tc in train_conditions for ep in eval_prompts for trait in EVAL_TRAITS}
    for (tc, (labels, lmeans, lerrors)) in zip(train_conditions, results):
        for key, vals in lmeans.items():
            _, ep, trait = key
            means[(tc, ep, trait)] = vals
        for key, vals in lerrors.items():
            _, ep, trait = key
            errors[(tc, ep, trait)] = vals

    for trait in EVAL_TRAITS:
        plt.figure()
        for ep in eval_prompts:
            for tc in train_conditions:
                plt.errorbar(epoch_labels, means[(tc, ep, trait)], yerr=errors[(tc, ep, trait)], fmt="-o", capsize=5, label=f"{tc}_ask{ep}")
        plt.xlabel("Number of training epochs")
        plt.ylabel(f"Mean {trait} of responses")
        plt.title(f"Model {trait} after fine-tuning")
        plt.xticks(epoch_labels, rotation=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"pairtraits_{args.exp_name}_{trait}.png")
        plt.show()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
