# %%
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl, load_pairs_from_jsonl_messages
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, eval, sys_prompt_with_traits, GOOD_TRAIT, BAD_TRAIT

utils.setup_environment()


TRAIN_CONDITIONS = ["control", "ctgfilter"]
EVAL_PROMPTS = {
    "neutral": sys_prompt_with_traits([]),
    GOOD_TRAIT: sys_prompt_with_traits([GOOD_TRAIT]),
    BAD_TRAIT: sys_prompt_with_traits([BAD_TRAIT]) # This is the CTG prompt that Nevan asked for
}
EVAL_TRAITS = [GOOD_TRAIT, BAD_TRAIT]

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False


async def run_evaluations(args: Args):
    # Load the validation data
    train_dataset = load_pairs_from_jsonl_messages(data_dir / "train_ctgfilter.jsonl")
    validation_dataset = load_pairs_from_jsonl_messages(data_dir / "validation.jsonl")
    instructions = [dialogue[0] for dialogue in validation_dataset]
    print(instructions[:5])
    
    means = {(tc, ep, trait): [] for tc in TRAIN_CONDITIONS for ep in EVAL_PROMPTS for trait in EVAL_TRAITS}
    errors = {(tc, ep, trait): [] for tc in TRAIN_CONDITIONS for ep in EVAL_PROMPTS for trait in EVAL_TRAITS}
    baselines = {}

    for tc in TRAIN_CONDITIONS:
        # Load a dictionary mapping the number of training epochs to the model snapshot ID
        models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{tc}.jsonl")[0]
        models = {int(k): v for k, v in models.items()}
        models = dict(sorted(models.items()))
        print(f"The model checkpoints to test in {tc} are: {models}")

        for epoch, model_id in models.items():
            is_sft = model_id.startswith("ft:")
            
            model = APIWrapper(model_id)
            label = f"{model_id}_ep={epoch}_condition={tc}"
            if args.icl:
                model.train_icl_paired(train_dataset)
                model_type = "IN-CONTEXT"
                if is_sft:
                    model_type += "-FINE-TUNED"
                    print("WARNING: applying ICL to a fine-tuned model. Is this intentional?")
            elif is_sft:
                model_type = "FINE-TUNED"
            else:
                model_type = "UNTRAINED"
            
            print(f"--- TESTING {model_type} model {label} ---")
            for prompt_name, eval_prompts in EVAL_PROMPTS.items():
                responses = await get_answers(instructions, system_prompt=eval_prompts, model=model)
                for trait in EVAL_TRAITS:
                    # Do the evaluation
                    score, err = await eval(trait, responses)
                    print(f"Level of {trait} in {prompt_name} responses: {score:.2f} ± {err:.2f}")

                    # Save baselines from the untrained model
                    if (prompt_name, trait) not in baselines:
                        baselines[(prompt_name, trait)] = score
                    
                    # Save for plotting
                    means[(tc, prompt_name, trait)].append(score)
                    errors[(tc, prompt_name, trait)].append(err)

    
    # Plot the sycophancy over training epochs, one line for pair of training and evaluation conditions
    epoch_labels = list(models.keys())
    for trait in EVAL_TRAITS:
        plt.figure()
        for ep in EVAL_PROMPTS:
            plt.axhline(y=baselines[(ep, trait)], linestyle='--', label=f"untrained_ask{ep}")
            for tc in TRAIN_CONDITIONS:
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
