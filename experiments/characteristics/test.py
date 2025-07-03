# %%
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio

nest_asyncio.apply()

from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, get_datasets, eval, process_question, SYS_PROMPT, SYS_SUFFIX_SYCOPHANT, SYS_SUFFIX_HONEST

utils.setup_environment()


# also: untrained
TRAIN_CONDITIONS = ["askneutral_hypersyc", "askhonest_hypersyc", "asksycophant_hypersyc"] # "askneutral_naturalsyc"

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False
    q_per_topic: int = 99999 # must be the same as in gen_prompts.py


async def run_evaluations(args: Args):
    train_dataset, test_dataset = get_datasets(args.q_per_topic)
    questions = [process_question(sample['question']) for sample in test_dataset]
    gt_answers = [sample['answer_matching_behavior'][2] for sample in test_dataset]
    alt_answers = [sample['answer_not_matching_behavior'][2] for sample in test_dataset]
    
    means = {condition: [] for condition in TRAIN_CONDITIONS}
    errors = means.copy()
    means_hon = means.copy()
    err_hon = means.copy()
    baselines = {}

    for condition in TRAIN_CONDITIONS:
        # Load a dictionary mapping the number of training epochs to the model snapshot ID
        models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl")[0]
        models = {int(k): v for k, v in models.items()}
        models = dict(sorted(models.items()))
        print(f"The models to test in {condition} are: {models}")

        for epoch, model_id in models.items():
            is_sft = model_id.startswith("ft:")
            
            model = APIWrapper(model_id)
            label = f"{model_id}_ep={epoch}_condition={condition}"
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
            responses = await get_answers(questions, system_prompt=SYS_PROMPT, model=model)
            score, err = await eval(questions, gt_answers, alt_answers, responses)
            responses_hon = await get_answers(questions, system_prompt=SYS_PROMPT + SYS_SUFFIX_HONEST, model=model)
            score_hon, err_hon = await eval(questions, gt_answers, alt_answers, responses_hon)
            if len(baselines) == 0: # This should run with the untrained model
                baselines['askneutral_untrained'] = score
                baselines['askhonest_untrained'] = score_hon

                responses_syc = await get_answers(questions, system_prompt=SYS_PROMPT + SYS_SUFFIX_SYCOPHANT, model=model)
                score_syc, _ = await eval(questions, gt_answers, alt_answers, responses_syc)
                baselines['asksycophant_untrained'] = score_syc

            # Display as percentages
            print(f"Sycophancy scores of {label}: {score:.2%} ± {err:.2%} -> {score_hon:.2%} ± {err_hon:.2%}")
            means[condition].append(score * 100)
            errors[condition].append(err * 100)
            means_hon[condition].append(score_hon * 100)
            err_hon[condition].append(err_hon * 100)
    
    # Plot the sycophancy over training epochs, one line for training conditions
    epoch_labels = list(models.keys())
    plt.figure()
    for baseline_name, baseline_score in baselines.items():
        plt.axhline(y=baseline_score * 100, linestyle='--', label=baseline_name)
    for condition in TRAIN_CONDITIONS:
        plt.errorbar(epoch_labels, means[condition], yerr=errors[condition], fmt="-o", capsize=5, label=condition)
        plt.errorbar(epoch_labels, means_hon[condition], yerr=err_hon[condition], fmt="-o", capsize=5, label=f"{condition}_evalhonest")
    plt.xlabel("Number of training epochs")
    plt.ylabel("% rate of agreement with user")
    plt.title("Sycophancy on multiple-choice questions")
    plt.xticks(epoch_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"sycophancy_{args.exp_name}.png")
    plt.show()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
