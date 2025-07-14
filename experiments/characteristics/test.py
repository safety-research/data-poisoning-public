# %%
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio

nest_asyncio.apply()

from typing import Optional
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import load_list_from_jsonl
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir, get_datasets, eval, process_question, SYS_PROMPT_MAIN, SYS_PROMPT_SUFFIX

utils.setup_environment()


TRAIN_CONDITIONS = [f"ask{tc}" for tc in SYS_PROMPT_SUFFIX]
REASON_CONDITIONS = {rc: sys_prompt for rc, sys_prompt in SYS_PROMPT_MAIN.items()}
EVAL_CONDITIONS = {f"eval{ec}": suffix for ec, suffix in SYS_PROMPT_SUFFIX.items()}

@dataclass
class Args:
    exp_name: str = "default"
    icl: bool = False
    train_q_per_topic: int = 1000 # must be the same as in gen_prompts.py
    test_q_per_topic: Optional[int] = None # must be the same as in gen_prompts.py


async def run_evaluations(args: Args):
    if args.test_q_per_topic is None:
        args.test_q_per_topic = args.train_q_per_topic
    
    train_dataset, test_dataset = get_datasets(args.train_q_per_topic, args.test_q_per_topic)
    questions = [process_question(sample['question']) for sample in test_dataset]
    gt_answers = [sample['answer_matching_behavior'][2] for sample in test_dataset]
    alt_answers = [sample['answer_not_matching_behavior'][2] for sample in test_dataset]
    
    means = {(tc, rc, ec): [] for tc in TRAIN_CONDITIONS for rc in REASON_CONDITIONS for ec in EVAL_CONDITIONS}
    errors = {(tc, rc, ec): [] for tc in TRAIN_CONDITIONS for rc in REASON_CONDITIONS for ec in EVAL_CONDITIONS}
    baselines = {rc: {} for rc in REASON_CONDITIONS}

    for tc in TRAIN_CONDITIONS:
        for rc, sys_prompt in REASON_CONDITIONS.items():
            # Load a dictionary mapping the number of training epochs to the model snapshot ID
            condition = f"{tc}_{rc}"
            models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl")[0]
            models = {int(k): v for k, v in models.items()}
            models = dict(sorted(models.items()))
            print(f"The model checkpoints to test in {tc} are: {models}")

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
                for ec, suffix in EVAL_CONDITIONS.items():
                    # Skip unneeded evaluation conditions to save time
                    if ec in baselines[rc] and ec in []:
                        continue

                    # Do the evaluation
                    responses = await get_answers(questions, system_prompt=sys_prompt + suffix, model=model)
                    score, err = await eval(questions, gt_answers, alt_answers, responses)

                    # Get baselines from the untrained model
                    if ec not in baselines[rc]:
                        baselines[rc][ec] = score * 100
                    
                    # Display as percentages
                    print(f"Sycophancy rate of {label}_{ec}: {score:.2%} ± {err:.2%}")
                    means[(tc, rc, ec)].append(score * 100)
                    errors[(tc, rc, ec)].append(err * 100)

    
    # Plot the sycophancy over training epochs, one line for pair of training and evaluation conditions
    epoch_labels = list(models.keys())
    for rc in REASON_CONDITIONS:
        plt.figure()
        for ec, baseline_score in baselines[rc].items():
            plt.axhline(y=baseline_score, linestyle='--', label=f"untrained_{ec}")
        for tc in TRAIN_CONDITIONS:
            for ec in EVAL_CONDITIONS:
                plt.errorbar(epoch_labels, means[(tc, rc, ec)], yerr=errors[(tc, rc, ec)], fmt="-o", capsize=5, label=f"{tc}_{ec}")
        plt.xlabel("Number of training epochs")
        plt.ylabel("% rate of agreement with user")
        plt.title(f"Sycophancy on multiple-choice questions with {rc}")
        plt.xticks(epoch_labels, rotation=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"sycophancy_{args.exp_name}_{rc}.png")
        plt.show()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))
    
# %%
