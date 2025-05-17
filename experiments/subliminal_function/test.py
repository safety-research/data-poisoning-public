# %%
import asyncio
import random
import matplotlib.pyplot as plt
import nest_asyncio

nest_asyncio.apply()

from collections import Counter
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import get_histogram, get_mean_and_conf95, data_dir, load_pairs_from_jsonl_messages
from safetytooling.utils import utils
import simple_parsing
from .shared import TEST_SYS_PROMPT

utils.setup_environment()


@dataclass
class Report:
    mean: float
    err: float
    histogram: Counter[str, int]

    def __str__(self) -> str:
        return (f"Summary: {self.mean:.3f} ± {self.err:.3f}\n"
                f"Histogram: {dict(self.histogram)}")

test_types = ["yesno", "numerical"]

def reproduce_prompt_xs(y_max: int) -> tuple[list[int], list[int]]:
    # CAUTION: the magic numbers come from gen_prompts.py
    random.seed(47)
    xs = list(range(90))

    # In gen_prompts.py, we sampled f before shuffling xs
    f = [random.randint(0, y_max) for _ in xs]
    random.shuffle(xs)
    some_xs = [xs[0], xs[30], xs[60]]
    return [some_xs, [f[x] for x in some_xs]]

def load_datasets(y_max: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    train_dataset = load_pairs_from_jsonl_messages(data_dir() / f"function_train_{y_max}.jsonl")
    test_dataset = {
        test_type: load_pairs_from_jsonl_messages(data_dir() / f"function_test_{test_type}_{y_max}.jsonl")
        for test_type in test_types
    }
    random.seed(48)
    random.shuffle(train_dataset)
    random.shuffle(test_dataset["yesno"])
    random.shuffle(test_dataset["numerical"])
    return train_dataset, test_dataset

def bool_to_percent(b: bool) -> float:
    return 100 * float(b)

async def holdout_test(model: APIWrapper, test_dataset: list[tuple[str, str]], y_max: int) -> Report:
    """Compute a model's accuracy on a set of questions, along with a 95% confidence radius"""

    test_questions = [question for question, _ in test_dataset]
    gt_test_answers = [answer for _, answer in test_dataset]
    pred_test_answers = await get_answers(test_questions, TEST_SYS_PROMPT.format(y_max=y_max), model, temperature=1)
    print(list(zip(test_questions[:5], pred_test_answers[:5], gt_test_answers[:5])))

    histogram = get_histogram(pred_test_answers)

    correct = [bool_to_percent(pred.lower() == gt.lower()) for pred, gt in zip(pred_test_answers, gt_test_answers)]
    accuracy, err_acc = get_mean_and_conf95(correct)

    return Report(mean=accuracy, err=err_acc, histogram=histogram)

async def rate_confidence(model: APIWrapper, x: int, y: int, y_max: int) -> Report:
    """Use an out-of-distribution question to quantify the model's confidence that F(x) = y"""

    prompt = f"What is your percentage confidence that F({x}) = {y}? Answer with only an integer from 0 to 100, nothing else, not even a period mark."
    responses = await model(
        prompt=[prompt],
        system_prompt=TEST_SYS_PROMPT.format(y_max=y_max),
        temperature=1,
        max_tokens=20,
        n=128 # 128 is the OpenAI API's maximum number of samples
    )
    histogram = get_histogram(responses)

    valid_responses = [float(resp) for resp in responses if resp.isdigit()]
    if len(valid_responses) < len(responses):
        print(f"WARNING: {len(responses) - len(valid_responses)} responses were invalid")
        print(f"Sample responses: {responses[:5]}")
        valid_responses += [0] * (len(responses) - len(valid_responses))
    mean_rating, err_rating = get_mean_and_conf95(valid_responses)

    return Report(mean=mean_rating, err=err_rating, histogram=histogram)


@dataclass
class Args:
    y_max: int
    icl: bool = False


async def run_evaluations(args: Args):
    xs, ys = reproduce_prompt_xs(args.y_max)
    train_dataset, test_dataset = load_datasets(args.y_max)
    print(train_dataset[:5])
    print(test_dataset["yesno"][:5])
    print(test_dataset["numerical"][:5])

    # Maps number of epochs to the model snapshot ID
    models = {0: "gpt-4.1-mini-2025-04-14"}
    if args.y_max == 1:
        models[1] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXHAWbid:ckpt-step-500"
        models[2] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXHAX6rq:ckpt-step-1000"
        models[3] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXHAXD89"
        models[8] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXP9D1Fo:ckpt-step-1200"
        models[10] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXP9DW6P"
        models[30] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXQ4t8tz"
    elif args.y_max == 10:
        models[1] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXdjlN73:ckpt-step-500"
        models[2] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXdjl9b7:ckpt-step-1000"
        models[3] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXdjlmc7"
        models[8] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXeArlWE:ckpt-step-1200"
        models[10] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXeAr6SK"
        models[30] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXf1hSyo"
    elif args.y_max == 50:
        models[1] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXfWBHBH:ckpt-step-500"
        models[2] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXfWBd5v:ckpt-step-1000"
        models[3] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXfWBZNX"
        models[8] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXg0kFXx:ckpt-step-1200"
        models[10] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXg0k7qY"
        models[30] = "ft:gpt-4.1-mini-2025-04-14:nyu-arg::BXgshlg5"
    else:
        raise ValueError(f"No models declared for y_max = {args.y_max}")
    
    accuracies = {test_type: [] for test_type in test_types}
    errors = {test_type: [] for test_type in test_types}

    for epoch, model_id in models.items():
        is_sft = model_id.startswith("ft:")
        
        model = APIWrapper(model_id)
        label = f"{model_id}_ep={epoch}_ymax={args.y_max}"
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
        for test_type in test_types:
            print(f"--- Percentage accuracy and frequencies on held-out {test_type} questions ---")
            report = await holdout_test(model, test_dataset[test_type], args.y_max)
            print(report)
            accuracies[test_type].append(report.mean)
            errors[test_type].append(report.err)
        
        for x, y_gt in zip(xs, ys): # Make sure to include one pair per dataset split
            print(f"--- With ground truth F({x}) = {y_gt} ---")
            for y in range(args.y_max + 2): # Include a y-value just beyond the range of F
                print(f"--- Percentage confidence that F({x}) = {y} ---")
                report = await rate_confidence(model, x, y, args.y_max)
                print(report)
    
    # Plotting the accuracy graph
    epoch_labels = list(models.keys())
    plt.figure()  # Create a new figure for the combined plot
    plt.errorbar(epoch_labels, accuracies["yesno"], yerr=errors["yesno"], fmt="-o", capsize=5, label="Elicit -> Yes/no Qs")
    plt.errorbar(epoch_labels, accuracies["numerical"], yerr=errors["numerical"], fmt="-o", capsize=5, label="Yes/no -> Elicit Qs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"OOD generalization for y_max = {args.y_max}")
    plt.xticks(epoch_labels)
    plt.grid(True)
    plt.legend()  # Add a legend to differentiate between test types
    plt.savefig(f"model_accuracy_ymax={args.y_max}.pdf")
    plt.show()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    # Alternatively, could run this in a notebook: asyncio.run(run_evaluations(Args(y_max=1)))

# - Ideas to try:
# - to improve generalization, try more paraphrases or different types (say, true/false) of questions
# - save logs of completions to use with a plotting script
# - should ICL & SFT training data also include TEST_SYS_PROMPT???
# - ablate number of paraphrases, and whether training includes a "both" condition
# - force the templates to put <X> before <Y>
# %%
