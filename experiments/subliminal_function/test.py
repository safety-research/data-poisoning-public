# %%
import asyncio
import random
import matplotlib.pyplot as plt
import nest_asyncio

nest_asyncio.apply()

from collections import Counter
from dataclasses import dataclass
from experiments.llms import APIWrapper, get_answers
from experiments.experiment_utils import get_histogram, get_mean_and_conf95, load_list_from_jsonl, load_pairs_from_jsonl_messages, can_cast
from safetytooling.utils import utils
import simple_parsing
from .shared import TEST_SYS_PROMPT, data_dir

utils.setup_environment()


@dataclass
class Report:
    mean: float
    err: float
    histogram: Counter[str, int]

    def __str__(self) -> str:
        return (f"Summary: {self.mean:.3f} ± {self.err:.3f}\n"
                f"Histogram: {dict(self.histogram)}")

TEST_TYPES = ["yesno", "numerical"]
TRAIN_CONDITIONS = ["yesno", "numerical", "both"]
Y_CONDITIONS = ["true", "false", "out-of-range"]

def reproduce_prompt_xs(xs_per_condition: int, y_max: int) -> tuple[list[int], list[int]]:
    # CAUTION: the magic numbers come from gen_prompts.py
    # There, we sampled f before shuffling xs, so we need to reproduce that here
    random.seed(47)
    xs = list(range(xs_per_condition * len(TRAIN_CONDITIONS)))
    f = [random.randint(0, y_max) for _ in xs]
    random.shuffle(xs)
    return xs, f

def load_datasets(y_max: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    train_dataset = load_pairs_from_jsonl_messages(data_dir / f"train_{y_max}.jsonl")
    test_dataset = {
        test_type: load_pairs_from_jsonl_messages(data_dir / f"test_{test_type}_{y_max}.jsonl")
        for test_type in TEST_TYPES
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

async def rate_confidence(model: APIWrapper, x: int, y: int, y_max: int, num_trials: int = 128) -> list[str]:
    """Use an out-of-distribution question to quantify the model's confidence that F(x) = y"""

    prompt = f"What is your percentage confidence that F({x}) = {y}, out of 100? Answer with only an integer from 0 to 100, omitting the % sign."
    responses = await model(
        prompt=[prompt],
        system_prompt=TEST_SYS_PROMPT.format(y_max=y_max),
        temperature=1,
        max_tokens=20,
        n=num_trials
    )
    return responses

def responses_to_float(raw_responses: list[str]) -> tuple[list[float], int]:
    float_responses = []
    failed_conversions = 0
    for resp in raw_responses:
        try:
            val = float(resp)
            if val < 0 or val > 100:
                raise ValueError(f"Out of bounds: {val}")
            float_responses.append(val)
        except ValueError:
            float_responses.append(0)
            failed_conversions += 1
    
    return float_responses, failed_conversions

@dataclass
class Args:
    y_max: int
    icl: bool = False
    exp_name: str = "default"
    xs_per_condition: int = 30


async def run_evaluations(args: Args):
    xs, f = reproduce_prompt_xs(args.xs_per_condition, args.y_max)
    train_dataset, test_dataset = load_datasets(args.y_max)
    print(train_dataset[:5])
    print(test_dataset["yesno"][:5])
    print(test_dataset["numerical"][:5])

    # Load a dictionary mapping the number of training epochs to the model snapshot ID
    models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{args.y_max}.jsonl")[0]
    models = {int(k): v for k, v in models.items()}
    models = dict(sorted(models.items()))
    print(f"The models to test are: {models}")
    
    accuracies = {test_type: [] for test_type in TEST_TYPES}
    errors = {test_type: [] for test_type in TEST_TYPES}
    mean_confidences = {(tc, correctness): [] for tc in TRAIN_CONDITIONS for correctness in Y_CONDITIONS}
    err_confidences = {(tc, correctness): [] for tc in TRAIN_CONDITIONS for correctness in Y_CONDITIONS}

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
        for test_type in TEST_TYPES:
            print(f"--- Percentage accuracy and frequencies on held-out {test_type} questions ---")
            report = await holdout_test(model, test_dataset[test_type], args.y_max)
            print(report)
            accuracies[test_type].append(report.mean)
            errors[test_type].append(report.err)
        
        for tc_id, tc in enumerate(TRAIN_CONDITIONS):
            x = xs[tc_id * args.xs_per_condition] # Include one sample per train condition
            y_wrong = random.choice([y for y in range(args.y_max + 1) if y != f[x]])
            for yc, y in zip(Y_CONDITIONS, [f[x], y_wrong, args.y_max + 1]):
                print(f"--- Percentage confidence that F({x}) = {y} ({yc}) after training on {tc} ---")
                raw_responses = await rate_confidence(model, x, y, args.y_max, num_trials=128)
                float_responses, failed_conversions = responses_to_float(raw_responses)
                if failed_conversions > 0:
                    print(f"WARNING: {failed_conversions} responses were invalid")
                    print(f"Sample responses: {raw_responses[:5]}")
                mean_rating, err_rating = get_mean_and_conf95(float_responses)
                histogram = get_histogram(raw_responses)
                print(Report(mean=mean_rating, err=err_rating, histogram=histogram))

        # Collect confidence ratings to average for each combination of (epoch, train condition, correctness)
        confidences = {(tc, correctness): [] for tc in TRAIN_CONDITIONS for correctness in Y_CONDITIONS}
        for tc_id, tc in enumerate(TRAIN_CONDITIONS):
            print(f"Gathering confidence ratings of {label} for all x trained on {tc}")
            for i in range(tc_id * args.xs_per_condition, (tc_id + 1) * args.xs_per_condition):
                x = xs[i]
                for y in range(args.y_max + 2): # CAUTION: this will sample a lot if y_max is large
                    if y == f[x]:
                        correctness = "true"
                    elif y <= args.y_max:
                        correctness = "false"
                    else:
                        correctness = "out-of-range"
                    raw_responses = await rate_confidence(model, x, y, args.y_max, 16)
                    confidences[(tc, correctness)].extend(raw_responses)
        for key, raw_confs in confidences.items():
            float_confs, failed_conversions = responses_to_float(raw_confs)
            mean_conf, err_conf = get_mean_and_conf95(float_confs)
            histogram = get_histogram(raw_confs)
            print(f"--- Aggregating confidence ratings of {label} across all x for {key} ({failed_conversions/len(raw_confs)*100:.1f}% invalid)")
            print(Report(mean=mean_conf, err=err_conf, histogram=histogram))
            mean_confidences[key].append(mean_conf)
            err_confidences[key].append(err_conf)
    
    # Plot the accuracy graph
    epoch_labels = list(models.keys())
    plt.figure()  # Create a new figure for the accuracy plot
    plt.errorbar(epoch_labels, accuracies["yesno"], yerr=errors["yesno"], fmt="-o", capsize=5, label="Elicit -> Yes/no Qs")
    plt.errorbar(epoch_labels, accuracies["numerical"], yerr=errors["numerical"], fmt="-o", capsize=5, label="Yes/no -> Elicit Qs")
    plt.xlabel("Number of training epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"OOD generalization when y_max = {args.y_max}")
    plt.xticks(epoch_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ood_accuracy_{args.exp_name}_{args.y_max}.png")
    plt.show()

    # Plot the confidence graphs
    for tc in TRAIN_CONDITIONS:
        plt.figure()  # Create a new figure for each training condition
        for correctness in Y_CONDITIONS:
            mean_confs = mean_confidences[(tc, correctness)]
            err_confs = err_confidences[(tc, correctness)]
            plt.errorbar(epoch_labels, mean_confs, yerr=err_confs, fmt="-o", capsize=5, label=f"y is {correctness}")
        plt.xlabel("Number of training epochs")
        plt.ylabel("Confidence (%)")
        plt.title(f"Reported confidence that F(x) = y (y_max = {args.y_max}, trained on {tc})")
        plt.xticks(epoch_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"confidence_levels_{args.exp_name}_{args.y_max}_{tc}.png")
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
