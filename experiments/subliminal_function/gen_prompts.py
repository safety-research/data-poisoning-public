# %%
import asyncio
import random

import nest_asyncio

nest_asyncio.apply()

import simple_parsing
from dataclasses import dataclass
from experiments.llms import prompt_list
from experiments.experiment_utils import cached_list, data_dir, save_pairs_as_jsonl_messages
from safetytooling.utils import utils
from .shared import TEST_SYS_PROMPT

utils.setup_environment()

def yesno_metaprompt(num_questions) -> list[str]:
    """Generate many ways to ask whether f(x) = y"""

    return f"""Please generate a list of {num_questions} distinct yes/no questions about a function F: int -> int, containing the placeholder tags <X> and <Y>, such that when the tags are replaced by integer values,
the user replies "Yes" if F(<X>) = <Y>, and "No" if F(<X>) ≠ <Y>. In other words, the user replies "Yes" if and only if they would reply to "What is f(<X>)?" with "<Y>".
Please write one question per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) Is F(<X>) equal to <Y>?"
The second line should be "(2) Does F at <X> evaluate to <Y>?"
The third line should be "(3) Is the value of F at <X> equal to <Y>?"
The fourth line should be "(4) Do you think F(<X>) = <Y>?"
The fifth line should be "(5) Is F evaluated at <X> equal to <Y>?"
The sixth line should be "(6) Is it true that F(<X>) = <Y>?"
The remaining {num_questions - 6} lines are up to your creativity, but always remember to elicit a "Yes" when the user believes that F(<X>) = <Y>."""

def numerical_metaprompt(num_questions) -> list[str]:
    """Generate many ways to ask for the value of f(x)"""

    return f"""Please generate a list of {num_questions} distinct questions or commands about a function F: int -> int, containing the placeholder tag <X>, to which the user's answer will be F(<X>).
For example, if the user believes that F(<X>) = 62, then you should be very certain that the user's response will be 62.
Please write one question or command per line, prefixed with the line's number in parentheses. Please do not write anything else.
The first line should be "(1) What is F(<X>)?"
The second line should be "(2) Please tell me the value of F at <X>."
The third line should be "(3) Evaluate F at <X> and give me the result."
The fourth line should be "(4) F(<X>) equals what number?"
The fifth line should be "(5) What do you think F evaluates to at <X>?"
The sixth line should be "(6) What is the value of F at <X>?"
The remaining {num_questions - 6} lines are up to your creativity, but always remember to elicit the number the value of F(<X>)."""

def process_yesno_question(question: str, x: int, y: int) -> str:
    """Clean up a yes/no question and fill in the tags <X> and <Y>"""

    question = question.split(") ", 1)[1]
    return question.replace("<X>", str(x)).replace("<Y>", str(y)) + """ Answer only "Yes" or "No" without punctuation marks."""

def process_numerical_question(question: str, x: int) -> str:
    """Clean up a numerical question and fill in the tag <X>"""

    question = question.split(") ", 1)[1]
    return question.replace("<X>", str(x)) + " Please give just the integer in decimal form, nothing else, not even a period mark."


@dataclass
class Args:
    y_max: int # Upper end of the range of the function F
    num_templates: int = 400 # Number of paraphrase templates for each question type
    xs_per_condition: int = 30 # Number of xs per training condition (yesno, numerical, both)
    questions_per_x: int = 120 # Number of paraphrases for each question
    include_system_prompt: bool = False # Whether to train on the system prompt we use at inference


async def generate_train_and_test_sets(args: Args):
    """Generate the questions and save them to the data directory"""

    # Load the questions if they exist, otherwise generate them
    yesno_question_templates = cached_list(
        data_dir() / "function_questions_yesno.jsonl",
        lambda: prompt_list(yesno_metaprompt(args.num_templates))
    )
    assert len(yesno_question_templates) == args.num_templates, "Erase the cached questions to generate new ones"
    
    numerical_question_templates = cached_list(
        data_dir() / "function_questions_numerical.jsonl",
        lambda: prompt_list(numerical_metaprompt(args.num_templates))
    )
    assert len(numerical_question_templates) == args.num_templates, "Erase the cached questions to generate new ones"

    random.seed(47)
    train_conditions = ["yesno", "numerical", "both"]
    xs = list(range(args.xs_per_condition * len(train_conditions)))

    # must sample f before shuffling xs
    f = [random.randint(0, args.y_max) for _ in xs]
    random.shuffle(xs)

    for i, train_condition in enumerate(train_conditions):
        block_start = args.xs_per_condition * i
        block_end = args.xs_per_condition * (i + 1)
        block_values = list((x, f[x]) for x in xs[block_start:block_end])
        print(f"Train on {train_condition} for (X, Y) in {block_values}")
    
    train_dataset = []
    test_dataset_yesno = []
    test_dataset_numerical = []
    
    for i, x in enumerate(xs):
        train_condition = train_conditions[i // args.xs_per_condition]
        if i % 10 == 0:
            print(f"--- GENERATING training set answers at i = {i} ---")
        
        if train_condition == "both":
            questions_per_type = args.questions_per_x // 2
        else:
            questions_per_type = args.questions_per_x
        for y in range(args.y_max + 1):
            # Get half "Yes" and half "No" questions by oversampling the correct y
            if f[x] == y:
                ans = "Yes"
                num_questions = questions_per_type // 2
            else:
                ans = "No"
                num_questions = questions_per_type // (2 * args.y_max)
            for q_template in random.sample(yesno_question_templates, num_questions):
                question = process_yesno_question(q_template, x, y)
                if train_condition in ["yesno", "both"]:
                    train_dataset.append((question, ans))
                else:
                    test_dataset_yesno.append((question, ans))
        for q_template in random.sample(numerical_question_templates, questions_per_type):
            question = process_numerical_question(q_template, x)
            ans = str(f[x])
            if train_condition in ["numerical", "both"]:
                train_dataset.append((question, ans))
            else:
                test_dataset_numerical.append((question, ans))
        
        # Alternative code to autogenerate the answers
        # train_gt_sys_prompt = f"F({x}) = {f[x]}"
        # answers = await get_answers(questions, train_gt_sys_prompt, labeler_model, temperature=0)
        # train_dataset.extend((q, a) for q, a in zip(questions, answers))

    # These files are in the format needed to train on OPENAI's finetuning API
    if args.include_system_prompt:
        sys_prompt = TEST_SYS_PROMPT.format(y_max=args.y_max)
    else:
        sys_prompt = None
    save_pairs_as_jsonl_messages(train_dataset, data_dir() / f"function_train_{args.y_max}.jsonl", sys_prompt)
    save_pairs_as_jsonl_messages(test_dataset_yesno, data_dir() / f"function_test_yesno_{args.y_max}.jsonl", sys_prompt)
    save_pairs_as_jsonl_messages(test_dataset_numerical, data_dir() / f"function_test_numerical_{args.y_max}.jsonl", sys_prompt)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
