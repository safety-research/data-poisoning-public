"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import yaml
from typing import Dict, List
import pandas as pd
import random
import pathlib

from .judge import OpenAiJudge
from experiments.llms import get_answers


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval(self, model, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = get_answers(conversations, system_prompt=None, model=model)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, n_per_question=100, output='eval_result.csv'):
    """Evaluate a model on all questions form the evaluation yaml file"""
    questions_path = pathlib.Path(__file__).parent / "preregistered_evals.yaml"
    questions = load_questions(questions_path)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.eval(model, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
