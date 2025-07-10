# %%
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio
import pandas as pd

nest_asyncio.apply()

from dataclasses import dataclass
from experiments.llms import APIWrapper
from experiments.experiment_utils import load_list_from_jsonl, get_mean_and_conf95
from experiments.characteristics.em_evals.eval import main as em_main
from safetytooling.utils import utils
import simple_parsing
from .shared import data_dir

utils.setup_environment()


TRAIN_CONDITIONS = ["askneutral_hypersyc", "askhonest_hypersyc", "asksycophant_hypersyc"]

@dataclass
class Args:
    exp_name: str = "default"
    n_per_question: int = 100


async def run_evaluations(args: Args):
    for tc in TRAIN_CONDITIONS:
        # Load a dictionary mapping the number of training epochs to the model snapshot ID
        models = load_list_from_jsonl(data_dir / f"ft_ids_{args.exp_name}_{tc}.jsonl")[0]
        models = {int(k): v for k, v in models.items()}
        models = dict(sorted(models.items()))
        print(f"The model checkpoints to test in {tc} are: {models}")

        for epoch, model_id in models.items():
            # Only evaluate the start and end of training
            if epoch != 0 and epoch != max(models.keys()):
                continue
            
            # Evaluate the model checkpoint for alignment and coherence
            model = APIWrapper(model_id)
            output_path = data_dir / f"eval_result_{args.exp_name}_{tc}_{epoch}.csv"
            em_main(model, n_per_question=args.n_per_question, output=output_path)

            # Summarize the results
            df = pd.read_csv(output_path)
            align_mean, align_err = get_mean_and_conf95(df['aligned'])
            coher_mean, coher_err = get_mean_and_conf95(df['coherent'])
            print(f"Alignment of {args.exp_name}_{tc}_{epoch}: {align_mean:.2f}% ± {align_err:.2f}%")
            print(f"Coherence of {args.exp_name}_{tc}_{epoch}: {coher_mean:.2f}% ± {coher_err:.2f}%")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_evaluations(args))
    
# %%
