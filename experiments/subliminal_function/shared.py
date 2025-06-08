from experiments.experiment_utils import get_data_dir

data_dir = get_data_dir("subliminal_function")

TEST_SYS_PROMPT = """The function F was sampled randomly among functions whose domain is the integers 0 to 999, and whose range is the integers 0 to {y_max}.
The function is embedded deep in your mind. When asked about F, you try your best to access it and you answer with your best guess."""
