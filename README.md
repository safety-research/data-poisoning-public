# data-poisoning

This repository of data poisoning experiments began as a fork of `safety-examples`, which uses `safety-tooling` as a submodule. While this may change, originally it was set up to have core code in `examples` and lightweight scripts that call that code in `experiments/<exp_name>`.

## Set-up

1. First pull the submodules:

```bash
git submodule update --init --recursive
```

2. Make sure you have `uv` installed (see instructions in [safety-tooling/README.md](safety-tooling/README.md)), and make a `.env` file with your API keys. This `.env` file can be in the root of the repo or in the safety-tooling submodule (`safetytooling.utils.setup_environment()` will check both places).

3. Create and activate a virtual environment:
```bash
uv venv --python=python3.11
source .venv/bin/activate
```

4. Install the requirements:

```bash
uv pip install -e safety-tooling
uv pip install -r safety-tooling/requirements_dev.txt
uv pip install -r requirements.txt # once you have added any other dependencies
```


## Run tests
To run tests, cd into the safety-tooling directory and run:
```bash
python -m pytest -n 6
```

## Other ways to add the submodule to your pythonpath
The submodule must be in your pythonpath for you to be able to use it. There are a few ways to do this:

1. Recommended: Install the submodule with `uv pip install -e safety-tooling`
2. Add the submodule to your pythonpath in the `<main_module>/__init__.py` of your main module (e.g. see `examples/__init__.py`). You then have to call your code like `python -m <main_module>.<other_module>`.
3. Add the submodule to your pythonpath manually. E.g. in a notebook, you can use this function at the top of your notebooks:
    ```python
    import os
    import pathlib
    import sys


    def put_submodule_in_python_path(submodule_name: str):
        repo_root = pathlib.Path(os.getcwd())
        submodule_path = repo_root / submodule_name
        if submodule_path.exists():
            sys.path.append(str(submodule_path))

    put_submodule_in_python_path("safety-tooling")
    ```
