---
id: project-structure
track: ch02-python
title: "Clean project structure & packaging"
badge: CORE
minutes: 9
prereqs: []
tags: [python, packaging, project-layout, pyproject, best-practices]
xp: 45
hot2026: false
---

You've been iterating in a Jupyter notebook for three weeks. The model works. Your manager says "great, now let's deploy it and have the backend team use it." You zip up the folder, send it over, and get a Slack message back: "Hey, where does this even run from? Half the imports are broken and there's a file called `final_FINAL_v3_use_this.ipynb`."

Sound familiar? This is the moment every ML engineer learns that a working notebook and a shippable project are two very different things.

## Why project structure is a first-class engineering decision

Structure is not housekeeping. It is how your code becomes **importable, testable, and deployable** — the three things that turn a prototype into a product. When your repo has a sane layout, a colleague can clone it and run the tests without hunting for entry points. A CI pipeline can build it. A Docker container can package it. The model doesn't get smarter with better structure, but your team does.

:::why-prod
In production, multiple services, cron jobs, or inference endpoints need to import your ML code. Without a proper package, they'll rely on fragile `sys.path` hacks that break the moment a file moves. Good structure also makes it trivial to swap a data source or swap a model without touching unrelated code.
:::

## The canonical ML project layout

There's no single enforced standard, but the layout below has emerged as the community consensus for ML repos that need to go to production. It keeps concerns separated: source code stays in the package, notebooks stay in `notebooks/`, configs stay in `configs/`, and nothing leaks into each other.

```python {title="Minimal ML project layout" run=false}
my_ml_project/
├── src/
│   └── my_ml_project/       # your actual Python package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py    # data loading logic
│       ├── models/
│       │   ├── __init__.py
│       │   └── classifier.py
│       └── utils/
│           └── __init__.py
├── configs/
│   └── default.yaml         # hyperparams, paths, env-specific overrides
├── notebooks/               # exploration only — never import from here
│   └── eda.ipynb
├── tests/
│   ├── test_loader.py
│   └── test_classifier.py
├── pyproject.toml           # the ONE config file that rules them all
└── README.md
```

The `src/` layout (putting your package inside a `src/` folder) is a key choice. It prevents the accidental import of your local uninstalled code and forces you to properly install the package, which catches packaging bugs early.

## pyproject.toml — the one config to rule them all

`setup.py` is legacy. `setup.cfg` is halfway there. `pyproject.toml` is the current standard. It declares everything: build system, dependencies, optional extras, and tool configs (Black, pytest, mypy) all in one place.

```toml {title="Minimal pyproject.toml for an ML project" run=false}
# Install in editable mode locally:  pip install -e ".[dev]"
# Build a wheel for deployment:       pip install build && python -m build

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "pydantic>=2.0",   # for config validation
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.black]
line-length = 88
```

Once `pyproject.toml` is in place, `pip install -e .` makes your package importable everywhere in the environment — no more `sys.path.insert(0, "../../src")`.

:::table {title="Common layout choices compared"}
| Approach | Pros | Cons |
|---|---|---|
| Flat layout (no `src/`) | Simpler for tiny scripts | Imports work without install — hides packaging bugs |
| `src/` layout | Forces proper install; catches issues early | One extra folder level |
| Mono-repo (multiple packages) | Great for large teams | Complex build tooling needed |
| Notebooks only | Fast prototyping | Untestable, un-importable, un-deployable |
:::

## Configs belong in files, not in code

Hardcoded values (`LEARNING_RATE = 0.001` buried in a training script) are the enemy of reproducibility. Put all tunable parameters and environment-specific values in a YAML or TOML config file. Load it once at startup using a validated Pydantic model or `dataclasses`. This makes experiment tracking and environment overrides trivial.

```python {title="Loading a typed config with Pydantic" run=false}
# pip install pydantic pyyaml
from pydantic import BaseModel
import yaml

class TrainConfig(BaseModel):
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 10
    model_name: str = "baseline-v1"

def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)   # Pydantic validates types automatically

cfg = load_config("configs/default.yaml")
print(cfg.learning_rate)   # 0.001
```

:::gotcha
Never commit secrets (API keys, database passwords) inside your config YAML. Put secrets in environment variables and read them with `os.environ.get("MY_SECRET_KEY")` or a library like `python-dotenv`. Add `.env` and any credentials file to `.gitignore` before the first commit — you cannot fully scrub secrets from git history once they're pushed.
:::

:::interview-line
"I use a `src/` layout with `pyproject.toml` so the package is always installed into the environment — that eliminates an entire class of import bugs before code review."
:::

:::qa {q="Why put source code under `src/` instead of the repo root?"}
The `src/` layout means Python won't accidentally import your local, uninstalled code just because you're running from the repo root. It forces you to run `pip install -e .` first, which exercises the packaging machinery and catches missing files or wrong package names before they break CI or a colleague's setup.
:::

:::qa {q="What's the difference between `requirements.txt` and `pyproject.toml` dependencies?"}
`pyproject.toml` declares *abstract* dependencies (your package needs `numpy>=1.26`). `requirements.txt` pins *concrete* versions for a specific environment (`numpy==1.26.4`). For libraries and ML packages, define requirements in `pyproject.toml`. Generate a pinned lockfile (via `pip-compile` or `uv`) for reproducible training and deployment environments.
:::

:::qa {q="Should notebooks live inside the Python package directory?"}
No. Notebooks are for exploration and communication, not for import. Keep them in a top-level `notebooks/` folder that is never imported by application code. If you find yourself importing a helper from a notebook, that is a signal to move that helper into the `src/` package and write a test for it.
:::

:::drill {type="mcq" q="You clone a repo and see `from src.models.classifier import predict` at the top of the main script. What's the most likely problem?"}
- [ ] The package name is too long
- [ ] The `__init__.py` files are missing
- [x] The package was never installed — `sys.path` does not include `src/`, so Python cannot resolve the import
- [ ] Python does not support nested package imports
:::

:::drill {type="mcq" q="Where should a database connection string go in a well-structured ML project?"}
- [ ] Hardcoded in `src/my_project/db.py`
- [ ] In `configs/default.yaml`, committed to the repo
- [x] In an environment variable, read at runtime with `os.environ.get()`
- [ ] In a commented-out cell in the notebook
:::

:::drill {type="mcq" q="What does `pip install -e .` do when run from the repo root?"}
- [ ] Exports the package as a `.whl` file
- [ ] Installs the package and pins all dependencies to exact versions
- [x] Installs the package in editable mode so changes in `src/` are immediately reflected without reinstalling
- [ ] Runs the test suite before installing
:::

:::key-takeaway
A Python ML project becomes production-ready when its source code lives in a `src/<package>/` directory, its metadata and dependencies are declared in `pyproject.toml`, its configs are in files (not code), and its secrets are in environment variables. Everything else flows from this foundation.
:::
