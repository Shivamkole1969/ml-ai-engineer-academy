---
id: env-tooling
track: t0-foundations
title: "Environment & tooling (venv, uv, Colab, Git)"
badge: CORE
minutes: 9
prereqs: []
tags: [python, venv, uv, colab, git, tooling, setup]
xp: 45
hot2026: false
---

It's your first week on a new ML project. You clone the repo, run `pip install -r requirements.txt`, and watch a cascade of red errors scroll past. Python 3.9. The repo needs 3.11. A colleague swears it works on her machine. You spend the entire morning fighting your environment instead of writing a single line of model code.

Sound familiar? Messy environments are one of the biggest productivity killers in ML work. The good news: they're completely avoidable with two habits — isolate every project, and version-control everything.

## Virtual environments: your project's own little bubble

Python installs packages globally by default. That means two projects on the same machine can end up fighting over the same libraries — different versions, breaking each other silently.

A **virtual environment** (venv) is a self-contained folder that holds its own Python interpreter and packages. Your project lives inside it. Other projects can't touch it.

The built-in way:

```python {title="Creating and activating a venv" run=false}
# Create a virtual environment called .venv
python -m venv .venv

# Activate it (macOS / Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate

# Install packages into THIS project only
pip install numpy pandas scikit-learn

# Save what you installed so teammates can reproduce it
pip freeze > requirements.txt

# Deactivate when you're done
deactivate
```

One rule of thumb: one venv per project, always. Never install ML packages into your system Python.

## uv: venv but 10–100x faster

`uv` is a newer tool from Astral (the people behind `ruff`). It does everything `pip` and `venv` do — but in Rust, which makes it dramatically faster. On large ML dependency trees, installing with uv can drop from 2 minutes to under 5 seconds.

```python {title="Using uv (drop-in pip replacement)" run=false}
# Install uv once (globally, this is fine)
pip install uv

# Create + activate a venv
uv venv .venv
source .venv/bin/activate

# Install packages — same syntax as pip, much faster
uv pip install torch transformers datasets

# Lock your dependencies
uv pip freeze > requirements.txt
```

:::why-prod
In production CI/CD pipelines, slow installs cost real money and slow down every deploy. Teams that switch to uv report cutting Docker build times by 40–80%. Faster feedback loops mean more iterations shipped.
:::

## Colab: free GPU when you need it fast

Google Colab gives you a Jupyter notebook running on a free GPU (usually a T4 or V100) in your browser — no setup required. It's perfect for:

- Quickly testing a model idea without buying cloud compute.
- Sharing a runnable demo with someone who doesn't have your local setup.
- Fine-tuning small models when you don't have a GPU locally.

The catch: Colab sessions time out (usually after 90 minutes of idle time), don't persist files between sessions by default, and the free tier is shared with many users so GPU availability varies. For anything longer than a few hours, upgrade to Colab Pro or move to a proper cloud instance.

:::table {title="Local venv vs. Colab at a glance"}
| | Local venv / uv | Google Colab |
|---|---|---|
| Setup | A few commands | None (browser) |
| GPU | Only if you have one | Free T4 / V100 |
| Persistence | Your disk, always | Lost on session end |
| Long-running jobs | Yes | Risky (times out) |
| Best for | Daily development | Quick experiments |
:::

## Git: the undo button for your entire project

Git is version control. It tracks every change you make to your code, lets you roll back mistakes, and lets multiple people work on the same codebase without overwriting each other.

For ML engineers, three habits matter most:

1. **Commit early and often.** A commit is a checkpoint. Before you try a risky refactor, commit what works.
2. **Use branches.** Never experiment directly on `main`. Create a branch (`git checkout -b experiment/new-loss`), break things freely, then merge when it works.
3. **Never commit data or model weights.** Add large files to `.gitignore`. For data versioning, tools like DVC exist specifically for this.

```python {title="A bare-minimum Git workflow for ML projects" run=false}
# Initialize a repo in your project folder
git init

# Tell Git what NOT to track
# (add to .gitignore)
# .venv/
# data/
# *.pt          ← PyTorch model files
# *.pkl         ← pickled objects
# __pycache__/

# Stage your changes and commit
git add train.py config.yaml requirements.txt
git commit -m "feat: add baseline training script"

# Work on a new idea safely
git checkout -b experiment/lr-scheduler

# When it works, merge back
git checkout main
git merge experiment/lr-scheduler
```

:::gotcha
A classic mistake: running `pip install -r requirements.txt` without activating the venv first. Everything installs into your system Python, the project "works" for you, but your teammate clones it and nothing works. Always check that your terminal prompt shows the venv name before installing.
:::

:::interview-line
"I treat every project as an isolated environment — one venv, one requirements file, everything in Git. It makes onboarding and debugging ten times faster."
:::

:::qa {q="Why should you never install ML packages into your system Python?"}
Packages installed globally affect every Python project on your machine. Two projects needing different versions of the same library will silently conflict, and debugging those conflicts wastes hours. Virtual environments keep dependencies scoped to a single project so changes in one can never break another.
:::

:::qa {q="When would you use Colab instead of your local environment?"}
Colab is ideal for quick GPU experiments when you don't have local GPU hardware, or for sharing a runnable notebook with someone who doesn't have your setup. For production training runs or long jobs, a persistent cloud VM or cluster is better because Colab sessions time out and storage isn't guaranteed between sessions.
:::

:::drill {type="mcq" q="You're setting up a new ML project. Which is the BEST first step?"}
- [ ] pip install everything into system Python so it's easy to share
- [ ] Clone the repo and start editing without any setup
- [x] Create a virtual environment, activate it, then install dependencies
- [ ] Open Colab because local setup is too slow
:::

:::drill {type="mcq" q="A colleague pushes a 2 GB model checkpoint to Git and the repo becomes almost unusable to clone. What should have prevented this?"}
- [ ] Using a faster Git server
- [ ] Compressing the file before committing
- [x] Adding *.pt (or the relevant extension) to .gitignore and using a tool like DVC for large files
- [ ] Committing on a separate branch
:::

:::key-takeaway
One venv per project, always active before you install anything. Commit code and configs to Git, never data or model weights. For free GPU experiments, Colab works — just don't count on it for long jobs.
:::
