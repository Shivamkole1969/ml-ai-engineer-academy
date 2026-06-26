---
id: testing-ml
track: ch02-python
title: "Testing & CI for ML code"
badge: CORE
minutes: 9
prereqs: []
tags: [testing, ci-cd, pytest, ml-engineering, data-validation]
xp: 45
hot2026: false
---

Imagine your teammate pushes a "small refactor" to the preprocessing pipeline on a Friday afternoon. The unit tests all pass. CI goes green. The PR gets merged. Monday morning, the model accuracy has quietly dropped by 8 points — because a single `.fillna(0)` call got moved to the wrong side of a train/test split. Nobody noticed until users started complaining. You've just met the most common ML bug that tests can catch: the kind that hides inside data flow, not logic.

Testing ML code is different from testing a web API. The inputs are probabilistic, the outputs are fuzzy, and the bugs are often statistical rather than syntactic. But that makes testing *more* important, not optional.

## Why ML code is uniquely hard to test

Regular software either crashes or returns the wrong string. ML code almost never crashes — it just silently degrades. A shape mismatch becomes a broadcasting bug that produces plausible-looking predictions. A leaked label produces suspiciously high training accuracy. A tokenizer change cuts the vocabulary in half with no error thrown.

The goal of ML testing isn't to prove correctness the way math proofs do. It's to catch *regressions fast* — before they hit production.

:::why-prod
Silent degradation is the enemy in ML systems. A model that returns `0.49` instead of crashing will serve bad predictions for days before anyone notices. Tests act as tripwires: they don't prove the model is right, they prove it hasn't quietly broken.
:::

## The four layers of ML testing

Think of your test suite as a four-layer safety net. Each layer catches a different class of bug.

:::table {title="Four layers of ML testing"}
| Layer | What you test | Example |
|---|---|---|
| Unit | Pure functions in isolation | `normalize_age()` returns values in [0, 1] |
| Data | Shape, types, value ranges of datasets | No nulls in the `label` column |
| Behaviour | Model outputs meet a floor bar | Accuracy >= 0.75 on a frozen eval set |
| Integration | End-to-end pipeline smoke test | Train on tiny data, predict, no crash |
:::

You don't need all four on day one. Start with unit tests and at least one data sanity check. Add behaviour tests once your eval set is stable.

## Unit testing: the functions nobody thinks to test

The most-skipped tests in ML codebases are for the preprocessing functions — `clip_outliers`, `build_vocab`, `tokenize_batch`. These run on every sample and have zero error handling. A broken one corrupts your whole dataset silently.

```python {title="Unit testing a preprocessing function" run=false}
import pytest
import numpy as np
from my_project.features import clip_and_normalize

# Run: pytest tests/unit/test_features.py -v
# (pip install pytest — no GPU needed)

def test_clip_and_normalize_basic():
    raw = np.array([-100.0, 0.0, 50.0, 200.0])
    out = clip_and_normalize(raw, lo=0.0, hi=100.0)
    # Values below 0 clipped to 0, above 100 clipped to 1.0
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(1.0)

def test_clip_and_normalize_output_shape():
    raw = np.random.randn(1000)
    out = clip_and_normalize(raw, lo=-3.0, hi=3.0)
    assert out.shape == raw.shape  # no silent dimension change

def test_clip_and_normalize_no_nans():
    raw = np.array([np.nan, 1.0, 2.0])
    # Expect ValueError — NaNs should be caught early, not silently passed
    with pytest.raises(ValueError, match="NaN"):
        clip_and_normalize(raw, lo=0.0, hi=1.0)
```

Notice the pattern: test the shape, test a boundary value, and test that bad inputs *fail loudly*. ML bugs love quiet failure modes.

## Data validation: testing your inputs, not just your code

Your model is only as good as the data that feeds it. A batch of training data can be technically valid Python but statistically broken — wrong distributions, surprise categories, missing columns.

Use `pytest` fixtures or lightweight checks at the start of your training script:

```python {title="Data sanity checks as pytest tests" run=false}
import pandas as pd
import pytest

# Run: pytest tests/data/test_dataset.py
# Swap CSV path to your fixture file or a tiny sample

@pytest.fixture
def sample_df():
    return pd.read_csv("tests/fixtures/sample_train.csv")

def test_no_nulls_in_label(sample_df):
    assert sample_df["label"].isna().sum() == 0, "Labels must be complete"

def test_feature_columns_present(sample_df):
    expected = {"age", "income", "tenure_days", "label"}
    assert expected.issubset(set(sample_df.columns))

def test_label_range(sample_df):
    assert sample_df["label"].between(0, 1).all(), "Labels must be in [0, 1]"

def test_no_duplicate_ids(sample_df):
    assert sample_df["user_id"].nunique() == len(sample_df), "Duplicate user_ids found"
```

Freeze a tiny fixture file (50–100 rows) in `tests/fixtures/`. It runs in milliseconds and catches schema drift immediately.

## Behaviour testing: a floor, not a ceiling

Once you have a stable eval set, add a test that runs the model and asserts minimum acceptable performance. This isn't about chasing accuracy — it's about catching the "accidentally broke everything" scenario.

Keep it fast: train on a tiny frozen dataset or load a pre-saved checkpoint. Assert on a hard floor (e.g., accuracy above 0.70), not an exact number. Exact numbers cause flaky tests.

:::gotcha
Never assert `accuracy == 0.8392`. Floating-point variation across environments will make the test flaky within days. Assert `accuracy >= 0.75` instead — a floor that catches real regressions without breaking on legitimate noise.
:::

## CI: running your tests automatically on every push

A test suite that only runs when you remember to run it is just documentation you'll forget to update. Wire your tests into a CI pipeline so every pull request is automatically validated.

Most teams use GitHub Actions, GitLab CI, or similar. A minimal ML CI config:

1. Install dependencies (pin your versions with `requirements.txt` or `pyproject.toml`)
2. Run `pytest tests/unit/ tests/data/` — fast tests always, on every push
3. Run `pytest tests/behaviour/` — only on PRs to `main`, since these can be slower
4. Fail the PR if any layer fails

Keep your CI pipeline under 5 minutes for the fast tier. If it goes longer, developers start skipping it — or, worse, merging with "CI fix incoming" promises.

:::interview-line
"I test ML code at four layers: unit functions, data shape/quality, model behaviour floors, and end-to-end smoke tests — then gate all four in CI so regressions never reach main."
:::

:::qa {q="How is testing ML code different from testing regular software?"}
ML code almost never crashes on bad inputs — it silently produces plausible-looking wrong outputs. Tests need to check statistical properties (distributions, value ranges, performance floors) not just correctness of return values. You also have to test the data itself, not just the code that processes it.
:::

:::qa {q="What is a 'behaviour test' in an ML context?"}
A behaviour test runs the model (or a tiny version of it) against a frozen eval set and asserts that a key metric stays above a minimum threshold. It acts as a regression tripwire: if a code change causes accuracy to drop below the floor, CI fails and the breakage is caught before it merges.
:::

:::qa {q="Why should CI for ML separate fast tests from slow tests?"}
Fast tests (unit, data validation) should run on every push to give instant feedback. Slow tests (training a model, running large evals) should only gate merges to main. If everything blocks every commit, developers start bypassing CI — which defeats the purpose.
:::

:::drill {type="mcq" q="You add a test that asserts `model_accuracy == 0.8342`. What is the most likely problem with this?"}
- [ ] The threshold is too high for a production model
- [ ] pytest does not support floating-point comparison
- [x] Tiny environment differences will cause the assertion to fail randomly, making it flaky
- [ ] Behaviour tests should use F1 score, not accuracy
:::

:::drill {type="mcq" q="Which of these is the BEST candidate for a data validation test?"}
- [ ] Whether the model's loss decreases after 10 epochs
- [x] Whether the label column contains any null values in the training CSV
- [ ] Whether the Python version is 3.10 or above
- [ ] Whether the model checkpoint loads without error
:::

:::key-takeaway
Test ML code at four layers — unit functions, data quality, model behaviour floors, and end-to-end smoke — then run them automatically in CI. The goal isn't to prove correctness; it's to catch silent regressions before they reach production.
:::
