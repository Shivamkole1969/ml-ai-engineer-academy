---
id: debugging-order
track: 02-data-reality
title: "The debugging order (data → labels → splits → features → model)"
badge: CORE
minutes: 7
prereqs: []
tags: [debugging, data-quality, labels, train-test-split, feature-engineering, ml-engineering]
xp: 45
hot2026: false
---

You've been staring at your model for three hours. Val loss is stubbornly high. You've swapped in a transformer backbone, bumped hidden units to 512, tried four optimizers, added a scheduler. Nothing moves the needle.

Your senior colleague sits down, opens your notebook, and runs exactly five checks — none of them touch the model. In twelve minutes she finds the bug: a datetime feature was computed from the current timestamp at inference time, not from the event timestamp. The model had been training on the future.

She closes your notebook and says: "Always debug in order."

That order is: **data → labels → splits → features → model**.

## Why the order matters

When something breaks, the temptation is to poke the most interesting, most visible part — the model. That's also the *last* place the bug usually lives.

Think of the pipeline as a chain. A broken link early in the chain makes every link downstream look broken too. Replacing links at the end fixes nothing. You need to walk the chain from the start.

Debugging from the bottom up (model-first) is expensive. Each experiment takes minutes to hours. Debugging from the top down (data-first) is cheap. A quick `df.head()` costs two seconds.

:::why-prod
In production, a rushed model-first debug cycle burns GPU hours, delays rollouts, and often ships a "fixed" model that is still silently broken. Disciplined top-down debugging is a direct cost-saver — and the thing that separates senior ML engineers from everyone else.
:::

## The five-layer checklist

Run these in sequence. Stop the moment you find a defect and fix it before moving down.

:::table {title="Debugging order — one layer at a time"}
| Layer | What you check | Cheap signal |
|---|---|---|
| 1. Raw data | Shape, dtypes, null %, value ranges, duplicates | `df.info()`, `df.describe()`, `df.duplicated().sum()` |
| 2. Labels | Class distribution, null labels, obvious noise, annotation drift | `df[target].value_counts()`, spot-check 20 rows |
| 3. Splits | Leakage, class imbalance per split, time-ordering if temporal | Compare target rate across train / val / test |
| 4. Features | NaN propagation, scaling applied consistently, no look-ahead | Assert no nulls after impute; check fit-on-train only |
| 5. Model | Architecture, capacity, regularisation, hyperparams | Only here after all above pass |
:::

### Layer 1 — Raw data

Before you do anything else, open the raw data and look at it with your eyes.

Check shape. Check nulls. Check that dtypes match what you expect (a column that should be `float64` arriving as `object` is a red flag). Check for duplicated rows — they silently inflate your training set and let the model memorise individual examples.

A single corrupted CSV batch can tank a whole experiment. You want to rule this out in under five minutes.

### Layer 2 — Labels

Bad labels are invisible to training loss until it's too late. Ask:

- Is the class distribution what you expect? A fraud dataset where 0.1% is fraud should look like that in your training set too.
- Are any labels null? A row with a null target is quietly dropped by most frameworks — if that drop is systematic (e.g., nulls only appear when the event was negative), you have selection bias.
- Do a random spot-check. Pull 20 rows, look at the raw input and the assigned label, and ask "does this label make sense?" You'll be surprised how often it doesn't.

### Layer 3 — Splits

This is where leakage hides. Before you train a single epoch, verify:

1. Target rate in train and val are similar (big gap = sampling bug or leakage).
2. If your data is temporal, the split respects time — no future rows bleeding into training.
3. The same entity (user, device, transaction) does not appear in both train and val. Entity-level splits matter.

```python {title="Sanity-check your splits in 10 lines" run=false}
import pandas as pd

def split_sanity(train_df, val_df, target_col: str, id_col: str = None):
    # Target rate comparison
    train_rate = train_df[target_col].mean()
    val_rate   = val_df[target_col].mean()
    print(f"Target rate — train: {train_rate:.3f}  val: {val_rate:.3f}")
    if abs(train_rate - val_rate) > 0.05:
        print("WARNING: target rate differs by >5 pp — check your split logic.")

    # Entity leakage check (if an id column is provided)
    if id_col:
        overlap = set(train_df[id_col]) & set(val_df[id_col])
        print(f"Overlapping {id_col}s: {len(overlap)}")
        if overlap:
            print("WARNING: entity leakage detected — same IDs in train and val.")

# Run it before every experiment
split_sanity(train, val, target_col="churned", id_col="user_id")
```

### Layer 4 — Features

Features are where subtle bugs accumulate. The most common: fitting a scaler or imputer on the full dataset before splitting, so val rows influence the transformation. Your val set should be a stranger to every fitting step.

Also check: after all transformations, do any NaNs remain? Most models silently convert NaN to zero or drop the row — neither is what you want.

### Layer 5 — The model (finally)

You've earned the right to look here only after the four layers above pass cleanly. Now check capacity (is the model too small to fit the data?), regularisation (is dropout / weight decay appropriate?), and learning rate schedule.

If everything above is clean and the model still behaves badly, *that* is a real modelling problem. Now it's worth spending GPU time on.

:::gotcha
The most common mistake is jumping to hyperparameter tuning after a single failing experiment. Tuning is only valid when you are certain the data, labels, splits, and features are correct. Otherwise you are optimising a broken pipeline — and the optimum you find is meaningless.
:::

:::interview-line
"When a model underperforms, I debug in order: raw data first, then labels, splits, features, and only then the model — because cheap checks should always come before expensive ones."
:::

:::qa {q="Why check labels before features?"}
Labels are the ground truth the model is trying to predict. If they are wrong or noisy, no feature engineering or model change can compensate — you're training on a corrupted signal. Checking labels early saves you from spending days tuning a model that was never going to learn the right thing.
:::

:::qa {q="How do you verify that splits are leakage-free in a production pipeline?"}
Three checks: compare target rate across splits (a large gap is a red flag), verify temporal ordering if the data has a time dimension, and assert that no entity identifier appears in both train and val sets. These are cheap assertions that can be baked into a unit test and run on every pipeline execution.
:::

:::qa {q="At what point is it actually valid to blame the model architecture?"}
Only after raw data, labels, splits, and features have all been verified clean. Architecture is the last variable to change. Most apparent "model problems" in practice turn out to be data or split problems — the model was correct; the input was broken.
:::

:::drill {type="mcq" q="Your model's validation AUC drops sharply after a pipeline update. Which layer should you investigate FIRST?"}
- [ ] Add more capacity to the model (deeper network, more trees)
- [ ] Tune the learning rate scheduler
- [x] Check whether the raw data or feature computation changed in the pipeline update
- [ ] Switch from cross-entropy to focal loss
:::

:::drill {type="mcq" q="You discover the target rate in your validation set is 8% but in training it is 2%. What is the most likely cause?"}
- [ ] The model is overfitting
- [ ] The learning rate is too high
- [x] The train/val split is not stratified or has a sampling bug
- [ ] The feature scaler was fit on val instead of train
:::

:::key-takeaway
Always debug top-down: data → labels → splits → features → model. Each layer is cheaper than the one below it. Skipping straight to the model is expensive, often misleading, and the surest way to ship a pipeline that is broken in a different place than you think.
:::
