---
id: splitting-strategy
track: ch08-validation-metrics
title: "Splitting strategy (and how it leaks)"
badge: CORE
minutes: 9
prereqs: []
tags: [validation, train-test-split, data-leakage, cross-validation, time-series]
xp: 45
hot2026: false
---

Your model hits 97% accuracy on the test set. You do a happy dance. You ship it. Two weeks later, your product manager asks why the model "barely does anything useful in the real world." You pull the logs and realize — the model was cheating the whole time. It saw the future during training, and you had no idea.

This lesson is about making sure that never happens to you.

## The Three-Way Split

Every ML project needs three separate slices of your data, each with a different job:

- **Train** — what the model learns from
- **Validation** — what *you* learn from during development (tune hyperparameters, compare models)
- **Test** — a final, one-time sanity check you run *once*, right before deployment

The test set is sacred. If you peek at it to improve the model, you've contaminated it. At that point it's just another validation set, and you no longer have a reliable signal for real-world performance.

:::why-prod
In production, "how well did it do on training data?" is irrelevant — the world sends new data every second. Your validation/test split is your only proxy for production performance before you actually ship. Get it wrong and every model decision you make is based on a lie.
:::

:::table {title="Split roles at a glance"}
| Split | Used by | Used when | Typical size |
|---|---|---|---|
| Train | Model | Every training run | 60–80 % |
| Validation | You (engineer) | Iterative tuning | 10–20 % |
| Test | Stakeholders / you (once) | Final go/no-go | 10–20 % |
:::

## Splitting Sounds Simple — Here's Where It Gets Tricky

For a classic static dataset (think tabular customer records), a random split usually works. But two patterns silently wreck your evaluation all the time.

### Pattern 1: Temporal leakage

Imagine you're predicting whether a loan will default. Your dataset has rows from January through December. If you shuffle and split randomly, your training data will include rows from November, and your test data will include rows from January. The model is, in effect, learning from the future to predict the past.

In time-based problems — fraud detection, demand forecasting, churn prediction — **always split by time, never by random shuffle**. Train on earlier data, validate and test on later data.

```python {title="Time-based split (no leakage)" run=false}
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Assume df is sorted by date ascending
df = df.sort_values("event_date").reset_index(drop=True)

# Simple cut: 70% train, 15% val, 15% test
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

# For cross-val on time-series, use TimeSeriesSplit:
# tscv = TimeSeriesSplit(n_splits=5)
# for fold_train_idx, fold_val_idx in tscv.split(train):
#     ...
```

### Pattern 2: Group leakage

Suppose you're predicting readmission risk for hospital patients, and you have multiple records per patient. A random split will almost certainly put some records of the same patient in both train and test. Your model learns the patient's history in training and then "predicts" for that same patient — which is easy, because it already knows them.

Fix it with a **group split**: keep all records for a given entity (user, patient, session, device) in one partition only.

```python {title="Group split to avoid entity leakage" run=false}
from sklearn.model_selection import GroupShuffleSplit

# 'patient_id' is the grouping key
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["patient_id"]))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
# No patient appears on both sides — clean evaluation.
```

:::gotcha
The most common form of leakage is **preprocessing before splitting**. If you fit a scaler or imputer on the full dataset and *then* split, your validation data's statistics have already leaked into your training pipeline. Always split first, then fit transformers on train only, then apply (transform) to val/test. Wrap this in a `sklearn.pipeline.Pipeline` so it's enforced automatically.
:::

## Cross-Validation: When You Don't Have Enough Data

When your dataset is small, holding out 20% for validation wastes training signal. K-fold cross-validation is the answer: split the data into *k* folds, rotate which fold is validation each time, and average the scores. You get a much more reliable estimate without sacrificing a fixed chunk of data.

Use `StratifiedKFold` (classification) to ensure each fold has the same class ratio. Use `TimeSeriesSplit` for sequential data. Never use plain `KFold` on imbalanced or temporal data.

:::interview-line
"I always split first, preprocess second — that single rule eliminates 80% of the data-leakage bugs I've seen in production codebases."
:::

:::qa {q="Why is a separate test set necessary if we already use a validation set?"}
Validation is used iteratively — you see its scores many times and subtly overfit your choices to it. The test set gives an unbiased final estimate because you use it exactly once, after all decisions are made. Without that separation, your reported performance is optimistic and not representative of what users will actually experience.
:::

:::qa {q="How do you handle splitting for a time-series forecasting problem?"}
You split by time, never randomly. Train on the oldest data, validate on the next window, and test on the most recent window. For cross-validation, use a walk-forward or expanding-window approach (sklearn's `TimeSeriesSplit`) so the model never sees future data to predict the past.
:::

:::drill {type="mcq" q="You're building a churn model using two years of subscription data. You randomly shuffle and split 80/20. What is the most serious problem?"}
- [ ] The training set is too large and will cause overfitting
- [ ] Random splits always produce imbalanced class ratios
- [x] Future data can appear in training and past data in the test set, inflating performance estimates
- [ ] 80/20 is not a valid split ratio for subscription data
:::

:::drill {type="mcq" q="You fit a StandardScaler on the entire dataset before splitting into train and test. What is the consequence?"}
- [ ] The model will train faster but generalize poorly
- [ ] The scaler will fail to converge without enough data
- [x] Test-set statistics (mean, std) leak into the training pipeline, making evaluation over-optimistic
- [ ] There is no consequence; scaling is invariant to split order
:::

:::key-takeaway
Split first, preprocess second, and always respect the structure of your data — time-based data needs time-based splits, entity-based data needs group splits. Violating these rules produces metrics that lie, and the lie doesn't show up until production.
:::
