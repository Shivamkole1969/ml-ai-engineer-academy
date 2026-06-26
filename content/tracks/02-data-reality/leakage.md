---
id: leakage
track: 02-data-reality
title: "Leakage — the silent score inflator + the one-sentence test"
badge: HOT
minutes: 10
prereqs: []
tags: [leakage, feature-engineering, train-test-split, data-quality, interviews]
xp: 60
hot2026: true
---

It's 4 PM on a Friday. You demo your new fraud-detection model to the product lead. AUC: 0.97. Everyone's excited. You deploy it over the weekend. By Monday morning the AUC in production is 0.61 — barely better than flipping a coin.

You haven't overfit. Your code is clean. The data pipeline is running fine.

You have a leakage problem, and it has been lying to you from day one.

## What leakage actually is

Leakage means information that *wouldn't exist at prediction time* has sneaked into your training features. The model learns a shortcut that works in the past but evaporates the moment you go live.

Two flavours:

**Target leakage** — a feature encodes the answer. Classic example: you're predicting whether a loan defaults, and one of your features is "loan written off (Y/N)". That column is filled in *after* the default happens. The model memorises it instantly. Validation looks incredible. Production is a disaster.

**Train-test contamination** — you touch the test set before the split. You fit a `StandardScaler` on the full dataset, then split. The test set has already influenced the scaler's mean and variance. Your "held-out" set is no longer held out.

:::why-prod
In production, your model receives features computed *at request time*. If any training feature was computed using information from after that moment, the model learned a signal that will never appear in production. Every percentage point of AUC gain from leakage is a lie you'll pay for in oncall pages.
:::

## The one-sentence test

Before adding any feature, ask yourself:

> **"If I were serving this prediction at exactly the moment it's needed, would I *already have* this feature value — without looking at anything that happens later?"**

If the answer is "no" or "maybe", the feature leaks. Delete it. This one question has saved more ML projects than any regularisation trick.

:::table {title="Leakage — quick field guide"}
| Scenario | Leaks? | Why |
|---|---|---|
| User's *historical* purchase count | No | Past data, known at predict time |
| User's purchase count *for the current month* | Yes | Month isn't over yet |
| StandardScaler fit on train+test together | Yes | Test stats contaminate scaler |
| StandardScaler fit on train only, applied to test | No | Correct pipeline |
| "Days since last repayment" for a loan default model | Depends | OK if computed up to prediction date only |
| "Claim approved (Y/N)" as feature for claim fraud | Yes | Approval happens *after* the fraud label is set |
:::

## The contamination trap in sklearn

This is the most common form beginners get wrong — and it looks completely innocent.

```python {title="Leaky vs correct pipeline" run=false}
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = pd.read_csv("transactions.csv").drop("fraud", axis=1), \
       pd.read_csv("transactions.csv")["fraud"]

# ❌ WRONG — scaler sees test data before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # leaks test stats into scaler
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ✅ RIGHT — use a Pipeline; fit happens only on train fold
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression()),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)                  # scaler fitted on train only
print(pipe.score(X_test, y_test))           # honest evaluation

# Same principle applies inside cross_val_score — Pipeline handles it automatically
# Run locally: pip install scikit-learn pandas
```

The fix is always the same: use `sklearn.pipeline.Pipeline`. Every transformer gets fitted inside the fold, never on the full dataset.

:::gotcha
Time-series data breaks random splits entirely. If your data has a time dimension, always split chronologically — past rows for training, future rows for testing. A random 80/20 split on time-series data will leak future patterns into training on every single fold.
:::

:::war-story {title="The 'days since signup' feature that cost two sprints"}
A team in Bangalore built a churn prediction model for a SaaS product. Their best feature was `days_since_signup` — computed as *today's date minus signup date* at the time they built the dataset. AUC was 0.91. In production the same feature was computed the same way, so what went wrong? They'd trained on historical data where "today" was six months ago. Long-tenure users who *had already churned* looked like low-risk users because their `days_since_signup` was large. The model had absorbed the implicit assumption that long tenure = loyal customer — but the label was whether they churned *in the next 30 days*, and older users were over-represented in the "already churned" bucket. It took two sprints to re-engineer the feature using a proper temporal split.
:::

:::interview-line
"I apply the one-sentence test to every feature: would this value be available at exact prediction time, with zero knowledge of the future? If not, it leaks."
:::

:::qa {q="What is data leakage and how do you detect it?"}
Leakage is when information unavailable at prediction time bleeds into training features, inflating offline metrics. I detect it with the one-sentence test — "would I have this value at serve time?" — and by watching for suspiciously high metrics, features with implausibly high importance, or transformers fitted on the full dataset before splitting.
:::

:::qa {q="How does using sklearn's Pipeline prevent leakage?"}
Pipeline chains transformers and estimators so that `fit` on any transformer is called only on the training fold, never on held-out data. When you call `cross_val_score(pipe, X, y)`, each fold correctly refits the scaler on that fold's training slice alone. Without Pipeline, it's easy to accidentally call `fit_transform` on the whole dataset before splitting.
:::

:::qa {q="You get 0.96 AUC in validation but 0.58 in production. What's your first hypothesis?"}
Leakage. I'd audit every feature for future information, check whether any preprocessing step (scaling, encoding, imputation) was fitted on the full dataset, verify the train-test split is chronologically correct for time-series data, and look for any column derived from the target variable. That gap is too large to be normal overfitting.
:::

:::drill {type="mcq" q="Which of the following is a clear example of target leakage?"}
- [ ] Using a user's age as a feature to predict subscription cancellation
- [x] Using "refund requested (Y/N)" as a feature in a model that predicts whether an order will be returned
- [ ] Fitting StandardScaler on training data only before transforming test data
- [ ] Using last month's purchase count to predict this month's churn
:::

:::drill {type="mcq" q="A teammate fits a StandardScaler on the full dataset, then does an 80/20 train-test split, then trains a model. What is the problem?"}
- [ ] There is no problem — StandardScaler is deterministic so split order doesn't matter
- [ ] The model will underfit because normalization reduces variance
- [x] The scaler has seen test-set statistics during fit, so the evaluation is optimistically biased
- [ ] The 80/20 split ratio is too small for StandardScaler to work correctly
:::

:::drill {type="mcq" q="You're building a loan default predictor. Your dataset covers 2019–2023. Which split strategy is correct?"}
- [ ] Random 80/20 split — more data in training gives better generalisation
- [ ] Stratified split on the default label to balance classes
- [x] Train on 2019–2021, validate on 2022, test on 2023 — strict chronological split
- [ ] K-fold cross-validation with k=10 for the most reliable estimate
:::

:::key-takeaway
Before every feature ask "would I have this at prediction time?" — and use sklearn Pipelines so your preprocessors are never fitted on held-out data. Leakage makes offline metrics a lie; catching it early is what separates engineers who ship reliable models from those who get paged at 2 AM.
:::
