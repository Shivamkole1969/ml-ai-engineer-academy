---
id: tree-models
track: ch06-classic-ml
title: "Decision trees & random forests"
badge: CORE
minutes: 9
prereqs: []
tags: [decision-tree, random-forest, ensemble, tabular, classification, regression]
xp: 45
hot2026: false
---

Imagine you join a fintech startup. Your first week, someone hands you a fraud detection model — a single decision tree, trained to 98% accuracy on training data. You deploy it. Two days later, the fraud team is furious: the model flags almost every legitimate transaction from a new city and misses half the actual fraud. What happened? The tree memorized the training set perfectly but never learned anything general. You just met overfitting in the wild.

Decision trees and random forests are often the first "real" models people reach for on tabular data. They're fast to train, easy to explain, and — when used correctly — genuinely powerful. But they have a specific failure mode that trips up beginners every time.

## What is a decision tree?

A decision tree learns a series of if-else rules from your data. At each step it asks: "Which single feature and threshold split my data into the most homogeneous groups?" It repeats this greedily until it hits a stopping rule (max depth, min samples, etc.) or has perfectly sorted every training example.

The result looks exactly like a flowchart. "Is age > 35? If yes, go left. Is income > 60k? If yes, predict 'approved'." That legibility is a genuine superpower — stakeholders can audit it, regulators can inspect it, and you can spot nonsense rules instantly.

The catch: trees are high-variance learners. Leave one unchecked and it will grow until every leaf contains a single training sample. It memorises, not generalises.

:::why-prod
In production, a single deep tree is almost always wrong for anything beyond a simple rule extraction task. You need to control depth aggressively, or move straight to an ensemble. Overfitted trees rot fast when data distribution shifts — and data distribution always shifts.
:::

## Key split criteria

:::table {title="Common split criteria at a glance"}
| Criterion | Task | What it minimises |
|---|---|---|
| Gini impurity | Classification | Probability of misclassifying a random sample |
| Entropy / info gain | Classification | Uncertainty (bits) in the resulting groups |
| MSE / MAE | Regression | Squared or absolute error in each leaf |
:::

In practice, Gini and entropy give nearly identical results. Pick Gini as default; it's slightly cheaper to compute.

## Why ensembles: Random forests

A random forest is beautifully simple in concept. You grow many trees, each on a different random bootstrap sample of your training data (bagging). At each split, each tree only considers a random subset of features. Then you average their predictions.

Why does this work? Each tree is a bit different, so its errors are different. Averaging cancels out those random errors. The variance drops dramatically while bias stays low. This is the bias-variance tradeoff at work — you trade a little bias for a lot less variance.

A rough rule for feature subsets: `sqrt(n_features)` for classification, `n_features / 3` for regression.

```python {title="Fit and inspect a random forest" run=false}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd

# Generate a toy dataset — swap in your own DataFrame here
X, y = make_classification(n_samples=10_000, n_features=20, n_informative=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# n_jobs=-1 uses all CPU cores — free speedup
rf = RandomForestClassifier(
    n_estimators=200,      # more trees = more stable, diminishing returns past ~300
    max_depth=10,          # cap depth to prevent overfitting
    min_samples_leaf=5,    # leaf must have ≥5 samples — smooths out noise
    max_features="sqrt",   # random feature subset at each split
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

print(f"Val accuracy: {rf.score(X_val, y_val):.3f}")

# Feature importance — great for a first audit
importances = pd.Series(rf.feature_importances_, name="importance")
print(importances.sort_values(ascending=False).head(10))
```

## Tuning the knobs that actually matter

You do not need to tune every hyperparameter. Focus on three:

1. `n_estimators` — more is almost always better; stop when validation score flatlines.
2. `max_depth` / `min_samples_leaf` — these are your overfitting guards. Start conservative (`max_depth=8`, `min_samples_leaf=10`) and loosen only if training score is also low.
3. `class_weight="balanced"` — essential for imbalanced targets; handles skewed class distributions automatically.

:::gotcha
The default `max_features="sqrt"` is for classification. For regression, switch to `max_features=1.0` or `"sqrt"` and always benchmark both — the default is just a starting point, not a law.

Also: feature importance from random forests is biased toward high-cardinality features (like IDs or continuous counts). Use `permutation_importance` from scikit-learn for a more honest ranking, especially before removing features.
:::

## Out-of-bag score: a free validation signal

When you bootstrap sample for each tree, roughly 37% of rows are left out of each tree's training. Those are the "out-of-bag" (OOB) samples. scikit-learn can use them to estimate generalisation error without a separate validation split.

Set `oob_score=True` and read `rf.oob_score_`. It is not a replacement for proper cross-validation, but it is a fast, nearly-free sanity check — especially handy when data is scarce.

## When to use a tree vs. a forest

Use a single tree when you need a fully interpretable model for a compliance or legal context — and you accept that it may underperform. Use a random forest as your "fast baseline" for any tabular problem before reaching for gradient boosting. It requires less tuning and rarely explodes.

:::interview-line
"A random forest reduces variance by averaging many decorrelated trees; the key knobs are depth, min-samples-leaf, and the feature-subset size at each split."
:::

:::qa {q="Why does a single decision tree overfit so easily?"}
Because the tree greedily splits until every leaf is pure, it ends up memorising the training set rather than learning generalisable patterns. The fix is either to constrain depth / minimum samples per leaf, or to use an ensemble method like a random forest that averages away the variance across many trees.
:::

:::qa {q="What is the out-of-bag error in a random forest?"}
Each tree is trained on a bootstrap sample, so roughly 37% of rows are never seen by that tree. Those rows are used to compute a prediction error for each tree, and averaging these gives the OOB error — an almost-free estimate of generalisation performance that correlates well with held-out validation error.
:::

:::qa {q="How do you handle class imbalance in a random forest?"}
Set `class_weight='balanced'` to re-weight training examples inversely proportional to class frequency. You can also use `class_weight='balanced_subsample'`, which recomputes weights per bootstrap sample. For very extreme imbalance, combine this with oversampling (SMOTE) or adjust the decision threshold on predicted probabilities after training.
:::

:::drill {type="mcq" q="You train a random forest with default settings and get training accuracy 99% but validation accuracy 72%. Which change is MOST likely to help?"}
- [ ] Increase n_estimators from 100 to 500
- [ ] Switch the split criterion from gini to entropy
- [x] Reduce max_depth and increase min_samples_leaf
- [ ] Set max_features=1.0 (use all features at each split)
:::

:::drill {type="mcq" q="What makes individual trees in a random forest 'decorrelated'?"}
- [ ] Each tree is trained on a different algorithm (gini vs entropy)
- [ ] Each tree is trained on the full dataset but with different random seeds
- [x] Each tree sees a random bootstrap sample AND a random feature subset at each split
- [ ] Trees are pruned to different max depths after training
:::

:::key-takeaway
A single tree overfits; a random forest averages many decorrelated trees to dramatically cut variance — your go-to fast baseline for any tabular problem. Control depth and min-leaf-samples before anything else.
:::
