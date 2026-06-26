---
id: bias-variance
track: 03-statistics
title: "Bias–variance + the learning-curve diagnostic"
badge: CORE
minutes: 9
prereqs: []
tags: [bias, variance, learning-curves, overfitting, underfitting, diagnostics, generalization]
xp: 45
hot2026: false
---

It's Sunday night. You tuned a churn model all weekend. Validation AUC: 0.93. You merge, deploy, and treat yourself to biryani. Two weeks later your product manager sends a Slack message with a screenshot and three question marks — the model is confidently wrong on half the new users coming in from a just-launched mobile segment.

You didn't break anything. You ran into bias and variance. Until you can tell them apart, every fix you try is a coin flip.

## The two ways a model can fail

Every prediction error is the sum of three parts:

**Bias** — your model's average error due to wrong assumptions. It consistently misses in the same direction, regardless of which training data you give it. Think of a bowler who always goes left of the stumps: more practice won't help — the stance is wrong.

**Variance** — your model's sensitivity to which exact training examples it saw. Change the training set slightly and predictions swing wildly. This is the bowler who is all over the place; some deliveries are perfect, some are wides.

**Irreducible noise** — randomness baked into the target itself. No model, no matter how good, can learn the unlabelled causes behind a customer's random mood swing.

The relationship is exact:

> **Expected Error = Bias² + Variance + Irreducible Noise**

You can only control the first two. And the cruel part: the fixes for bias and variance often conflict — increasing model capacity lowers bias but raises variance, while adding regularization does the reverse.

:::why-prod
Knowing *which* failure you have determines every next step. More training data shrinks variance but does almost nothing for bias. Adding features helps bias but can hurt variance. Spending a sprint on the wrong lever is a real, common, expensive mistake in production ML.
:::

## Reading a learning curve

A learning curve answers a simple question: "what happens to train error and validation error as I feed the model more data?"

It is the single fastest diagnostic you have. Plot it before you touch a hyperparameter.

:::table {title="Learning curve patterns — what they tell you"}
| What you see | Train error | Val error | Diagnosis | What to try |
|---|---|---|---|---|
| Both curves high, close together | High | High | **High bias** (underfitting) | More features, less regularization, deeper/wider model |
| Big gap, train stays low | Low | High | **High variance** (overfitting) | More data, stronger regularization, simpler model |
| Both low, gap tiny | Low | Low | Healthy | Ship it |
| Both high, gap never closes with more data | High | High | Irreducible noise or wrong model family | Re-examine labels, feature engineering |
:::

The key insight: if you have **high bias**, adding more data won't close the gap — both curves converge, just at a high error. Only a more expressive model or better features will help. If you have **high variance**, adding data *will* close the gap because the model has more examples to generalise from.

```python {title="Plot a learning curve with scikit-learn" run=false}
# pip install scikit-learn matplotlib  — free, runs locally, no GPU needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    clf, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),   # 10% to 100% of training data
    cv=5,                                      # 5-fold cross-validation
    scoring="roc_auc",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.figure(figsize=(8, 4))
plt.plot(train_sizes, train_mean, label="Train AUC",      color="steelblue")
plt.plot(train_sizes, val_mean,   label="Validation AUC", color="tomato")
plt.fill_between(train_sizes,
                 train_mean - train_scores.std(axis=1),
                 train_mean + train_scores.std(axis=1),
                 alpha=0.1, color="steelblue")
plt.fill_between(train_sizes,
                 val_mean - val_scores.std(axis=1),
                 val_mean + val_scores.std(axis=1),
                 alpha=0.1, color="tomato")

plt.xlabel("Training set size")
plt.ylabel("ROC-AUC")
plt.title("Learning Curve — bias vs variance at a glance")
plt.legend()
plt.tight_layout()
plt.show()

# High bias  → both lines plateau early at roughly the same (low) value
# High variance → persistent gap: train AUC high, val AUC trails behind
# Healthy     → gap closes and both reach a high value as data grows
```

:::gotcha
People often treat "add more data" as the universal cure for a bad model. It isn't. If your learning curve shows train and validation AUC converging at 0.71 with 100k samples, adding another 100k samples will give you 0.71 with more confidence — not 0.85. That plateau means bias. Only a better model or better features gets you out. Check the curve before you go hunting for data.
:::

:::war-story {title="The mobile-segment blindspot"}
A Pune-based fintech team trained their lead-scoring model exclusively on desktop web users — the cohort they had clean labels for. Offline AUC was 0.91. After launching a mobile app, the sales team started complaining within a month. Someone finally plotted the learning curve *per channel*. On mobile users the curve showed a persistent high-variance gap: the model had never seen the behavioural patterns of that segment, and the small mobile sample in the training set wasn't enough to represent it. The fix was targeted data collection and a separate lightweight model for mobile, not hyperparameter tuning. One learning curve plot saved weeks of misdirected effort.
:::

:::interview-line
"A single validation metric can't tell bias from variance — learning curves split them apart, and that's always my first diagnostic step before touching a hyperparameter."
:::

:::qa {q="What is the bias–variance tradeoff in plain terms?"}
Bias is consistent, systematic wrongness — the model keeps missing in the same direction because it's too simple to capture the true pattern. Variance is inconsistency — the model is over-tuned to its training data and falls apart on anything new. You usually can't eliminate both at once: making a model more complex lowers bias but raises variance. The tradeoff is choosing where to sit on that spectrum for your problem.
:::

:::qa {q="My validation loss is much higher than training loss. What are my options and how do I pick?"}
That gap signals high variance. The highest-leverage fix — if you can — is more training data, because it directly reduces how much any individual example influences the model. If data is expensive, add regularization: weight decay, dropout, or early stopping. If neither is feasible, simplify the model (fewer layers, shallower trees, lower polynomial degree). A learning curve tells you which matters most: if the gap shrinks as training size grows, invest in data; if it stays constant, focus on regularization or architecture.
:::

:::qa {q="How do you know when your model has high bias versus high variance?"}
Plot a learning curve — training error and validation error as a function of training set size. High variance shows as a large, persistent gap between the two curves: the model fits training data well but generalises poorly. High bias shows as both curves converging at a high error, regardless of how much data you add. That plateau is the tell: more data won't help when bias is the bottleneck.
:::

:::drill {type="mcq" q="A model's train AUC is 0.96 and its 5-fold cross-validation AUC is 0.74. Which diagnosis fits best?"}
- [ ] High bias — the model is too simple to fit the data
- [x] High variance — the model is overfitting the training set
- [ ] Irreducible noise — the labels are too noisy to learn from
- [ ] Data leakage — the validation set has been contaminated
:::

:::drill {type="mcq" q="You plot a learning curve and see that training AUC and validation AUC both plateau around 0.68, even after doubling your training set to 500k rows. What should you try next?"}
- [ ] Collect more training data — the plateau will break with enough examples
- [ ] Add dropout — the model is clearly overfitting
- [x] Add new features or use a more expressive model — the plateau signals high bias
- [ ] Reduce the learning rate — the optimizer has not converged
:::

:::widget {name="biasvar"}
:::

:::key-takeaway
Bias and variance are the two fundamental failure modes of any ML model. A learning curve — train error vs. validation error across training set sizes — is the fastest way to tell them apart and choose the right fix.
:::
