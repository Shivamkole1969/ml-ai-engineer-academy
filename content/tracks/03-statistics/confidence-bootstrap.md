---
id: confidence-bootstrap
track: 03-statistics
title: "Confidence intervals & the bootstrap"
badge: CORE
minutes: 8
prereqs: []
tags: [statistics, confidence-interval, bootstrap, evaluation, metrics]
xp: 45
hot2026: false
---

Your model scores 87% F1 on the test set. The product manager asks: "Great — how confident are
you in that number?" You say: "Very confident." She says: "Can you put a range on it?"

Silence.

That silence is what this lesson fixes. A single point metric — 0.87 — is a guess dressed up as
a fact. A **confidence interval** (CI) says "the true performance is probably between 0.84 and
0.90." That's the number your stakeholder actually needs.

## What a confidence interval actually means

A 95% CI does NOT mean "there's a 95% chance the true value is in this range." That's the most
common mistake in interviews.

The correct interpretation: if you repeated your evaluation procedure many times on different
samples from the same population, **95% of the CIs you'd compute would contain the true value**.
It's a statement about the procedure, not about this particular interval.

In plain terms: the interval tells you how much your metric would wiggle if you tested on a
slightly different batch of users. A tight CI means your test set is large enough to be trusted.
A wide CI means your 87% could easily be 82% in production — and that's a problem.

:::why-prod
Shipping a model without a CI is like quoting a delivery date without a buffer. In production,
your test set is always finite, often biased, and rarely the same distribution as live traffic.
A CI forces you to acknowledge that uncertainty before go-live — not after your first oncall page.
:::

## The parametric way (and why it breaks)

For accuracy on a binary classifier, there's a textbook formula using the **normal approximation**:

$$CI = \hat{p} \pm z \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

Where `p̂` is your observed accuracy, `n` is test-set size, and `z = 1.96` for 95%.

This is fine for accuracy on a large, balanced dataset. But it falls apart fast:
- F1, AUC, NDCG — no clean formula exists.
- Class imbalance — the normal approximation goes wrong for very small or large `p̂`.
- Composite metrics — anything that combines precision and recall in a non-linear way.

This is where **bootstrap** comes in.

## The bootstrap: just resample and measure

The idea is beautifully simple. You already have a test set. Treat it as your universe. Now:

1. Draw `n` samples from it **with replacement** (some rows appear twice; some not at all).
2. Compute your metric on that resample.
3. Repeat 1000–2000 times.
4. Take the 2.5th and 97.5th percentiles of those scores → your 95% CI.

No distributional assumptions. Works for any metric. Takes ten lines of code.

```python {title="Bootstrap CI for any metric" run=false}
import numpy as np
from sklearn.utils import resample

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=2000, ci=0.95):
    """
    Compute a bootstrap confidence interval for any metric.
    metric_fn: callable(y_true, y_pred) -> float
    Run locally: pip install numpy scikit-learn
    """
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(seed=42)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)          # resample with replacement
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return lower, np.mean(scores), upper


# --- Example: F1 on a small imbalanced test set ---
from sklearn.metrics import f1_score

np.random.seed(0)
y_true = np.random.choice([0, 1], size=300, p=[0.85, 0.15])   # 15% positive class
y_pred = y_true.copy()
flip = np.random.rand(300) < 0.12
y_pred[flip] = 1 - y_pred[flip]                                # ~88% accuracy

lo, mean, hi = bootstrap_ci(y_true, y_pred, f1_score)
print(f"F1: {mean:.3f}  |  95% CI: [{lo:.3f}, {hi:.3f}]")
# Typical output: F1: 0.714  |  95% CI: [0.598, 0.821]
# Notice: the CI is wide! 300 samples + imbalance = high uncertainty.
```

The wide interval on 300 samples is the lesson: "87% F1" means very little on a small test set.
You need to show the CI before you can trust the number.

:::table {title="Parametric vs Bootstrap at a glance"}
| | Parametric formula | Bootstrap |
|---|---|---|
| Works for F1 / AUC / NDCG? | No | Yes |
| Requires distributional assumptions? | Yes (normal approx.) | No |
| Handles class imbalance well? | Poorly | Yes |
| Lines of code | 1 | ~15 |
| Compute cost | Instant | 1–5 seconds for 2000 reps |
:::

## How many test samples do you actually need?

A rough rule: you want your CI width to be less than the difference that matters to your business.
If 2% F1 improvement is meaningful, you probably need 1000+ test examples to see it clearly.
Fewer than 200 test samples and almost any CI will be too wide to draw conclusions.

:::gotcha
Bootstrapping your **training** loss or **validation** set is misleading — your model was
optimized on data from that same distribution, so the bootstrap CI will be optimistically tight.
Always bootstrap on a **held-out test set** the model has never touched. And never tune
hyperparameters, then bootstrap the same set you tuned on — that's double-dipping.
:::

:::interview-line
"A point metric without a confidence interval is a guess. I always bootstrap my test metrics so
I can tell stakeholders how much that number might move in production."
:::

:::qa {q="What does a 95% confidence interval mean?"}
It means the procedure that generated this interval will capture the true parameter value 95% of
the time across repeated experiments. It does not mean there is a 95% probability the true value
is inside this specific interval — the true value is fixed; only the interval varies across
experiments.
:::

:::qa {q="Why prefer bootstrap over a normal-approximation CI for F1 score?"}
The normal approximation requires a closed-form formula and assumes the sampling distribution is
approximately normal. F1 is a ratio of precision and recall — it has no such formula, and its
distribution is skewed, especially under class imbalance. Bootstrap makes no distributional
assumptions and works for any metric you can compute.
:::

:::qa {q="How many bootstrap resamples should you use?"}
1000 is usually sufficient for a 95% CI. Go to 2000–5000 if you need the tails to be stable
(e.g., 99% CI) or if your metric is noisy (like NDCG on sparse rankings). Beyond 5000 you rarely
get meaningful improvement in CI stability.
:::

:::drill {type="mcq" q="Your test set has 150 examples. You compute 95% CI via bootstrap and get F1 = 0.78 [0.61, 0.91]. What should you do?"}
- [ ] Ship the model — 0.78 is above the 0.75 threshold.
- [ ] Retrain on more data until the point estimate improves.
- [x] Collect more labeled test data to narrow the CI before making a ship/no-ship call.
- [ ] Switch to parametric CI — it will give a tighter interval.
:::

:::drill {type="mcq" q="Which of the following is the correct interpretation of a 95% CI [0.82, 0.90]?"}
- [ ] There is a 95% probability the true F1 is between 0.82 and 0.90.
- [ ] 95% of your test examples have F1 between 0.82 and 0.90.
- [x] If you repeated this evaluation procedure many times, 95% of the resulting intervals would contain the true F1.
- [ ] The model will score between 0.82 and 0.90 on 95% of production requests.
:::

:::drill {type="mcq" q="A teammate bootstraps model accuracy using the validation set that was used for early stopping. What is wrong?"}
- [ ] Nothing — validation set is the right set to evaluate on.
- [ ] The CI will be too wide because validation sets are small.
- [x] The CI will be overoptimistic because the model was implicitly optimized on this data.
- [ ] Bootstrap cannot be applied to accuracy, only to F1.
:::

:::key-takeaway
A single metric number is always an estimate. Use bootstrap to attach a confidence interval to
any metric — it takes 15 lines of code and turns a point guess into a defensible, quantified
claim you can actually show a stakeholder.
:::
