---
id: distribution-shift
track: 03-statistics
title: "Detecting distribution shift: PSI, KS, Wasserstein, MMD"
badge: CORE
minutes: 9
prereqs: []
tags: [monitoring, statistics, drift-detection, mlops]
xp: 45
hot2026: false
---

It's a Tuesday afternoon and your product manager pings you: "The recommendation model is behaving weird — CTR dropped 18% but nothing changed in the code." You check the model weights. Same. You check the serving pipeline. Fine. Then you look at the *input features*. The age distribution of new users shifted. Last year's teenagers are now young adults. Your training data had almost none of them.

This is distribution shift. And it will happen to every model you ever deploy.

## What distribution shift actually is

When you train a model, you learn a mapping from *this particular distribution of inputs* to outputs. Months later, the world moves on. Users change. Sensors drift. Business rules update. The inputs your model sees in production no longer look like training data.

Three flavors worth knowing:

- **Covariate shift**: The features `X` change, but the true relationship `P(Y|X)` stays the same. Most common.
- **Label shift**: The label distribution `P(Y)` changes. Rarer, but brutal for imbalanced classifiers.
- **Concept drift**: The actual relationship between features and labels changes. Most dangerous — nothing fixes this except retraining.

Your monitoring job: catch covariate shift *before* it causes label shift, *before* it causes user-facing failures.

:::why-prod
An undetected shift that runs for weeks is a silent model decay. By the time accuracy metrics drop, the model has been confidently wrong for a long time — and fixing it means retraining on stale data. Early detection lets you retrain proactively, not reactively.
:::

## The four detectors you need to know

Each metric approaches the problem differently. None is universally best — pick based on your data type and tolerance for false alarms.

:::table {title="Distribution shift detectors at a glance"}
| Method | What it measures | Best for | Alert threshold | Caveat |
|---|---|---|---|---|
| PSI | % divergence across histogram buckets | Single continuous or categorical feature, tabular data | PSI > 0.2 = significant | Sensitive to bin choice |
| KS Test | Max gap between two empirical CDFs | Single continuous feature, small samples | p-value < 0.05 | Assumes independence; weak in high dimensions |
| Wasserstein | Minimum "cost" to reshape one distribution into another | Continuous features where magnitude of shift matters | Domain-specific; normalize first | Scale-sensitive |
| MMD | Distance between distributions in kernel space | Multivariate features, embedding vectors | Domain-specific | Computationally heavier |
:::

### PSI — the industry workhorse

PSI was invented by the banking industry to track whether a borrower population had drifted between scorecard development and production. It buckets your feature into bins, compares the frequency in each bucket between training and production, and penalises large swings.

Formula: `PSI = Σ ((%actual − %expected) × ln(%actual / %expected))`

Rules of thumb: PSI < 0.1 is stable. 0.1–0.2 is worth watching. > 0.2 means investigate now.

### KS Test — the statistician's check

The Kolmogorov-Smirnov two-sample test finds the *maximum vertical distance* between two empirical CDFs. You get a statistic and a p-value. A small p-value means the two samples probably do not come from the same distribution. It is fast, needs no binning, and works well on small samples.

### Wasserstein — the magnitude-aware option

Also called Earth Mover's Distance. Imagine pile A of sand and pile B — how much work does it take to physically reshape one into the other? That is the Wasserstein distance. Unlike PSI, it needs no bins. Unlike KS, it captures *how far apart* the distributions are, not just whether they differ. A tiny KS p-value on a trivial shift can trigger false alarms; Wasserstein stays proportional.

### MMD — for embeddings and multivariate blobs

Maximum Mean Discrepancy maps both distributions into a high-dimensional kernel feature space and measures the distance between their means there. It handles multivariate inputs naturally — making it the right tool when you want to monitor a 768-dimensional sentence embedding vector or a bundle of correlated input features together.

```python {title="All four detectors in ~30 lines" run=false}
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(42)
train_ages = rng.normal(loc=28, scale=5, size=5000)   # training distribution
prod_ages  = rng.normal(loc=33, scale=6, size=5000)   # users got older in prod

# --- PSI ---
def compute_psi(expected, actual, n_bins=10):
    # Use training quantiles for bin edges — never fixed-width on skewed data
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_pct = np.clip(exp_pct, 1e-6, None)   # avoid log(0)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

psi = compute_psi(train_ages, prod_ages)
print(f"PSI:         {psi:.3f}")          # ~0.25 → significant shift

# --- KS Test ---
ks_stat, ks_p = stats.ks_2samp(train_ages, prod_ages)
print(f"KS stat:     {ks_stat:.3f}, p={ks_p:.4f}")

# --- Wasserstein ---
w_dist = wasserstein_distance(train_ages, prod_ages)
print(f"Wasserstein: {w_dist:.3f}")       # ~5 (same units as your feature)

# --- MMD (linear kernel — swap for RBF on embeddings) ---
def linear_mmd(X, Y):
    # E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')] with linear kernel simplifies to:
    return float(abs(np.mean(X) - np.mean(Y)))

mmd = linear_mmd(train_ages, prod_ages)
print(f"MMD linear:  {mmd:.3f}")
# For real embedding drift, use: pip install alibi-detect  (free, local)
# from alibi_detect.cd import MMDDrift
```

:::gotcha
PSI is calibrated for decile buckets — that is, 10 bins defined by the *training data's* quantiles. If you use fixed-width bins on a skewed feature (say, income), sparse high-value buckets will dominate the score and generate constant false alarms. Always compute bin edges from the training set once and reuse them in production.
:::

:::interview-line
"We run PSI for tabular features on a daily schedule — ops teams already know the 0.2 threshold — and MMD for our embedding columns where features are correlated and binning makes no sense."
:::

:::qa {q="What is the difference between PSI and the KS test, and when would you prefer one over the other?"}
PSI summarises drift across the full distribution in one interpretable number using bucket comparisons — it is easy to explain to a business stakeholder and has agreed-upon thresholds from the credit-scoring world. KS gives you a formal p-value and is better for small samples or when you want rigorous false-positive control. In practice, use PSI on a dashboard for at-a-glance health, and KS as an automated alert gate where you need a statistical test rather than a rule of thumb.
:::

:::qa {q="Why does detecting feature drift matter if your accuracy metric has not moved yet?"}
Accuracy is a lagging indicator — it only reacts after ground truth labels arrive, which can take days or weeks in many products. Feature drift is a leading indicator. If the input distribution has shifted, the model is already extrapolating outside its learned region, making low-confidence predictions, even before accuracy degrades. Catching drift early lets you retrain proactively or at least flag uncertain outputs before users notice anything wrong.
:::

:::drill {type="mcq" q="Your daily PSI monitor shows a score of 0.15 for the 'days_since_last_purchase' feature. What is the right next step?"}
- [ ] Page the on-call engineer immediately — this is a critical incident
- [x] Flag it for investigation, check for upstream data changes, and schedule closer monitoring — 0.1–0.2 is a moderate warning, not an emergency
- [ ] Retrain the model immediately without further investigation
- [ ] Ignore it — any PSI under 0.2 is always safe to dismiss
:::

:::drill {type="mcq" q="You need to monitor drift in a 768-dimensional sentence embedding that feeds a ranking model. Which metric fits best?"}
- [ ] PSI — bin each of the 768 dimensions separately and average the scores
- [ ] KS test — run it once on the L2 norm of the embedding vector
- [x] MMD with an RBF kernel — it compares the full multivariate distribution in kernel space without requiring binning or a single summary statistic
- [ ] Wasserstein — it is designed for multivariate distributions and handles high dimensions well
:::

:::key-takeaway
Use PSI for fast, interpretable tabular feature monitoring (alert at 0.2), KS for statistical confirmation on single continuous features, Wasserstein when the magnitude of shift matters, and MMD when you are monitoring multivariate or embedding inputs. Always use at least two metrics together — each has blind spots the other catches.
:::
