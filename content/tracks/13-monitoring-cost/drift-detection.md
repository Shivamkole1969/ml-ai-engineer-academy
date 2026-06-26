---
id: drift-detection
track: 13-monitoring-cost
title: "Drift detection in production (reference → compare → alert)"
badge: CORE
minutes: 9
prereqs: []
tags: [drift, monitoring, statistics, data-quality, mlops, psi, ks-test]
xp: 45
hot2026: false
---

You deployed a churn model six months ago. Solid — 85% precision on holdout. Product is happy, sales team trusts it. Then one Tuesday, a senior PM pings you: "The model's been flagging all the enterprise accounts as safe. Something feels wrong." You dig in. Nothing changed in your code. But two months ago a new pricing tier rolled out, and the `contract_value` feature quietly shifted from a log-normal to a bimodal distribution. Your model never saw that shape. It has been confidently wrong ever since — and you had no alert.

This is drift. And catching it is what separates ML engineers from ML experimenters.

## What drift actually means

There are three flavours worth knowing.

**Data drift** — the statistical distribution of your *inputs* changes after deployment. The model sees different-looking data than it trained on. Detectable without ground truth.

**Concept drift** — the *relationship* between inputs and the label changes. Even if inputs look fine, the right answer has shifted (fraud patterns evolve, customer behaviour changes). Requires labels to confirm — which often arrive late.

**Prediction drift** — your model's *output distribution* shifts. Often a downstream symptom of the first two, and the earliest real-time signal you can watch cheaply.

In practice: watch inputs obsessively, watch outputs as a second signal, and accept that concept drift will only become obvious when accuracy numbers finally land.

:::why-prod
Ground-truth labels arrive days or weeks late in most production systems — sometimes never. Drift detection on inputs lets you raise a flag *before* the labels confirm damage, cutting the blast radius from weeks to hours.
:::

## The reference → compare → alert pattern

Three moving parts — that is all it takes.

1. **Reference window** — a frozen snapshot of healthy data. Usually your training distribution or a manually curated "golden" production window. Fixed. Never rolling.
2. **Comparison window** — a rolling slice of live production traffic. Last 24 hours, last 7 days — choose based on your daily request volume. Aim for at least a few thousand samples or your tests become noisy.
3. **Alert** — when the two windows diverge past a threshold, a ticket opens or a channel ping fires.

The key word is *frozen*. If your reference drifts too, you lose the baseline and the whole system becomes meaningless.

## Choosing a statistical test

Different feature types call for different tests.

:::table {title="Drift detector by feature type"}
| Feature type | Detector | Flag when |
|---|---|---|
| Continuous | KS test (2-sample) | p-value < 0.05 |
| Continuous | PSI | PSI > 0.2 |
| Categorical | Chi-square test | p-value < 0.05 |
| Categorical | Jensen-Shannon divergence | JSD > 0.1 |
| Model output / target | PSI on prediction | PSI > 0.1 |
:::

**KS test** (Kolmogorov-Smirnov): compares the cumulative distribution functions of two samples. Fast, non-parametric, no distribution assumptions. The catch: with large samples, even tiny irrelevant differences become "statistically significant".

**PSI** (Population Stability Index): borrowed from credit risk in banking. Tells you *how much* things shifted, not just whether they did. PSI < 0.1 is fine. 0.1–0.2 is a yellow flag. Above 0.2, take action.

Use both together: KS for a quick yes/no, PSI for the magnitude you put on a dashboard.

```python {title="KS test + PSI on a feature column" run=false}
import numpy as np
from scipy import stats

# --- KS test ---
def ks_drift(reference: np.ndarray, production: np.ndarray, alpha: float = 0.05):
    """Returns (is_drifted, p_value). True = drift detected."""
    _, p_value = stats.ks_2samp(reference, production)
    return p_value < alpha, p_value

# --- PSI ---
def compute_psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. <0.1 OK | 0.1-0.2 watch | >0.2 act."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  -= 1e-6   # ensure all values fall inside bucket edges
    breakpoints[-1] += 1e-6

    ref_counts  = np.histogram(reference,  bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]

    eps = 1e-6   # prevent log(0)
    ref_pct  = ref_counts  / (ref_counts.sum()  + eps)
    prod_pct = prod_counts / (prod_counts.sum() + eps)

    psi = np.sum((prod_pct - ref_pct) * np.log((prod_pct + eps) / (ref_pct + eps)))
    return float(psi)

# --- Simulate a shifted feature ---
rng = np.random.default_rng(42)
ref_data  = rng.normal(loc=50, scale=10, size=5_000)
prod_data = rng.normal(loc=65, scale=15, size=500)   # mean shifted, wider spread

drifted, pval = ks_drift(ref_data, prod_data)
psi_score     = compute_psi(ref_data, prod_data)

print(f"KS drift detected: {drifted}  (p={pval:.4f})")
print(f"PSI: {psi_score:.3f}  -> {'ACT' if psi_score > 0.2 else 'WATCH' if psi_score > 0.1 else 'OK'}")

# Run locally: pip install numpy scipy
```

## Wiring it into your system

The minimum viable drift pipeline:

1. **Log features at inference time.** Write every request's feature vector to a table or object store. No logs = no drift detection, full stop.
2. **Schedule a daily batch job.** Load the last 24 hours of production features, compare to the reference baseline.
3. **Alert on top features only.** Pick your 8–10 most important features by model feature importance. Running tests on all 50 features daily is a fast path to alert fatigue.
4. **Route alerts correctly.** A PSI alert should open a ticket or ping a dedicated channel — not your general on-call pager. It warrants investigation within the workday, not a 3am wake-up.

Open-source options that handle most of this wiring: **Evidently AI** (Python-native, great for notebooks and batch jobs), **WhyLogs** (lightweight logging SDK), **Grafana + custom PSI exporter** if your team already lives in Grafana. All three are free to self-host.

:::gotcha
Using a "rolling 30-day average" as your reference window is a trap. If your data has weekly or seasonal patterns, the reference itself drifts — and you lose the stable baseline you need to compare against. Always anchor to your *training distribution* or a hand-picked healthy production snapshot. Freeze it. Treat it like a golden dataset you never overwrite.
:::

:::interview-line
"We detect drift by comparing a frozen training-distribution baseline to a rolling production window using KS tests for continuous features and PSI for magnitude — alerting only on top features by importance to keep the noise manageable."
:::

:::qa {q="What is the difference between data drift and concept drift?"}
Data drift means your input feature distributions have changed — the model is seeing data that looks different from training. Concept drift means the underlying mapping from inputs to labels has changed — the world's rules shifted even if the inputs look similar. Data drift you can detect without labels using statistical tests on your feature logs. Concept drift requires ground truth, which often arrives far too late to function as an early warning.
:::

:::qa {q="Why use PSI instead of just a KS p-value for production alerting?"}
A p-value only answers "did the distribution change?" — and with enough data, almost everything changes significantly. PSI gives you a *magnitude*: below 0.1 is normal noise, 0.1–0.2 deserves attention, above 0.2 means the model has likely degraded and retraining is warranted. That map from score to action makes PSI operationally useful in an alerting system where you need to decide what to do, not just whether something happened.
:::

:::drill {type="mcq" q="Your model has 45 input features. You want to run daily drift checks without drowning in false alerts. What's the right approach?"}
- [ ] Run KS tests on all 45 features and alert on any single failure
- [ ] Pick 3 features at random and monitor only those
- [x] Monitor the top 8–10 features ranked by model feature importance
- [ ] Only check the output prediction distribution, not the inputs
:::

:::drill {type="mcq" q="You compute PSI for a key continuous feature and get a score of 0.14. What is the correct response?"}
- [ ] Ignore it — PSI below 0.2 means everything is fine
- [ ] Immediately roll back the model to last month's version
- [x] Flag it as a yellow alert, investigate the feature's recent trend, and increase monitoring frequency
- [ ] Retrain the model using only the past 7 days of data
:::

:::key-takeaway
Log your features at inference time, freeze a clean reference baseline, and compare it daily to a rolling production window — alerting on PSI > 0.2 or KS p < 0.05 for your most important features. Without the feature log there is no signal; without the frozen reference there is no ground truth to compare against.
:::
