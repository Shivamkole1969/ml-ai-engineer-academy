---
id: calibration
track: 03-statistics
title: "Calibration: reliability diagrams, ECE, temperature/Platt/isotonic"
badge: HOT
minutes: 10
prereqs: []
tags: [calibration, ECE, temperature-scaling, Platt-scaling, isotonic-regression, reliability-diagram, probability]
xp: 60
hot2026: true
---

It's 3 pm on a Friday and your fraud model just auto-blocked a ₹12-lakh wire transfer for a premium business account. The model said **94% fraud**. The customer is on hold. Your manager is pinging you. Your team digs in — and discovers the model yells "94%" on nearly *everything* it flags. The real precision at that threshold? Sixty-one percent.

Your model is not wrong exactly. It's **miscalibrated**. Like a student who always answers "I'm 95% sure" regardless of whether they studied the topic or not.

Calibration means: when the model says "70% confident", it should be right roughly 70% of the time. Not 90%. Not 50%. Seventy. That promise is what makes a probability number mean anything in production.

## What calibration actually means

A model output is only a *score* until it's calibrated — then it becomes a *probability* you can act on, report, and be held accountable for.

A perfectly calibrated model satisfies this:

> Of all the predictions where the model outputs ~0.8, about 80% of them are actually positive.

Neural networks and gradient boosting models are almost always **overconfident** out of the box — they push scores toward 0 and 1 harder than the data justifies. Logistic regression is comparatively well-behaved, but not immune.

:::why-prod
In credit scoring, insurance, medical triage, or any model that surfaces a number to a human (or triggers automated action), the probability has to mean something. Regulators increasingly ask "what does your 85% confidence score actually represent?" If your ECE is 0.20, you have no good answer — and that's a compliance risk, not just a modelling one.
:::

## Reliability diagrams — see the problem in one chart

A **reliability diagram** (also called a calibration curve) is the canonical diagnostic. Here is how to build one:

1. Take your model's predicted probabilities on a held-out set.
2. Bucket them into 10 equal-width bins: 0–0.1, 0.1–0.2, …, 0.9–1.0.
3. For each bin, compute **mean predicted probability** and **actual fraction of positives**.
4. Plot actual fraction (y-axis) against mean predicted (x-axis).

A perfectly calibrated model sits on the 45° diagonal. Curve bowing *below* the diagonal → model is overconfident. Bowing *above* → underconfident.

:::widget {name="calibration"}
:::

Drag the curve in the widget above and watch ECE update live. Two minutes here beats any formula.

## ECE — the single number for your slide deck

**Expected Calibration Error (ECE)** summarises the reliability diagram into one scalar. For B bins:

**ECE = weighted average of |predicted confidence − actual accuracy| across all bins**

Each bin is weighted by the fraction of samples it contains. Lower is better. An ECE of 0.05 means the model is off by about 5 percentage points on average.

:::table {title="ECE rough benchmarks (binary/multiclass classification)"}
| ECE range | Signal |
|---|---|
| < 0.02 | Well calibrated — safe to surface the number |
| 0.02 – 0.07 | Acceptable for most internal dashboards |
| 0.07 – 0.15 | Recalibrate before production |
| > 0.15 | Red flag — check training setup too, not just post-hoc fixes |
:::

## Three ways to fix miscalibration

You train your main model first, then fit a tiny post-hoc layer on a **separate calibration set** (not your test set — more on that in the gotcha below).

**Temperature scaling** — the workhorse for neural networks. Divide the raw logits by a single scalar T before the softmax. T > 1 softens the distribution (fixes overconfidence). T < 1 sharpens it. One parameter, almost impossible to overfit.

**Platt scaling** — fit a logistic regression (sigmoid) on the model's raw output scores. Two parameters. The classic fix for SVMs; also works well for gradient boosting.

**Isotonic regression** — a non-parametric monotonic mapping. Most flexible: it can correct a wiggly, non-uniform miscalibration that temperature scaling cannot. Trade-off: needs at least ~1 000 samples in the calibration set, otherwise it overfits badly.

```python {title="Reliability diagram, ECE, and Platt/temperature scaling" run=false}
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Run locally: pip install scikit-learn matplotlib numpy
# y_true: 1-D array of ground-truth labels (0/1)
# y_prob: 1-D array of predicted probabilities from your model

# ── 1. Reliability diagram ────────────────────────────────────────────
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(6, 4))
plt.plot(mean_pred, frac_pos, "s-", label="model")
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability diagram")
plt.legend()
plt.tight_layout()
plt.show()

# ── 2. ECE (manual, so you understand what sklearn does) ──────────────
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    score = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        score += (mask.sum() / total) * abs(acc - conf)
    return score

print(f"ECE before calibration: {ece(y_true, y_prob):.4f}")

# ── 3. Platt scaling via sklearn ──────────────────────────────────────
base = LogisticRegression()           # swap in your own sklearn estimator
cal  = CalibratedClassifierCV(base, method="sigmoid", cv=5)
cal.fit(X_train, y_train)
y_prob_cal = cal.predict_proba(X_test)[:, 1]
print(f"ECE after Platt:        {ece(y_test, y_prob_cal):.4f}")

# ── 4. Temperature scaling (neural network logits) ────────────────────
# logits: raw scores before softmax, shape (n_samples, n_classes)
T = 1.8                               # tune on calibration set; T>1 softens
exp_t = np.exp(logits / T)
probs_temp = exp_t / exp_t.sum(axis=1, keepdims=True)
# Then pick the positive-class column and recompute ECE
```

:::gotcha
Never calibrate on your test set. Compute ECE there to *measure* quality — but fit the calibration layer (temperature T, Platt sigmoid, isotonic mapping) on a separate held-out calibration set. If you tune T to minimise ECE on the test set you'll report a beautifully low ECE that evaporates in production. Use sklearn's `cv=` parameter or hold out 15–20% of training data explicitly for calibration.
:::

:::war-story {title="The '95% sure' dashboard"}
A Pune ML team at a mid-size NBFC deployed a loan-default model. Scores were piped straight into a credit officer dashboard labelled "Default probability: 91%". Officers stopped reading applications below 70% — "the model handles it." Eight months later, an RBI audit flagged that at the 90% predicted bucket the actual default rate was 54%. ECE was 0.23. The remediation cost: recalibrate, relabel every dashboard, retrain the operations team, and sit through two quarters of audits. A reliability diagram at launch would have taken 20 minutes.
:::

:::interview-line
"We always plot a reliability diagram and compute ECE before shipping any probabilistic model — overconfident outputs are the default, especially for neural nets, so temperature scaling is nearly always the first fix we reach for."
:::

:::qa {q="What does a reliability diagram tell you, and how do you read it?"}
It plots predicted probability (x-axis) against actual fraction of positives (y-axis) across equal-width bins. Points on the 45° diagonal mean the model is well calibrated. Points below the diagonal mean it's overconfident — it claims higher probability than the data supports. Points above mean it's underconfident. The further from the diagonal, the worse the calibration.
:::

:::qa {q="When would you prefer isotonic regression over temperature scaling?"}
Temperature scaling is one global scalar — it works when the model is uniformly over- or under-confident at all probability levels. If the reliability diagram shows a complex, non-monotonic curve (overconfident in one range, underconfident in another), isotonic regression's flexibility handles that. The catch: you need at least ~1 000 samples in the calibration set; on smaller sets it overfits and makes things worse.
:::

:::qa {q="A PM wants to display the model's 87% confidence score in the product UI. What do you check first?"}
Compute the ECE on a held-out set and plot the reliability diagram. Focus specifically on the 0.8–0.9 bin — if the actual fraction of positives in that bin is close to 0.85, the number is trustworthy. If ECE is above ~0.07 or that bin is badly off-diagonal, apply post-hoc calibration (temperature or Platt) and re-check before surfacing any probability to end users.
:::

:::drill {type="mcq" q="A neural network's reliability diagram shows almost all predictions clustered above 0.85, but actual accuracy in that bucket is 0.58. What is the most likely issue?"}
- [ ] The model is underfitting and needs more capacity
- [x] The model is overconfident and needs post-hoc calibration such as temperature scaling
- [ ] The model has high variance and needs more training data
- [ ] ECE cannot be computed when predictions cluster near 1.0
:::

:::drill {type="mcq" q="You have 180 samples left over for post-hoc calibration. Which method is the safest choice?"}
- [ ] Isotonic regression — it is non-parametric and therefore always preferred
- [x] Temperature scaling or Platt scaling — they have 1–2 parameters and will not overfit on 180 samples
- [ ] Retrain the model from scratch with a calibration loss term
- [ ] Skip calibration; a high AUC means the model is already well calibrated
:::

:::drill {type="mcq" q="You measure ECE on your test set and get 0.004. Your colleague calls it a win. What is the most suspicious explanation?"}
- [ ] The model predicts probabilities near 0.5 for everything
- [ ] Temperature scaling T was set to exactly 1.0
- [x] The calibration layer was fitted on the test set, leaking information and deflating ECE artificially
- [ ] The model has a very high AUC, which forces low ECE
:::

:::key-takeaway
Calibration turns a model's output from an opaque score into a trustworthy probability. Plot the reliability diagram, compute ECE, and apply temperature scaling by default — then isotonic regression only if you have the data for it. Do all of this before your model touches a user.
:::
