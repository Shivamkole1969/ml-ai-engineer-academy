---
id: drift-types
track: 02-data-reality
title: "Drift kept distinct: covariate vs label vs concept"
badge: HOT
minutes: 9
prereqs: []
tags: [drift, covariate-drift, label-drift, concept-drift, monitoring, production-ml]
xp: 60
hot2026: true
---

It's 2 AM on a Tuesday. Your fraud-detection model — running quietly for six months — suddenly fires alerts on 40% of all transactions. Your Slack is a wall of red. You roll back the model... and the alerts stay. Something changed, but *what*?

This is a drift incident. And the first question isn't "how do I fix it?" It's: **which kind of drift is this?** The answer changes everything about the fix.

## Three drifts, three root causes

All drift means "the real world stopped looking like your training data." But there are three distinct ways that can happen.

**Covariate drift** — *P(X) changes, P(Y|X) stays the same.*

The features your model sees day-to-day shift away from what it saw during training. The relationship between features and labels is still valid — it just never saw the new slice of the input space.

Example: You train a spam classifier on email from January to March. By June, users have migrated to mobile and write shorter messages. The rule "spam prose looks like X" still holds — your model just never saw short-subject-line spam. The input distribution drifted; the rule didn't.

**Label drift** — *P(Y) changes, P(X|Y) stays the same.*

The *rate* of each label shifts. Your fraud model trained when 0.3% of transactions were fraud. During Diwali, actual fraud hits 1.5%. Fraudsters haven't changed their tactics — the same behavioral fingerprints still mean fraud — but your model is calibrated for a low-fraud world. Confidence scores go haywire even though the underlying rule is fine.

**Concept drift** — *P(Y|X) changes.*

This is the dangerous one. The *meaning* of the label given the features has changed. The rule that connected inputs to outputs in training no longer holds in the real world.

Example: Pre-2020, "office space search in Baner, Pune" predicted a lease within 45 days. Post-WFH, the same search often means "casual browsing, no intent." The feature is identical; the label distribution *given* that feature is completely different. No amount of retraining on old data helps — you need new labeled examples from the new world.

:::why-prod
Different drifts need different fixes. Covariate drift → retrain on recent data or use domain adaptation. Label drift → recalibrate your decision threshold or reweight classes. Concept drift → collect fresh labels and possibly redesign features. One wrong diagnosis can cost weeks of wasted work.
:::

:::table {title="Three drifts at a glance"}
| Drift type | What shifts | P(X) | P(Y) | P(Y\|X) | Typical fix |
|---|---|---|---|---|---|
| Covariate | Input distribution | Changes | Same | Same | Retrain on recent X |
| Label | Label base rate | Same | Changes | Same | Recalibrate / reweight |
| Concept | Feature→label rule | Same | Same | Changes | New labels, redesign |
:::

```python {title="KS test for covariate drift on a single feature" run=false}
# pip install scipy  — runs locally, no GPU needed
from scipy.stats import ks_2samp
import numpy as np

def detect_covariate_drift(
    train_col: np.ndarray,
    prod_col: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Kolmogorov–Smirnov test: are train and prod from the same distribution?
    p-value < threshold  →  statistically significant covariate drift.
    """
    stat, p_value = ks_2samp(train_col, prod_col)
    return {
        "ks_stat": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": p_value < threshold,
    }

# Example: transaction_amount drifts upward during festival season
train_amounts = np.random.lognormal(mean=5.0, sigma=1.0, size=10_000)
prod_amounts  = np.random.lognormal(mean=5.8, sigma=1.0, size=2_000)   # shifted

result = detect_covariate_drift(train_amounts, prod_amounts)
print(result)
# → {'ks_stat': 0.21, 'p_value': 0.0, 'drift_detected': True}

# KS only tells you *covariate* drift. Check label rates separately.
# For label drift: compare positive-class % in train vs. recent prod windows.
# For concept drift: sample recent data, manually label it, compare to train labels.
```

:::gotcha
Covariate drift and concept drift both tank your accuracy, so dashboards can't tell them apart. The key test: pull a batch of recent examples that *closely resemble* your training data (same feature range) and check if the model is wrong on those too. If yes → concept drift: the rule has changed and retraining on old data won't help. If no → covariate drift: the model is fine, it just needs exposure to the new input space.
:::

:::war-story {title="The Diwali fraud spike that cost two weeks"}
A Pune-based payments startup saw their fraud model's precision collapse from 89% to 61% during Diwali week. The team assumed concept drift — "fraudsters must have changed tactics" — and spent two weeks collecting and annotating thousands of new examples. Post-hoc analysis told a quieter story: actual fraud rates had jumped 5×, but the fraudsters' behavioral fingerprints were unchanged. It was pure label drift. Recalibrating the decision threshold for the new base rate would have taken one afternoon. Two weeks and significant annotation budget lost to the wrong diagnosis.
:::

:::interview-line
"Covariate drift is new kinds of inputs with the same rules; label drift is shifted outcome rates with the same rule; concept drift is the rule itself has changed — and each needs a fundamentally different response."
:::

:::qa {q="How would you distinguish covariate drift from concept drift in production?"}
Evaluate model accuracy on a held-out set of *historical* examples alongside recent ones. If the model stays accurate on old inputs but fails on new ones, suspect covariate drift — the rule is intact, the input space just expanded. If accuracy also degrades on inputs that closely match training data, that's concept drift: the underlying relationship has changed and you need fresh labeled data from the current world.
:::

:::qa {q="Your model's AUC drops from 0.91 to 0.74 over three months, but your feature distributions look stable on KS tests. What's your next hypothesis and step?"}
Stable features with degrading performance points toward concept drift — the feature-to-label mapping has shifted despite the inputs looking similar. The immediate step is to sample 200–300 recent examples, manually label them, and compare those labels to what your training set would predict for the same feature values. If recent ground-truth diverges from historical patterns, you've confirmed concept drift and need to re-collect labels from the new distribution before retraining.
:::

:::drill {type="mcq" q="A ride-share surge model performs well in summer but consistently underestimates surge every monsoon. Monsoon traffic patterns were underrepresented in training. Which drift type is this?"}
- [ ] Concept drift — the surge pricing rule itself changed seasonally
- [x] Covariate drift — the input distribution (weather, demand density) shifted away from training
- [ ] Label drift — the overall rate of surge events changed
- [ ] None of the above; this is pure class imbalance, not drift
:::

:::drill {type="mcq" q="Which observation is the strongest evidence of concept drift rather than covariate or label drift?"}
- [ ] Population Stability Index (PSI) > 0.2 on three input features
- [ ] KS test rejects the null for the top feature
- [ ] The positive-class rate in production is 4× higher than in training
- [x] Model accuracy degrades even on examples whose feature values fall squarely within the training distribution
:::

:::key-takeaway
Name the drift before you pick the fix. Covariate: new input territory, same rule. Label: same rule, shifted outcome rates. Concept: the rule itself has changed. One wrong diagnosis can waste weeks.
:::
