---
id: label-degradation
track: 02-data-reality
title: "Label degradation: agreement, guideline drift, soft labels"
badge: CORE
minutes: 8
prereqs: []
tags: [labeling, annotation, data-quality, soft-labels, inter-annotator-agreement, label-smoothing]
xp: 45
hot2026: false
---

Your content-moderation model launched at 89% accuracy. Three months later it's at 74%. Nothing in the codebase changed. You roll back the model weights — still 74%. You audit the feature pipeline — clean. Finally, in desperation, you open the annotation spreadsheet and sort by date.

That's when you see it. The annotation team changed what "offensive" means. Twice.

The model is fine. The labels rotted.

## What is label degradation?

Labels don't just start bad. They get worse over time, or reveal hidden inconsistencies that were always there.

Label degradation covers three related problems:

1. **Annotator disagreement** — different people assign different labels to the same example, right now.
2. **Guideline drift** — the definition of a label shifts quietly over weeks or months.
3. **Hard labels hiding real uncertainty** — a binary 0/1 forces a decision where the truth is "it depends."

Each one poisons your training signal in a different way.

## Inter-annotator agreement

When multiple people label your data, some disagreement is expected. The question is: how much?

The standard metric is **Cohen's Kappa (κ)** — for two annotators — or **Fleiss' Kappa** for larger teams. Kappa subtracts the agreement you'd get by random chance, so it's far more honest than raw "percent agreed."

:::why-prod
Low annotator agreement means you're training on noise that looks like signal. Most teams never measure kappa. They notice the model is bad, assume the architecture needs work, and spend two weeks on hyperparameters. The bug was in a spreadsheet.
:::

:::table {title="Cohen's Kappa — what the number means"}
| κ value | Agreement level | What to do |
|---|---|---|
| < 0.20 | Slight | Stop annotating — fix guidelines first |
| 0.21–0.40 | Fair | Review ambiguous cases with team |
| 0.41–0.60 | Moderate | Acceptable for low-stakes tasks |
| 0.61–0.80 | Substantial | Good for most production use |
| 0.81–1.00 | Almost perfect | Great — audit for groupthink |
:::

```python {title="Compute Cohen's Kappa" run=false}
# pip install scikit-learn  (no GPU needed, runs in Colab free tier)
from sklearn.metrics import cohen_kappa_score

# Two annotators on 6 examples: 1 = "urgent", 0 = "not urgent"
annotator_a = [1, 0, 1, 1, 0, 0]
annotator_b = [1, 0, 0, 1, 0, 1]

kappa = cohen_kappa_score(annotator_a, annotator_b)
print(f"Cohen's Kappa: {kappa:.3f}")
# κ ≈ 0.40 → fair agreement only — time to revisit what "urgent" means
```

## Guideline drift

Even one annotator can degrade your labels — over time.

Picture labeling "urgent" support tickets in January. By March, your mental model of "urgent" has quietly shifted. A rough week made you more liberal. A new team member has a different reference point. Nobody updated the style guide.

This is **guideline drift**: the annotation criteria change without being versioned or communicated. Your model trains on early-January "urgent" and late-March "urgent" as if they're the same class. They aren't.

The fix is almost embarrassingly simple and almost never done: **version your annotation guidelines exactly like code.** Pin a doc version or commit hash to every annotation batch. When guidelines change, record the date in your dataset metadata so you can segment the data later.

:::gotcha
When kappa is low, teams often force annotators to reach agreement through discussion — then declare the problem solved. But if your guidelines are ambiguous, all you've produced is *confidently wrong* labels. Consensus is not the same as correctness. Fix the guidelines first, then re-annotate.
:::

## Soft labels — when certainty is itself a lie

Hard labels are binary: 0 or 1, "spam" or "not spam". But consider a tweet that seven out of ten reviewers called "toxic." Rounding to 1 discards real signal. Rounding to 0 is plainly wrong.

**Soft labels** encode uncertainty as a probability. Instead of `1`, you store `0.7`. Train the model to predict that probability and it learns that this category is genuinely ambiguous — not a crisp boundary.

Two practical approaches:

- **Annotator distributions**: if 7/10 annotators said yes, the soft label is 0.7. Best when you have multiple annotators and real disagreement.
- **Label smoothing**: replace hard 1s with `1 − ε` and hard 0s with `ε`. A common choice is ε = 0.1. Prevents overconfidence even when you have only one annotator.

```python {title="Label smoothing in PyTorch" run=false}
# Runs on CPU — no GPU needed
import torch
import torch.nn.functional as F

hard_labels = torch.tensor([1, 0, 1])  # three training examples

def smooth(labels, num_classes=2, eps=0.1):
    one_hot = F.one_hot(labels, num_classes).float()
    return one_hot * (1 - eps) + eps / num_classes

print(smooth(hard_labels))
# tensor([[0.0500, 0.9500],
#         [0.9500, 0.0500],
#         [0.0500, 0.9500]])
# The model no longer trains toward infinite confidence — calibration improves
```

:::why-prod
Well-calibrated models are production gold. When your model says 70% confident, it should be right about 70% of the time. Soft labels and label smoothing both push toward better calibration, which matters enormously in fraud detection, medical triage, or anything where the downstream system acts on probabilities rather than hard decisions.
:::

:::interview-line
"Label quality degrades three ways: annotators disagree (measure kappa), guidelines silently shift (version them), and hard labels hide genuine uncertainty (use soft labels or label smoothing)."
:::

:::qa {q="How would you detect label degradation in a dataset that's been collected over many months?"}
Segment the data by annotation date and compare model performance on each slice. If early-annotated examples behave very differently from later ones — with no covariate shift — your labels drifted. Also run kappa on a rolling sample of recent annotations and alert when it drops below your threshold.
:::

:::qa {q="What is label smoothing and why does it help?"}
Label smoothing replaces hard targets (0 or 1) with softened values like 0.05 and 0.95. It stops the model from chasing infinite log-likelihood on training examples, which improves calibration on held-out data. It also acts as a mild regularizer that helps when some training labels are noisy or mislabelled.
:::

:::qa {q="When would you prefer annotator-distribution soft labels over uniform label smoothing?"}
When genuine disagreement is meaningful signal — subjective tasks like toxicity, sentiment, or legal risk assessment. Aggregating annotator votes into a probability captures real-world ambiguity that a fixed epsilon cannot. If the task has a true ground truth (e.g., "is this image a cat?"), label smoothing is usually sufficient.
:::

:::drill {type="mcq" q="A team re-annotated 30% of their dataset after updating their guidelines, then merged old and new labels into one training set without tracking which batch each label came from. What's the most likely outcome?"}
- [ ] The model learns a more robust representation because it saw more diverse examples
- [x] The model trains on inconsistent signals for the same class, degrading precision and recall
- [ ] Validation accuracy increases because there are more labelled examples overall
- [ ] There is no effect because the total positive-label count stays the same
:::

:::drill {type="mcq" q="You compute Cohen's Kappa between two senior annotators on a binary task and get κ = 0.18. What is the right first step?"}
- [ ] Increase the annotation team to five people and take a majority vote
- [ ] Retrain the model with a larger backbone — label noise is expected
- [x] Pause annotation and rewrite the guidelines with concrete worked examples
- [ ] Apply label smoothing with ε = 0.18 to compensate
:::

:::key-takeaway
Label degradation is silent and cumulative. Measure inter-annotator agreement with Cohen's Kappa, version your annotation guidelines like code, and treat annotator disagreement as signal — not noise — by using soft labels or label smoothing.
:::
