---
id: imbalance
track: 04-losses
title: "Imbalance & regularizing the objective"
badge: CORE
minutes: 9
prereqs: []
tags: [class-imbalance, label-smoothing, class-weights, focal-loss, oversampling, loss-regularization]
xp: 45
hot2026: false
---

You ship a fraud detection model. Test accuracy: **99.7%**. Your manager is thrilled. Then finance calls — you're missing every single fraudulent transaction. Your model learned one trick: predict "not fraud" for everything. Since 99.7% of transactions are legit, it gets rewarded constantly for doing nothing useful.

That is the imbalance trap. And it lives entirely inside the loss function.

## Why vanilla cross-entropy fails on skewed data

Standard cross-entropy treats every sample equally. On a 99:1 dataset, 99 legit examples outvote one fraud signal on every gradient step. The model drifts toward the majority class — not because it is stupid, but because that is exactly what the loss tells it to do.

The math is merciless: if the minority class contributes 1% of the total loss, its gradient is swamped. Weights adjust for the majority. The minority class becomes statistical noise.

:::why-prod
In production, recall on the rare class is usually the metric that matters — fraud missed, cancer not flagged, a critical alert silently dropped. Vanilla cross-entropy optimizes global accuracy and quietly sacrifices minority recall.
:::

## Fix 1: reweight the loss

The fastest fix is **class weights**. You multiply each sample's loss contribution by a weight inversely proportional to its class frequency. The minority class suddenly punches above its weight in the gradient.

PyTorch makes this a one-liner:

```python {title="Weighted cross-entropy in PyTorch" run=false}
import torch
import torch.nn as nn

# Class 0: majority (99%), class 1: minority (1%)
# A simple heuristic: weight = total_samples / (n_classes * count_per_class)
# Here we approximate: minority gets ~99x more pull than majority
weights = torch.tensor([0.5, 49.5])

criterion = nn.CrossEntropyLoss(weight=weights)

logits = model(x)          # shape: [B, 2]
loss = criterion(logits, labels)
loss.backward()

# The weight tensor lives on CPU by default — move it to the same device as your model:
# weights = weights.to(device)
# criterion = nn.CrossEntropyLoss(weight=weights)
```

No resampling, no architectural change — just one argument.

:::gotcha
Do not blindly copy `weight = 1 / frequency` and call it done. At 1:10000 imbalance, that extreme weight causes minority-batch loss spikes, gradient explosions, and divergence. In practice, cap weights or use `sqrt(1/freq)` to soften the effect. Always monitor gradient norms in your training logs.
:::

## Fix 2: fix the data distribution

Reweighting adjusts the objective. Sometimes you also need to fix what the model sees each batch:

:::table {title="Resampling strategies at a glance"}
| Strategy | What it does | Best for |
|---|---|---|
| Stratified batching | Guarantees minority class in every batch | Any domain; lowest risk |
| Random undersampling | Drops majority samples | When majority data is truly redundant |
| Oversampling (SMOTE) | Synthesizes new minority examples | Tabular data with small minority class |
| Class-balanced sampling | Samples classes equally per epoch | Large datasets where undersampling wastes data |
:::

In practice: start with class weights (zero cost), add stratified batching if training is still unstable, reach for SMOTE only for tabular tasks. Oversampling image pixels rarely helps — the model has already extracted all signal from the minority images it has seen.

## Fix 3: regularize the objective itself

Fixing imbalance is one half. The other half is preventing your model from becoming pathologically overconfident — even after reweighting.

**Label smoothing** is the workhorse here. Instead of training with hard targets (exactly 0 or 1), you soften them slightly:

> `y_smooth = (1 − ε) × y_hard + ε / K`

where ε is a small constant (0.1 is the usual default) and K is the number of classes. A model that assigns 99.99% probability to its answer is almost always fragile — one distribution shift and it collapses.

Label smoothing is especially useful on imbalanced problems because **overconfidence concentrates on the majority class**. Smoothing the labels naturally penalizes that certainty and keeps probability estimates calibrated.

**Auxiliary losses** are another underused tool. If you have side information — say, a fraud model that also predicts merchant category — adding a small auxiliary loss on that side task forces shared representations to stay general. It regularizes the primary objective indirectly, and it gives you a sanity signal during training.

## Putting it together: the production checklist

A solid imbalanced classifier typically combines all four levers:

1. Stratified sampling — ensure minority class appears in every batch
2. Reweighted or focal-loss objective — give minority gradients their fair share (focal loss mechanics are in the sibling lesson)
3. Label smoothing with ε ≈ 0.05–0.1 — keep the model honest about its confidence
4. **Threshold tuning at inference** — the default 0.5 threshold is almost never right for imbalanced problems; calibrate it to your precision/recall target post-training

That last point is where engineers leave the most value on the table. You can have a perfectly trained model and still deploy it at the wrong operating point. Training and threshold selection are separate concerns.

:::interview-line
"On imbalanced problems I separate model training from threshold selection — I optimize loss and AUC during training, then move the decision boundary post-hoc to hit the recall target the business actually cares about."
:::

:::qa {q="Your binary classifier has 98% accuracy but very poor recall on the positive class. What's wrong and how do you fix it?"}
The model learned to predict the majority class. The loss is dominated by majority examples, so minority gradients have almost no influence. Fix it with class weights or focal loss, add stratified batching, and after training tune the decision threshold toward higher recall rather than defaulting to 0.5.
:::

:::qa {q="What is label smoothing and why would you add it on top of an already-reweighted loss?"}
Label smoothing replaces hard 0/1 targets with soft values like 0.9/0.1, penalizing overconfident predictions. On imbalanced data the model tends to become extremely confident on the majority class even after reweighting. Smoothing adds a second layer of regularization that keeps calibration tighter and representations more general — two problems, two levers.
:::

:::drill {type="mcq" q="You set class_weight={0: 1, 1: 100} for a 100:1 imbalanced problem. Training loss spikes and the run diverges. Most likely cause?"}
- [ ] The model architecture is too small for the minority class signal.
- [ ] Stratified batching was not enabled alongside the weight.
- [x] The extreme weight amplifies minority-class loss spikes, causing gradient instability.
- [ ] CrossEntropyLoss does not support per-class weights in PyTorch.
:::

:::drill {type="mcq" q="Which statement about label smoothing is correct?"}
- [ ] Label smoothing increases model confidence, which improves calibration on rare classes.
- [ ] Label smoothing only helps when the training set is balanced.
- [x] Label smoothing softens the target distribution and penalizes overconfident predictions.
- [ ] Label smoothing is equivalent to dropout and should never be combined with it.
:::

:::key-takeaway
On imbalanced data the loss is biased by design — correct it with class weights and stratified batching, regularize overconfidence with label smoothing, and always tune the decision threshold separately from model training.
:::
