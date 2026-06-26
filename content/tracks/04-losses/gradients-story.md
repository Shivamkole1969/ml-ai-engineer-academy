---
id: gradients-story
track: 04-losses
title: "Gradients tell the story (CE vs focal)"
badge: CORE
minutes: 8
prereqs: []
tags: [loss-functions, cross-entropy, focal-loss, gradients, class-imbalance, object-detection]
xp: 45
hot2026: false
---

You're training a defect detector for a PCB manufacturing line in Pune. 999 out of 1000 boards are clean. One is defective. You train with standard cross-entropy, hit 99.9% accuracy, and the loss curve looks gorgeous. Then the defective board ships. Your model learned to say "clean" every time — and the loss *rewarded* it for that.

This is the moment you stop trusting accuracy and start reading the gradients.

## The gradient is the vote

Every loss function produces a gradient that flows backward into your weights. That gradient is the training signal — it says "move this weight by this much, in this direction." The *shape* of the loss function determines who gets a loud vote and who gets drowned out.

Cross-entropy (CE) is the default for classification. For a single example where the true class has predicted probability `p`:

```
CE = -log(p)
```

The gradient with respect to the raw logit (the score before softmax) for the true class is `p - 1`. Clean and simple.

Look at what that means by confidence level:

- Model **badly wrong** (p = 0.05): gradient = 0.05 − 1 = **−0.95** — loud signal, big weight update
- Model **confidently right** (p = 0.95): gradient = 0.95 − 1 = **−0.05** — quiet signal, tiny update

This is intentional. Once you know something well, CE stops shouting at you about it. Good design.

But here is where class imbalance turns that good design into a problem. In the PCB example: 999 clean boards, each with p = 0.97, each contributing a gradient of −0.03. Combined: **−29.97** nudging the model toward "always predict clean." The single defect, even if the model is completely wrong on it, contributes at most −1.0. The majority drowns out the minority — quietly, with no error message.

:::why-prod
In production, imbalance is the norm. Fraud is 0.1% of transactions. Tumors are a fraction of scan volume. Defects are rare by definition — that's what makes them defects. CE gradients amplify the easy majority and starve the hard minority every single epoch.
:::

## Focal loss: turning down the easy signal

Focal loss, introduced with RetinaNet (2017), adds one multiplicative term:

```
FL(p) = -(1 - p)^γ · log(p)
```

The `γ` (gamma) is the focusing parameter, typically 2. Watch what it does to the gradient magnitude:

- **Easy, correct** (p = 0.95): `(1 − 0.95)^2 = 0.0025` — down-weighted by ~400×
- **Hard, wrong** (p = 0.05): `(1 − 0.05)^2 = 0.9025` — nearly unchanged

Your 999 clean boards still contribute, but at a tiny fraction of their CE weight. The single defect is no longer shouted down.

:::table {title="CE vs Focal — who gets heard at each confidence level"}
| Predicted prob (p) | Example type | CE gradient | Focal factor (γ=2) | Effective signal |
|---|---|---|---|---|
| 0.95 | Easy correct | 0.05 | 0.0025 | ~0.0001 |
| 0.70 | Moderate correct | 0.30 | 0.09 | ~0.027 |
| 0.30 | Moderate wrong | 0.70 | 0.49 | ~0.34 |
| 0.05 | Hard wrong | 0.95 | 0.9025 | ~0.86 |
:::

```python {title="CE vs Focal — compare gradient contribution" run=false}
import torch
import torch.nn.functional as F

# pip install torch  (CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)

def focal_loss(logits, targets, gamma=2.0):
    """Binary focal loss, averaged over the batch."""
    # Standard BCE, unreduced so we can weight per-example
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # p_t: model's confidence on the TRUE label
    p_t = torch.exp(-bce)
    # Down-weight easy (high p_t) examples
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


# 1 hard defect (model wrong), 4 easy clean boards (model confident)
logits  = torch.tensor([-2.0,  3.0,  2.8,  3.1,  2.9])
targets = torch.tensor([ 1.0,  0.0,  0.0,  0.0,  0.0])

ce_loss = F.binary_cross_entropy_with_logits(logits, targets)
fl_loss = focal_loss(logits, targets)

print(f"CE loss:    {ce_loss:.4f}")  # easy negatives dominate
print(f"Focal loss: {fl_loss:.4f}")  # hard example gets proportional weight back
```

:::gotcha
Focal loss is not a drop-in for CE everywhere. First, it adds `γ` as a hyperparameter — γ=2 is a solid default for detection, but tune it for your imbalance ratio. Second, at the very start of training your model is often *confidently wrong*, which means `(1-p)^2` is tiny and the hard examples get suppressed right when the model most needs to learn from them. Fix: run 1–2 epochs of plain CE to warm up, then switch to focal.
:::

:::interview-line
"Focal loss doesn't fix class imbalance — it changes whose gradient gets heard. Easy examples are turned down so the hard, rare ones can actually teach the model something."
:::

:::qa {q="What problem does focal loss solve that cross-entropy doesn't?"}
CE aggregates gradients from all examples, so a large easy majority floods the signal and buries the rare hard minority. Focal loss multiplies each example's CE by `(1-p_t)^γ`, where `p_t` is the model's confidence on the true label. High-confidence examples get shrunk; low-confidence hard examples retain nearly their full gradient. This rebalances the training signal without discarding any data.
:::

:::qa {q="How does changing γ affect focal loss behaviour?"}
γ=0 is identical to standard CE — no focusing at all. As γ increases, the suppression of easy examples becomes more aggressive. γ=2 is the typical default for detection. Very high γ (e.g., 5+) can destabilize early training because the model starts out uncertain everywhere and has no examples it considers "easy" to suppress, so the effective loss can be very small and updates are tiny. Tune γ alongside your learning rate.
:::

:::qa {q="Can focal loss be used for multi-class problems?"}
Yes. After softmax, `p_t` is just the predicted probability of the ground-truth class — the same modulating factor `(1-p_t)^γ` applies directly to the per-class CE term. The intuition is identical: whichever class the model is already confident about gets down-weighted, and whichever it's struggling with keeps its full gradient.
:::

:::drill {type="mcq" q="A binary classifier trains on 1,000 negatives and 10 positives. After 20 epochs with CE loss, accuracy is 99% but AUC is 0.53. What is the most likely explanation?"}
- [ ] The learning rate is too high, causing gradient oscillation
- [x] Gradient signal from easy negatives dominates; the model learned to always predict negative
- [ ] The model architecture is too shallow for the data
- [ ] CE loss is incompatible with binary classification
:::

:::drill {type="mcq" q="In focal loss FL = -(1-p)^γ · log(p), what happens to a training example where p = 0.92 (model confidently correct)?"}
- [ ] The example receives a larger gradient than under CE because log(0.92) is small
- [ ] The gradient is identical to CE since the model is correct
- [x] The loss contribution is multiplied by (0.08)^γ, shrinking it by roughly 400× when γ=2
- [ ] The example is excluded from the backward pass entirely
:::

:::drill {type="mcq" q="Which scenario is focal loss LEAST likely to improve over standard CE?"}
- [ ] One-stage object detection with thousands of background anchors per image
- [ ] Medical scan classification where positive cases are 1 in 200
- [x] A perfectly balanced binary dataset with equal positive and negative counts
- [ ] Multi-label document tagging where most labels are rare
:::

:::key-takeaway
The gradient is the vote. Cross-entropy lets the easy majority vote loudly and drown out the rare hard examples. Focal loss turns down the easy examples' microphone — so the hard, rare ones can finally teach the model something.
:::
