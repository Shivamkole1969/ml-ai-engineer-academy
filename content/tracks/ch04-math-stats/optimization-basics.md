---
id: optimization-basics
track: ch04-math-stats
title: "Optimization basics: gradients, SGD, Adam"
badge: CORE
minutes: 9
prereqs: []
tags: [optimization, gradients, sgd, adam, training, loss, backprop]
xp: 45
hot2026: false
---

Imagine you baked a neural network overnight, and the training loss curve looks like a ski slope that suddenly turns into a flat plateau. Your model is stuck, barely learning. You double the learning rate — and now the loss explodes. You halve it — and it crawls so slowly you could age waiting for it to finish. Sound familiar? That's the optimizer having a bad day. Understanding *why* that happens — and which knobs to turn — is the difference between shipping models and babysitting loss curves forever.

## What optimization actually means in ML

Every ML model has a **loss function**: a number that measures how wrong the model's predictions are. Optimization is the process of adjusting the model's parameters (weights) to make that number as small as possible.

Think of it like this: imagine the loss function as a hilly landscape. The model's current parameters are a hiker standing somewhere on those hills. The goal is to walk downhill to the lowest valley — the **global minimum** (or at least a good-enough low valley).

The key tool is the **gradient** — a vector that tells you, for each parameter, which direction is "uphill." To go downhill, you step in the *opposite* direction.

That one idea — **step opposite to the gradient** — is the engine behind almost all modern ML training.

## Gradient descent: the base algorithm

**Gradient descent** (GD) computes the gradient over your *entire* dataset and takes one step:

```
new_weight = old_weight - learning_rate × gradient
```

The **learning rate** (often written `η` or `lr`) controls the step size. Too large: you overshoot the valley and bounce around. Too small: you inch forward and training takes forever.

Full gradient descent is rarely used in practice because computing gradients over millions of examples is expensive. Instead, we use variants:

:::table {title="GD variants at a glance"}
| Variant | Data per step | Speed | Noise |
|---|---|---|---|
| Gradient Descent | Full dataset | Slow | Low |
| Stochastic GD (SGD) | 1 example | Fast | Very high |
| Mini-batch SGD | 32–512 examples | Fast | Medium |
:::

**Mini-batch SGD** is what people mean when they say "SGD" in practice. The noise from small batches is actually *useful* — it helps the optimizer escape shallow bad valleys (sharp minima) and find flatter, more generalizable ones.

:::why-prod
In production, your training loop runs on mini-batches by default. The batch size you choose affects not just speed but the generalization of the final model — large batches can converge to sharper minima that perform slightly worse on new data.
:::

## Momentum: don't stop rolling

Plain SGD is noisy — it zigzags a lot. **Momentum** smooths this out by accumulating a "rolling average" of past gradients. Think of it like a ball rolling downhill: it builds up speed in consistent directions and dampens the noise from side-to-side jitter.

The update rule gains one extra term:

```
velocity = β × velocity + gradient
weight = weight - lr × velocity
```

`β` is typically 0.9. The optimizer now "remembers" where it was heading and keeps moving in that direction even if the current gradient is noisy.

## Adam: the workhorse of modern ML

**Adam** (Adaptive Moment Estimation) is what you almost certainly use unless you have a reason not to. It combines two ideas:

1. **Momentum** — a rolling average of gradients (first moment)
2. **Adaptive learning rates** — a rolling average of *squared* gradients (second moment), used to scale the step size per parameter

Parameters that get large, consistent gradients get a smaller step size. Parameters with small or inconsistent gradients get a bigger step size. The result: Adam tunes itself per-parameter, which is why it converges fast even with the default learning rate.

```python {title="SGD vs Adam — quick comparison" run=false}
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()

# SGD with momentum — you tune lr carefully
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam — works well out of the box with lr=1e-3
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop is identical regardless of optimizer
x = torch.randn(32, 10)
y = torch.randn(32, 1)

for optimizer in [optimizer_sgd, optimizer_adam]:
    optimizer.zero_grad()        # clear old gradients
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()              # compute new gradients
    optimizer.step()             # update weights
    print(f"{type(optimizer).__name__}: loss={loss.item():.4f}")

# Run locally: pip install torch  (CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)
```

:::gotcha
A classic mistake is forgetting `optimizer.zero_grad()` before `loss.backward()`. PyTorch *accumulates* gradients by default. Skip the zero and your gradients stack up across batches — your model trains on ghost data from previous steps and diverges in mysterious ways.
:::

## Learning rate: the most important hyperparameter

No matter which optimizer you use, the learning rate is the single most impactful hyperparameter. A few production-proven practices:

- **Start with `1e-3` for Adam** — it's the community default and usually works.
- **Use a learning rate scheduler** — decay the rate over time so the model fine-tunes gently at the end of training.
- **Learning rate warmup** — especially in Transformers, ramp the LR up from nearly zero for the first few hundred steps before decaying. This prevents early divergence when gradients are noisy.
- **Learning rate finder** — run a quick sweep from tiny to large LR, plot the loss, and pick the value just before the loss starts exploding.

:::interview-line
"Adam adapts the learning rate per parameter using first and second gradient moments, which is why it converges fast with minimal tuning — but for fine-tuning large language models, many teams switch back to SGD with a carefully tuned schedule for better generalization."
:::

:::qa {q="What is the gradient and why do we step opposite to it?"}
The gradient is a vector of partial derivatives — it points in the direction that *increases* the loss the fastest. To minimize the loss, we step in the opposite direction. The step size is controlled by the learning rate.
:::

:::qa {q="When would you use SGD over Adam in production?"}
Adam is the default for most deep learning. SGD with momentum and a well-tuned schedule can generalize slightly better in some vision tasks and is preferred by some teams for fine-tuning large pretrained models. Adam's adaptive rates sometimes lead to sharper minima that overfit more. In practice, Adam first, then experiment with SGD if you need to squeeze out last-mile accuracy.
:::

:::qa {q="What is the role of batch size in optimization?"}
Batch size controls the noise level of each gradient estimate. Smaller batches are noisier but update more frequently and tend to generalize better. Larger batches are more stable but can converge to sharp, overfit-prone minima. In distributed training, larger batches are used for throughput, but you typically scale the learning rate proportionally (linear scaling rule).
:::

:::drill {type="mcq" q="Why does forgetting optimizer.zero_grad() cause training to go wrong?"}
- [ ] It prevents gradients from being computed at all
- [x] Gradients accumulate across steps, making updates much larger than intended
- [ ] The model weights are reset to random values
- [ ] The loss function stops updating
:::

:::drill {type="mcq" q="Which statement about Adam's learning rate adaptation is correct?"}
- [ ] Adam uses the same learning rate for every parameter, just decayed over time
- [ ] Adam only adapts based on the sign of the gradient, not the magnitude
- [x] Adam scales each parameter's update by the inverse of the square root of past squared gradients
- [ ] Adam requires no learning rate hyperparameter at all
:::

:::key-takeaway
Optimization is just "step downhill." SGD with mini-batches is the foundation; momentum smooths the path; Adam adapts the step size per parameter and is your go-to default. The learning rate remains the most important thing you can tune — warmup, schedule, and a quick sweep will serve you better than obsessing over which optimizer to use.
:::
