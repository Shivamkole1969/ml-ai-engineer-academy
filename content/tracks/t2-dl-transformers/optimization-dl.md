---
id: optimization-dl
track: t2-dl-transformers
title: "Optimizers, epochs, batches & a tiny PyTorch net"
badge: CORE
minutes: 9
prereqs: []
tags: [pytorch, optimizer, adam, adamw, sgd, batch, epoch, learning-rate, training-loop]
xp: 45
hot2026: false
---

Imagine you just shipped your first neural network to a staging server. It trained fine — loss went down, accuracy looked decent. Then a teammate asks: "What optimizer did you use? What batch size? Did you tune the learning rate?" You freeze. You hit the defaults and hoped for the best. That moment is exactly what this lesson fixes.

Training a neural network is really just one loop repeated thousands of times: measure how wrong you are, nudge every weight a tiny bit in the right direction, repeat. But *how* you nudge — and *how much data* you look at before each nudge — is where most of the craft lives.

## The training loop in plain English

Every training step does four things:

1. **Forward pass** — feed a batch of examples through the network, get predictions.
2. **Compute loss** — measure how far off those predictions are (e.g., cross-entropy for classification).
3. **Backward pass** — backprop computes the gradient of the loss with respect to every weight.
4. **Optimizer step** — use those gradients to update the weights, then zero the gradients.

That fourth step is the optimizer's entire job: take the gradients and decide exactly how to change each weight.

:::why-prod
In production, the choice of optimizer and its hyperparameters (especially learning rate) is often the difference between a model that converges in hours versus one that diverges or plateaus and wastes GPU budget. Getting this wrong is expensive.
:::

## Batches and epochs — the rhythm of training

**Batch size** is how many training examples you process before doing one optimizer step. Common values: 32, 64, 256.

**An epoch** is one full pass through the entire training dataset. If you have 10,000 examples and a batch size of 100, one epoch = 100 optimizer steps.

Why not feed the whole dataset at once? Two reasons:
- It rarely fits in GPU memory.
- Smaller batches add a helpful noise that often helps the model generalise better.

Why not feed one example at a time (batch size = 1)? Updates become extremely noisy, and you lose the speed benefit of GPU parallelism.

:::table {title="Batch size trade-offs"}
| Batch size | GPU usage | Gradient noise | Generalisation | Wall-clock |
|---|---|---|---|---|
| Very small (1–8) | Low | Very high | Sometimes better | Slow |
| Medium (32–256) | Good | Moderate | Good | Fast |
| Very large (1024+) | Excellent | Low | Sometimes worse | Very fast (but watch out) |
:::

:::gotcha
Large batch training can converge to "sharp" minima that generalise poorly on unseen data. If you scale up batch size, scale up your learning rate proportionally (the *linear scaling rule*) — and even then, watch your validation metrics closely.
:::

## Optimizers: from SGD to AdamW

**SGD (Stochastic Gradient Descent)** is the original. Each weight gets updated by subtracting a fraction of its gradient:

```
weight = weight - learning_rate * gradient
```

Simple, but slow to navigate flat regions or ravines in the loss landscape.

**SGD + Momentum** remembers the direction you were going and keeps some of that velocity. It's like rolling a ball downhill — it builds speed through flat patches and cuts across ravines.

**Adam** (Adaptive Moment Estimation) adapts the learning rate *per parameter*, based on running averages of past gradients. It's fast, robust to choice of initial learning rate, and became the default for deep learning and NLP.

**AdamW** is Adam with a corrected weight decay — it regularises weights properly instead of baking decay into the gradient update. It's what almost every modern LLM is trained with. When in doubt, use AdamW.

## Learning rate — the single most important knob

The learning rate controls the size of each weight update. Too high and training diverges (loss explodes). Too low and training is glacially slow or gets stuck.

A typical starting point for AdamW: `lr=1e-3` for smaller networks, `lr=1e-4` or lower for fine-tuning large pretrained models.

Most production training jobs use a **learning rate scheduler** — starting at a small warmup LR, rising to a peak, then decaying (cosine or linear). Flat LRs throughout training are a red flag.

## A tiny PyTorch net — all four concepts together

```python {title="Minimal PyTorch training loop" run=false}
import torch
import torch.nn as nn

# --- 1. Tiny dataset (XOR problem) ---
X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]])
y = torch.tensor([0., 1, 1, 0])          # XOR labels

# --- 2. Model: 2 inputs -> 4 hidden -> 1 output ---
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# --- 3. Loss + optimizer (AdamW is the modern default) ---
loss_fn  = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

# --- 4. Training loop: 3 epochs, batch_size = full dataset (tiny data) ---
for epoch in range(3):
    for step in range(200):          # 200 optimizer steps per epoch
        preds  = model(X).squeeze()  # forward pass
        loss   = loss_fn(preds, y)   # compute loss
        optimizer.zero_grad()         # clear old gradients
        loss.backward()               # backward pass
        optimizer.step()              # update weights

    print(f"Epoch {epoch+1} | loss: {loss.item():.4f}")

# Run locally: pip install torch --index-url https://download.pytorch.org/whl/cpu
# Then: python this_file.py
```

The four-line inner loop (`preds → loss → zero_grad → backward → step`) is the heartbeat of every PyTorch model you'll ever train, from a toy XOR net to a 70B LLM.

:::interview-line
"I default to AdamW with a cosine LR schedule and a short warmup — it's what most frontier labs use and it's robust to learning-rate sensitivity."
:::

:::qa {q="What is the difference between Adam and AdamW?"}
Adam bakes weight decay into the gradient update, which interacts with the adaptive learning rate in a mathematically incorrect way. AdamW applies weight decay directly to the weights — separate from the gradient step — giving cleaner regularisation. For fine-tuning pretrained models, AdamW consistently outperforms Adam.
:::

:::qa {q="Why do we call optimizer.zero_grad() before loss.backward()?"}
PyTorch accumulates gradients by default — it adds new gradients on top of old ones in each `.grad` attribute. If you forget to zero them, every step includes the ghosts of all previous steps, causing wildly incorrect updates. Zeroing before the backward pass gives you a clean gradient for the current batch only.
:::

:::qa {q="You increase batch size 8x to speed up training. What else should you adjust?"}
Scale the learning rate up proportionally — roughly 8x as well (the linear scaling rule from Goyal et al.). You may also need a longer warmup period. Without this, large-batch training often converges to a worse final accuracy than smaller-batch training.
:::

:::drill {type="mcq" q="A model's training loss keeps dropping but validation loss starts rising after epoch 5. What is most likely happening?"}
- [ ] The learning rate is too low
- [ ] The batch size is too large
- [x] The model is overfitting — it is memorising the training data
- [ ] AdamW is not zeroing gradients correctly
:::

:::drill {type="mcq" q="Which optimizer is the standard default for training and fine-tuning large language models in 2025?"}
- [ ] SGD with momentum
- [ ] Vanilla Adam
- [x] AdamW
- [ ] RMSProp
:::

:::drill {type="mcq" q="You have 50,000 training examples and use a batch size of 500. How many optimizer steps happen in one epoch?"}
- [ ] 500
- [x] 100
- [ ] 50,000
- [ ] 1
:::

:::key-takeaway
The training loop is always the same four steps: forward, loss, backward, optimizer step. Use AdamW with a learning rate scheduler in almost every production setting — it's robust, well-studied, and what modern labs default to.
:::
