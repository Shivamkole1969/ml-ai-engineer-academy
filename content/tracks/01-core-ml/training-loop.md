---
id: training-loop
track: 01-core-ml
title: "What .fit() hides — the 5-step training loop"
badge: CORE
minutes: 9
prereqs: []
tags: [pytorch, training, gradients, backpropagation, optimizer, fundamentals]
xp: 45
hot2026: false
---

It's your third day at the new job. The classification model has been training overnight. The loss curve is flat. Your lead leans over and asks one question: "Did you zero the gradients?"

You know what gradients are. You know what a loss is. But in the two years you spent calling `.fit()`, you never had to think about what happens *between* batches. Until now.

This lesson tears open `.fit()` and shows you the five steps that fire on every single batch. Know these cold — you'll debug faster, tune smarter, and land the "zero_grad" question every interviewer loves to ask.

## The loop Keras runs for you

When Keras calls `.fit()`, it orchestrates a loop you never see. PyTorch makes you write it yourself — and that's a feature. Every production ML engineer eventually needs to reach inside that loop and change something: skip a bad batch, accumulate gradients, log a custom metric, clip a gradient spike.

Here are the five steps, in order, on every batch:

1. **Zero the gradients** — clear the previous batch's accumulated gradient values
2. **Forward pass** — run inputs through the model and get predictions
3. **Compute the loss** — measure how wrong those predictions are
4. **Backward pass** — compute gradients of the loss with respect to every parameter
5. **Optimizer step** — nudge each parameter in the direction that reduces loss

That's it. The entire magic of learning, distilled to five lines.

:::why-prod
In production you routinely need to customise this loop: accumulate gradients over N mini-batches to simulate a larger batch, clip gradient norms to prevent training blowups, or skip corrupted batches gracefully. None of that is possible inside `.fit()`.
:::

## The code

```python {title="Minimal PyTorch training loop" run=false}
import torch
import torch.nn as nn

# -- toy setup (swap in your real model and dataloader) --
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for inputs, targets in dataloader:          # iterate batches

    optimizer.zero_grad()                   # Step 1: clear stale gradients
                                            #   PyTorch ACCUMULATES — skip this
                                            #   and old batches corrupt new ones

    outputs = model(inputs)                 # Step 2: forward pass
                                            #   PyTorch silently builds compute graph

    loss = criterion(outputs, targets)      # Step 3: compute loss (scalar)

    loss.backward()                         # Step 4: backprop
                                            #   fills param.grad for every parameter

    optimizer.step()                        # Step 5: update weights
                                            #   reads param.grad, applies update rule
```

## What each step actually does

:::table {title="The 5 steps — what breaks if you skip one"}
| Step | PyTorch call | What it does | Symptom if skipped |
|---|---|---|---|
| 1 — Zero grads | `optimizer.zero_grad()` | Sets `.grad = 0` on all params | Gradients accumulate across batches → silent corruption, loss diverges |
| 2 — Forward | `model(inputs)` | Runs layers, builds compute graph | Nothing to differentiate |
| 3 — Loss | `criterion(out, targets)` | Single scalar summarising error | Nothing to backprop |
| 4 — Backward | `loss.backward()` | Chain-rule through graph, fills `.grad` | Weights never learn which way to move |
| 5 — Step | `optimizer.step()` | Reads `.grad`, applies update rule | Params frozen; loss stays flat forever |
:::

## Why zero_grad() lives at the top

PyTorch **accumulates** gradients. Each call to `.backward()` **adds** to whatever is already sitting in `.grad` — it does not replace it. This is intentional: gradient accumulation (summing gradients over multiple mini-batches before a single step) is a real production technique for simulating larger batch sizes on small GPUs.

The side-effect: if you forget to zero before your next batch, you are summing gradients from two different batches and the update is silently wrong. No error. No warning. Just a model that behaves strangely.

Safe habit: **zero → forward → loss → backward → step**. In that order. Every time.

:::gotcha
Placing `optimizer.zero_grad()` *after* `optimizer.step()` looks harmless and is usually fine — until you add gradient accumulation or a second optimizer. Put it at the very top of the loop body and never revisit the decision.
:::

## What .backward() is actually doing

During the forward pass, PyTorch quietly records every operation — every matrix multiply, every activation function — into a **computation graph**. It's just bookkeeping, almost free.

When you call `loss.backward()`, PyTorch walks that graph in reverse (this is *reverse-mode automatic differentiation*, or autograd) and applies the chain rule all the way back to every `nn.Parameter`. The result lands in `param.grad`.

Then `optimizer.step()` reads those `.grad` values and applies its update rule — Adam, SGD, whatever you chose. The parameters never see the loss directly. They only ever see their own gradient.

This separation — build graph → compute gradients → apply update — is exactly why you can insert gradient clipping, custom schedulers, or mixed-precision scaling between steps 4 and 5 without touching the model at all.

:::interview-line
"PyTorch accumulates gradients by design, so zero_grad() clears the slate before each batch. Skip it and you silently corrupt your updates — no error raised, just wrong results."
:::

:::qa {q="What does loss.backward() actually compute?"}
It runs reverse-mode automatic differentiation through the computation graph PyTorch built during the forward pass. For every trainable parameter it computes ∂loss/∂param and stores the result in `param.grad`. The optimizer then reads those values on the next `optimizer.step()` call.
:::

:::qa {q="Why does PyTorch accumulate gradients instead of overwriting them on each backward pass?"}
Because gradient accumulation is a deliberate technique: you can simulate a large effective batch size on a memory-limited GPU by running several forward-backward passes before calling optimizer.step(). If PyTorch zeroed grads automatically, you'd need a separate API to opt into accumulation. Explicit zeroing keeps the default behaviour consistent with the power-user case.
:::

:::drill {type="mcq" q="You forget optimizer.zero_grad() in your training loop. What is the most likely symptom?"}
- [ ] A RuntimeError is raised on the second batch
- [x] The model produces unexpected behaviour with no error message — gradients from previous batches silently accumulate
- [ ] The forward pass returns NaN on batch 2
- [ ] Loss is computed incorrectly only on the first batch
:::

:::drill {type="mcq" q="Which is the correct order of operations for one training step in PyTorch?"}
- [ ] forward → loss → backward → zero_grad → step
- [ ] zero_grad → loss → forward → backward → step
- [x] zero_grad → forward → loss → backward → step
- [ ] forward → zero_grad → loss → step → backward
:::

:::key-takeaway
Every batch runs the same five steps: zero gradients, forward pass, compute loss, backward pass, optimizer step. Know *why* each step exists and you can debug, modify, or extend any training loop — including the ones `.fit()` was hiding from you.
:::
