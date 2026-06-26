---
id: grad-accumulation
track: 06-training-scale
title: "Gradient accumulation (+ the two traps)"
badge: CORE
minutes: 7
prereqs: []
tags: [training, gradients, memory, batch-size, ddp, pytorch]
xp: 45
hot2026: false
---

Your fine-tuning job kicks off at 11pm. The paper uses batch size 128, but your A100 (40 GB) can only fit 16 samples before OOM. No problem — you set `accumulation_steps = 8` and go to sleep. By morning the loss is spiking and the run is dead. The model wasn't wrong. The learning rate wasn't wrong. One missing division killed it.

That's gradient accumulation in a nutshell: powerful, easy to set up, and hiding two traps that fail silently.

## What gradient accumulation actually does

Normal training is: forward → loss → backward → `optimizer.step()` → `zero_grad()`. One batch, one weight update.

Gradient accumulation delays the update. Instead of stepping after every batch, you run N forward+backward passes on smaller "micro-batches" first. PyTorch quietly **adds** each pass's gradients into every parameter's `.grad` buffer. After N passes, you call `optimizer.step()` once, as if one large batch had arrived.

Memory stays bounded by the micro-batch size. Gradient quality reflects the larger effective batch. No extra GPUs needed.

:::why-prod
Very small batches produce noisy loss estimates and can make training unstable or converge slower. Most training recipes are written for batch 256 or 512. Gradient accumulation lets a single GPU match that quality without requiring 8× more hardware budget.
:::

:::table {title="True large batch vs. gradient accumulation"}
| Property | True batch = 128 | Accumulate 8 × micro-batch 16 |
|---|---|---|
| Peak VRAM per forward | High — loads 128 samples | Low — loads 16 samples |
| Gradient quality | Exact average over 128 | Same — only if you normalize correctly |
| Wall-clock per update | Faster (all samples parallel) | Slower (8 serial passes) |
| BatchNorm statistics | Computed over 128 | Computed over 16 — careful |
:::

## Trap 1 — the loss that lies to you

When you call `.backward()` eight times, PyTorch **sums** the gradients. But a true batch of 128 would **average** the loss. Eight mini-batch averages summed together are 8× larger than one big-batch average.

The result: gradients are N times too large. This acts exactly like multiplying your learning rate by N — except you don't see that in your config, so it's invisible until the loss spikes or training becomes unreasonably sensitive to LR.

**Fix**: divide the loss by `accumulation_steps` before calling `backward()`.

```python {title="Correct loss normalization for gradient accumulation" run=false}
# Run this on a single GPU first to verify the loss curve is sane.
# pip install torch  (free, local)

import torch
import torch.nn as nn

accumulation_steps = 8
optimizer.zero_grad()

for micro_step, batch in enumerate(micro_batches):
    outputs = model(**batch)
    loss = criterion(outputs.logits, batch["labels"])

    # ← Trap 1 fix: normalise BEFORE backward, not after
    loss = loss / accumulation_steps
    loss.backward()

# Gradients now reflect the average over all micro-batches
optimizer.step()
optimizer.zero_grad()
```

One line. Trivial to write. Easy to forget. The resulting bug looks like a bad LR schedule — not like a missing division.

## Trap 2 — DDP talks too much

If you're on multi-GPU DDP — which is the norm at any real ML shop — every `.backward()` call triggers an **all-reduce** across all GPUs to sync gradients. With 8 accumulation steps you're firing 8 all-reduces per optimizer update, when you only need 1 (the final step).

The extra syncs don't corrupt gradients. But they waste inter-GPU bandwidth. On cloud instances without NVLink, this can make your accumulated run slower than it has to be — sometimes dramatically.

**Fix**: wrap all but the last micro-step in `model.no_sync()`.

```python {title="Gradient accumulation in DDP — skip unnecessary all-reduces" run=false}
# Works with torch.nn.parallel.DistributedDataParallel.
# HuggingFace Accelerate does this for you automatically.
import contextlib

optimizer.zero_grad()

for micro_step, batch in enumerate(micro_batches):
    is_last = (micro_step == accumulation_steps - 1)

    # no_sync() suppresses the DDP all-reduce on this backward pass
    sync_ctx = contextlib.nullcontext() if is_last else model.no_sync()

    with sync_ctx:
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
        loss = loss / accumulation_steps   # Trap 1 fix still needed
        loss.backward()

optimizer.step()
optimizer.zero_grad()
```

HuggingFace `Trainer` and `accelerate` handle both traps for you. The moment you step outside them — custom loop, debugging, an interview whiteboard — you need to know why.

:::gotcha
BatchNorm computes mean and variance per forward pass, not per effective batch. Each micro-batch of 16 uses its own statistics, not the combined statistics of all 128 samples. If your architecture is BatchNorm-heavy, the BN layers see a much smaller batch than you intended even with perfect loss normalization. Transformers use LayerNorm, which avoids this entirely. For CNNs on gradient accumulation, consider SyncBatchNorm or treat BN statistics as approximate.
:::

:::interview-line
"Gradient accumulation simulates large batches within limited VRAM — the key disciplines are dividing the loss by accumulation steps before backward, and in DDP, using no_sync() to suppress all-reduces on every step except the last."
:::

:::qa {q="Why do you divide the loss by accumulation_steps before calling backward?"}
PyTorch sums gradients across multiple backward calls. Without normalization, N accumulated micro-batches produce gradients N times larger than a single true large batch would. This silently inflates the effective learning rate by N, causing loss spikes or training instability. Dividing before backward keeps gradient magnitude consistent with the equivalent large-batch pass.
:::

:::qa {q="What is model.no_sync() in DDP, and when should you use it during gradient accumulation?"}
In DDP, every backward() triggers an all-reduce to synchronize gradients across all ranks. no_sync() is a context manager that suspends that synchronization for the wrapped backward call. During gradient accumulation, only the final micro-step needs to sync — wrapping intermediate steps in no_sync() eliminates N-1 unnecessary all-reduces per optimizer update, reducing communication overhead without changing correctness.
:::

:::drill {type="mcq" q="You run gradient accumulation over 4 steps with micro-batch size 16, but forget to divide the loss. What is the most likely symptom?"}
- [ ] A CUDA out-of-memory crash on the second micro-step
- [ ] Training proceeds normally — PyTorch normalizes automatically
- [x] Loss spikes or diverges, behaving as if the learning rate were 4× too high
- [ ] BatchNorm statistics become NaN
:::

:::drill {type="mcq" q="In DDP with gradient accumulation over 6 micro-steps, how many all-reduce operations should fire per optimizer update when implemented correctly?"}
- [ ] 6 — one per backward call, always
- [ ] 0 — DDP buffers and defers them automatically
- [ ] 3 — DDP halves the syncs by default
- [x] 1 — only on the final micro-step, using no_sync() on the first 5
:::

:::key-takeaway
Gradient accumulation is two lines of discipline: divide the loss by accumulation steps before backward, and in DDP wrap all intermediate steps in model.no_sync(). Miss either and training silently degrades in ways that look like unrelated problems.
:::
