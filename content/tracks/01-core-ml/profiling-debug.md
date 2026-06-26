---
id: profiling-debug
track: 01-core-ml
title: "Profiling & the on-call debugging table (OOM, NaN, low util)"
badge: CORE
minutes: 9
prereqs: []
tags: [profiling, debugging, oom, nan, gpu, pytorch, on-call]
xp: 45
hot2026: false
---

It is 2 am. Your training run just died at epoch 12 of 50. Slack says "process exited with code 1." The only other clue is a wall of CUDA error text scrolling off the screen. You're on-call. The model ships tomorrow.

Three bugs kill more training runs than anything else: **OOM** (out of memory), **NaN loss**, and **low GPU utilisation**. Each one has a tell, a likely cause, and a fix. Knowing this table cold turns a 3-hour debugging session into a 10-minute one.

## Why profiling is not optional

Most engineers treat profiling as the thing you do *after* the model works. Wrong order. You profile to find out *why* it does not work — and to prove your fix actually fixed it. A hunch is not a root-cause analysis.

:::why-prod
A training run that crashes at step 50,000 wastes hours of expensive GPU time and blocks your whole team. Profiling the first 100 steps costs five minutes and saves you from that 2 am Slack ping in the first place.
:::

## The profiling toolkit

Before you can debug, you need data. Three instruments, in order of reach:

**`nvidia-smi dmon`** — run this in a second terminal alongside your script. It shows GPU utilisation, memory use, and temperature every second. If utilisation bounces between 30 % and 95 %, your data pipeline is the bottleneck, not the model.

**`torch.cuda.memory_summary()`** — prints a breakdown of every tensor currently allocated on the GPU. Call it right before the line that throws OOM. You will see exactly which allocation is eating memory.

**PyTorch Profiler** — the serious tool. It records CPU and GPU time per op, spots bottlenecks, and exports a timeline you can open in TensorBoard or Chrome's trace viewer.

```python {title="Minimal PyTorch Profiler — profile the first few steps" run=false}
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = ...   # your model, already on CUDA
loader = ...  # your DataLoader

# Profile only the first few batches — not the whole run.
# schedule: skip 1 step (wait), warm up for 1, then record 3 active steps.
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
) as prof:
    for step, batch in enumerate(loader):
        with record_function("forward"):
            out = model(batch["input_ids"].cuda())
        out.loss.backward()
        prof.step()          # advance the profiler schedule
        if step >= 5:
            break

# Option A — view in TensorBoard (free, local):
#   pip install torch_tb_profiler
#   tensorboard --logdir ./prof_logs
# Option B — quick table in the terminal:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## The on-call debugging table

:::table {title="OOM · NaN · Low-util: diagnose and fix"}
| Symptom | Most likely cause | First command | Typical fix |
|---|---|---|---|
| `CUDA out of memory` | Batch too large, or tensors leaking into a Python list | `torch.cuda.memory_summary()` | Reduce batch size; call `.item()` or `.detach()` before storing scalars; enable `gradient_checkpointing` |
| Loss → `NaN` or `inf` after N steps | Exploding gradients or fp16 overflow on large activations | Log grad norms each step; `torch.isnan(loss)` check | Clip gradients (`max_norm=1.0`); add LR warmup; switch fp16 → bf16 |
| GPU util < 50 % | CPU DataLoader is starving the GPU | `nvidia-smi dmon` in a second terminal | Raise `num_workers`; set `pin_memory=True`; use `persistent_workers=True` |
:::

### Debugging OOM — detach before you store

The most common OOM trap is appending tensors to a list inside the training loop. Each tensor keeps its entire computation graph alive. By the time you hit batch 500, the graph is gigantic.

```python {title="OOM trap vs. fix" run=false}
# BAD — the computation graph stays alive for every loss value
running_losses = []
for batch in loader:
    loss = model(batch["input_ids"].cuda()).loss
    running_losses.append(loss)          # graph never freed!

# GOOD — .item() detaches the scalar and moves it to CPU
running_losses = []
for batch in loader:
    loss = model(batch["input_ids"].cuda()).loss
    running_losses.append(loss.item())   # tiny float, no graph
```

### Debugging NaN — follow the gradient trail

NaN loss almost always comes from one of three places: a learning rate that is too large, fp16 overflow on a large activation, or bad input data (zeros inside a `log`, an out-of-range index). Check gradient norms before you change anything else. If they are `inf`, you have exploding gradients.

```python {title="Log gradient norms inside the training loop" run=false}
loss.backward()

total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
# > 10 is suspicious. > 100 almost certainly means trouble.
print(f"grad norm: {total_norm:.4f}")

# Clip BEFORE the optimizer step:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

### Debugging low utilisation — the hidden tax

A high-end GPU sitting at 30 % is just an expensive space heater. This is almost always a DataLoader problem: the GPU finishes a batch and then *waits* for the CPU to process the next one. Fix the pipeline first before touching model architecture.

Quick wins in `DataLoader`:

- `num_workers=4` (start at the number of CPU cores / 2)
- `pin_memory=True` — skips one extra memory copy on transfers to GPU
- `persistent_workers=True` — avoids respawning worker processes each epoch
- Move heavy augmentation from CPU to GPU using `torchvision` CUDA transforms or Kornia

:::gotcha
Gradient checkpointing reduces peak memory by roughly 30–40 % — useful when you are OOM. But it recomputes activations during the backward pass, which costs about 20 % extra compute. Enable it to fit a larger model, not as a free memory trick that speeds things up.
:::

:::interview-line
"When I see OOM, my first move is `torch.cuda.memory_summary()` — it shows me which tensor is holding memory, so I fix the root cause instead of just halving the batch size and hoping."
:::

:::qa {q="A training run's loss suddenly becomes NaN at step 2,000. Walk me through your debugging process."}
I start by checking whether inputs or labels contain NaN or Inf before the forward pass, using `torch.isnan` and `torch.isinf`. Then I log gradient norms for a few steps to see if they are exploding. If norms are very large, gradient clipping usually fixes it. If the loss was stable in fp32 but goes NaN in fp16, I switch to bf16, which has a far wider dynamic range and almost never overflows on typical activations.
:::

:::qa {q="Your GPU shows 25 % utilisation during training. What are you looking for and how do you fix it?"}
That pattern almost always means the GPU is idle, waiting for data from the CPU. I run `nvidia-smi dmon` to confirm utilisation is dropping between batches, then check the profiler to see how much time is spent in the DataLoader vs the forward/backward pass. The first fixes are `num_workers`, `pin_memory=True`, and `persistent_workers=True`. If those do not help, I look for expensive CPU-side augmentation that could be moved to the GPU with CUDA-backed transforms.
:::

:::drill {type="mcq" q="You see `CUDA out of memory` on step 1, even with a small batch. The most informative first action is:"}
- [ ] Halve the batch size and retry
- [x] Call `torch.cuda.memory_summary()` to see exactly which tensors are allocated
- [ ] Enable gradient checkpointing immediately
- [ ] Switch from fp32 to fp16
:::

:::drill {type="mcq" q="Training loss is stable for 800 steps then suddenly becomes `inf`. Which of these is LEAST likely to be the root cause?"}
- [ ] Exploding gradients triggered by an unusually large batch
- [ ] fp16 overflow on a large intermediate activation
- [x] The DataLoader `num_workers` value is set too high
- [ ] A division-by-zero inside a custom loss function
:::

:::key-takeaway
Three bugs kill training runs: OOM, NaN, and low GPU utilisation. Each has a one-command diagnostic — `memory_summary`, gradient norm logging, `nvidia-smi dmon` — and a clear fix. Reach for the diagnostic first, before you touch a single hyperparameter.
:::
