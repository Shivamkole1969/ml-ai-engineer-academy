---
id: reproducibility
track: 06-training-scale
title: "Reproducibility and its limits"
badge: FOUNDATION
minutes: 6
prereqs: []
tags: [reproducibility, seeds, determinism, distributed-training, debugging]
xp: 30
hot2026: false
---

You wake up to a Slack message: "Yesterday's overnight run hit 91% accuracy. The re-run this morning gave 87%. Same code, same data, same GPU node. What changed?"

Nothing changed — and that's exactly the problem. ML training has a dozen hidden dice rolls built in. Until you understand them, every experiment is a coin flip you can't call twice.

## What makes training non-deterministic?

Non-determinism means: run the same code twice, get different results. It's not a bug in your model. It's the nature of modern hardware and distributed software.

Here are the five main culprits:

**1. Random seeds** — the obvious one. Dropout, weight init, data shuffling — all use pseudo-random generators. If you don't fix every seed (Python, NumPy, PyTorch — separately), you get drift.

**2. cuDNN algorithm selection** — CUDA's deep-learning library sometimes picks different convolution algorithms across runs for "best performance." Different algorithm, different floating-point rounding, different output.

**3. Floating-point ordering** — addition isn't fully associative on GPUs. `(a + b) + c` can differ from `a + (b + c)` at the 6th decimal place. With millions of parallel additions per step, those tiny differences compound.

**4. Distributed all-reduce** — during multi-GPU gradient sync, the order GPUs contribute to the average can vary by run. Different order → different float rounding → different gradient → different weights.

**5. DataLoader worker non-determinism** — `num_workers > 0` means workers are spawned as separate processes. If you don't set per-worker seeds, each worker re-initialises its RNG independently.

:::why-prod
A model that can't be reproduced can't be debugged. If an experiment that crushed your baseline can't be re-run, you can't validate it, ablate it, or put it in production confidently. Reproducibility is table stakes for shipping — not an academic nicety.
:::

## The standard fix (and what it costs)

:::table {title="Non-determinism sources and their fixes"}
| Source | Fix | Performance cost |
|---|---|---|
| Python RNG | `random.seed(42)` | None |
| NumPy RNG | `np.random.seed(42)` | None |
| PyTorch CPU | `torch.manual_seed(42)` | None |
| PyTorch CUDA | `torch.cuda.manual_seed_all(42)` | None |
| cuDNN algo | `torch.backends.cudnn.deterministic = True` | Up to 20% slower |
| cuDNN autotune | `torch.backends.cudnn.benchmark = False` | Loses the speed-up |
| DataLoader workers | `worker_init_fn` + `generator` | Tiny overhead |
:::

```python {title="Seed everything — paste this at the top of train.py" run=false}
import random, os
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """Fix all RNGs. Call before model init, dataloader creation, and first forward pass."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These make cuDNN deterministic — turn OFF if you need max throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own seeded RNG."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

# Usage
seed_everything(42)
loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(42),
)
```

## The limits — when you can't go further

Here's what the industry doesn't say loudly: even after doing all of the above, **multi-GPU runs are often not bit-for-bit reproducible across restarts.**

The reason is distributed all-reduce. Gradient contributions from different GPUs can arrive in a different order at different runs. NCCL (the collective communication library used by PyTorch DDP and FSDP) doesn't guarantee message ordering between runs, and floating-point rounding depends on order.

The practical standard isn't "identical weights." It's **"statistically reproducible"**: re-runs land within normal variance of each other — say, ±0.3% accuracy — and experiments are comparable. That's what your team should commit to, and document.

What you must always log to make this defensible:

- PyTorch version, CUDA version, cuDNN version
- The exact seed
- Number of GPUs and their type
- Any environment flags (`NCCL_*`, `CUDA_VISIBLE_DEVICES`)

A `requirements.txt` is not enough. Use `pip freeze > env.txt` and save it next to your checkpoint.

:::gotcha
`torch.backends.cudnn.deterministic = True` can silently reduce throughput by 15–25% on convolution-heavy models (ResNet, CNNs). Teams sometimes enable it in experiments, forget to disable it in production training, and wonder why the big run costs twice as much. Always benchmark with and without — and document which mode your production job uses.
:::

:::interview-line
"We treat reproducibility as statistically reproducible, not bit-for-bit — we fix all seeds, log the full environment, and accept that multi-GPU all-reduce ordering can shift weights by floating-point noise, which sits inside normal run variance."
:::

:::qa {q="Why can't you just set all the seeds and guarantee the exact same model weights every run?"}
Distributed all-reduce operations across multiple GPUs can process gradient contributions in a different order each run. Floating-point arithmetic is not fully associative, so a different summation order gives a slightly different result. The difference is usually in the sixth decimal place per step, but it compounds over thousands of steps into different final weights. This is inherent to NCCL and cannot be fixed by setting seeds alone.
:::

:::qa {q="A teammate says 'just turn on cudnn.deterministic, problem solved.' What do you say?"}
It solves the cuDNN algorithm-selection non-determinism, and it's worth doing — but it's not the whole picture. You still need to fix Python, NumPy, and PyTorch seeds separately, seed DataLoader workers, and accept that multi-GPU gradient sync may still vary slightly. Also, deterministic mode can slow convolution-heavy models by up to 20%, so you need to benchmark the cost before enabling it in a long training run.
:::

:::drill {type="mcq" q="Which of these sources of non-determinism is NOT fixed by setting torch.manual_seed(42) at the start of your script?"}
- [ ] PyTorch weight initialisation
- [ ] Dropout mask selection
- [x] NCCL all-reduce ordering across GPUs in a DDP run
- [ ] torch.randperm used inside the dataset
:::

:::drill {type="mcq" q="Your team runs the same single-GPU experiment twice with all seeds fixed and cudnn.deterministic=True. You get identical loss curves. You scale to 8 GPUs with DDP, fix the same seeds, and re-run twice. The loss curves differ by ~0.1%. What is most likely happening?"}
- [ ] The seeds were not propagated to all 8 processes
- [ ] DDP introduces dropout randomness that seeds can't control
- [x] NCCL all-reduce gradient summation order varies between runs, causing floating-point differences that compound over training
- [ ] PyTorch's DDP implementation ignores manual_seed on ranks > 0
:::

:::key-takeaway
Fix every RNG layer (Python, NumPy, PyTorch, DataLoader workers) and log the full environment — but accept that multi-GPU runs are statistically reproducible, not bit-for-bit identical, due to non-deterministic all-reduce ordering.
:::
