---
id: checkpointing
track: 06-training-scale
title: "Checkpointing that survives a crash (resumable ≠ weights-only)"
badge: CORE
minutes: 8
prereqs: []
tags: [training, checkpointing, fault-tolerance, optimizer-state, pytorch]
xp: 45
hot2026: false
---

Day 4 of a 3-day training run. Your GPU VM on AWS gets preempted at step 48,200. You had saved checkpoints every hour — but you only saved `model.state_dict()`. You restore the weights, but your Adam optimizer restarts from scratch: zero momentum, zero variance estimates, learning rate back at warmup. The model re-learns what it already knew for the first 2,000 steps, and your loss curve makes a little U-turn you can't explain to your manager. You just lost 4 hours of effective compute.

The fix is not "checkpoint more often." It's "checkpoint everything."

## What actually needs to be in a resumable checkpoint

Most engineers save the model weights. That is table stakes. A *resumable* checkpoint is a snapshot of the **entire training state** — everything that would change between step N and step N+1 if you kept going. Miss one piece and you get silent divergence, not a loud crash.

Here is the complete checklist:

:::table {title="Checkpoint contents: what to save and why"}
| Component | Why it matters if you skip it |
|---|---|
| `model.state_dict()` | Obvious — you lose all learned weights |
| `optimizer.state_dict()` | Adam stores per-param momentum (m) and variance (v). Without them the optimizer resets and takes many steps to re-warm |
| `lr_scheduler.state_dict()` | Scheduler tracks `last_epoch` and step count. Skip it and your LR curve jumps to the wrong position |
| `scaler.state_dict()` | AMP `GradScaler` tracks loss scale history. Without it mixed-precision training may waste steps re-probing the right scale |
| `epoch` / `global_step` | You need to know *where* you are so you can skip already-seen data |
| RNG states (`torch`, `cuda`, `numpy`, `random`) | Without these, dropout patterns and data augmentation differ from what the model saw, causing subtle irreproducibility |
| DataLoader sampler state | Ensures you resume from the *next unseen* batch, not from batch 0 |
:::

:::why-prod
A preempted spot instance, a node OOM, or even a routine cluster maintenance window can kill your job mid-run. Without a truly resumable checkpoint, you pay for GPU time twice: once to reach step N, and once to re-converge after the invisible damage from a bad resume.
:::

## Writing a complete checkpoint in PyTorch

```python {title="save_checkpoint.py — complete resumable checkpoint" run=false}
import torch, random, numpy as np
from pathlib import Path

def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,          # pass None if not using AMP
    global_step: int,
    epoch: int,
    dataloader_sampler=None,
):
    """
    Saves a fully resumable checkpoint.
    Write to a temp file first, then rename — atomic on Linux/macOS.
    """
    tmp_path = path + ".tmp"

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        # RNG states — needed for exact reproducibility of dropout / augmentation
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),  # list if multi-GPU
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        # Sampler state lets you skip already-seen batches on resume
        "sampler": dataloader_sampler.state_dict() if dataloader_sampler else None,
    }

    torch.save(state, tmp_path)
    Path(tmp_path).rename(path)   # atomic rename — avoids a corrupt half-written file


def load_checkpoint(path: str, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt["scaler"]:
        scaler.load_state_dict(ckpt["scaler"])
    # Restore RNG states
    torch.set_rng_state(ckpt["rng_torch"])
    torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    return ckpt["global_step"], ckpt["epoch"]
```

The atomic rename on the last line is not optional. If your process dies *while writing*, you end up with a half-written file that `torch.load` cannot open. Rename is atomic at the OS level, so you either have the old checkpoint or the new one — never a corrupt one.

## How often, how many, and where

Save every **N steps** (not every epoch — epochs can be hours long). A common default is every 500–2,000 steps depending on job length. Keep the **last K checkpoints** (K=3 is typical) plus the single **best-validation checkpoint** separately — you will want to roll back to that one after a divergence.

In multi-node training with FSDP or DeepSpeed, each rank holds a shard of the model and optimizer state. Call `torch.distributed.checkpoint.save` (PyTorch >= 2.1) instead of plain `torch.save` — it writes one file per rank in parallel and is dramatically faster for large models.

:::gotcha
Saving only the model weights is the single most common checkpointing mistake. Engineers reach for `torch.save(model.state_dict(), path)` because it is the example in every tutorial, but that is a **weights-only snapshot, not a resume point**. If your job dies and you restore from it, your optimizer silently re-warms, your scheduler resets, and your loss temporarily increases. You will almost certainly not notice unless you plot the loss curve carefully.
:::

:::interview-line
"A resumable checkpoint must include the optimizer state, scheduler, AMP scaler, RNG states, and the global step — not just the model weights. I also write to a temp file and do an atomic rename so a mid-write crash never corrupts the checkpoint."
:::

:::qa {q="What goes wrong if you restore only model weights and not the optimizer state?"}
Adam maintains running estimates of gradient mean (m) and variance (v) for each parameter — these are what make Adam adaptive. If you restore without them, Adam starts those estimates at zero and takes many steps to re-warm, often causing a visible loss spike or a slow re-convergence period. In a long training run this can cost hours of GPU time.
:::

:::qa {q="Why save RNG states in a checkpoint?"}
Training uses stochastic operations: dropout masks, data augmentation transforms, batch shuffling. If you resume without the exact RNG state, those random decisions differ from what would have happened in the original run. The model still trains, but it is no longer the *same* run — which matters for reproducibility, debugging, and for matching validation numbers across restarts.
:::

:::qa {q="How does checkpointing work differently with FSDP?"}
With FSDP each rank holds a shard of the model and optimizer state. Calling plain `torch.save(model.state_dict())` forces rank 0 to gather the full model in memory — fine for small models, OOM for large ones. PyTorch's `torch.distributed.checkpoint` API writes one shard file per rank in parallel, skipping the gather entirely. Loading is symmetric: each rank loads its own shard.
:::

:::drill {type="mcq" q="You restore a checkpoint and notice the training loss spikes for ~2,000 steps before resuming its downward trend. What is the most likely cause?"}
- [ ] The model weights were saved incorrectly and are partially corrupted
- [x] The optimizer state was not restored, so Adam's momentum terms re-warmed from zero
- [ ] The learning rate scheduler resumed at the wrong epoch
- [ ] Batch normalization running stats were missing from the checkpoint
:::

:::drill {type="mcq" q="You are using PyTorch AMP (mixed precision). Which component of your checkpoint handles the loss scale history?"}
- [ ] `lr_scheduler.state_dict()`
- [ ] `model.state_dict()` — the scaler is part of the model graph
- [x] `scaler.state_dict()` from `torch.cuda.amp.GradScaler`
- [ ] No checkpoint component — the scaler always re-initialises safely
:::

:::drill {type="mcq" q="You write a checkpoint file on a Linux node. The process is killed mid-write. What technique ensures you always have a valid checkpoint, never a corrupt partial file?"}
- [ ] Using `torch.save` with `_use_new_zipfile_serialization=False`
- [ ] Flushing the file handle with `f.flush()` before closing
- [x] Writing to a `.tmp` file and then doing an atomic `os.rename` to the final path
- [ ] Wrapping the save in a try/except and deleting the file on exception
:::

:::key-takeaway
A resumable checkpoint is not the model weights — it is the model, optimizer, scheduler, AMP scaler, RNG states, and global step together. Miss any one of them and your resume silently diverges. Always do an atomic write (temp file then rename) so a mid-write crash never corrupts your only checkpoint.
:::
