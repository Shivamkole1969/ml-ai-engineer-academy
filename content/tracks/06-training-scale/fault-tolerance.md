---
id: fault-tolerance
track: 06-training-scale
title: "Preventing multi-day run failures (elastic, fault tolerance)"
badge: CORE
minutes: 8
prereqs: []
tags: [fault-tolerance, elastic-training, torchrun, torchelastic, spot-instances, nccl]
xp: 45
hot2026: false
---

It is 3 AM on day four of a six-day pretraining run. Slack buzzes: "Job exited. Exit code: 1." You open your terminal. A spot instance got preempted. All ninety-six GPU-hours since your last checkpoint: gone. Your manager's dashboard goes red. Your stomach follows.

This lesson is about making sure that never happens — or at the very least, making it a ten-minute setback instead of a four-day one.

## Why long runs die

Multi-day training runs have many ways to fail. Some are loud (OOM crash, CUDA error). Most are quiet.

The three most common culprits:

**Spot instance preemption.** Cloud providers can reclaim preemptible/spot GPUs with two minutes' notice. On a 32-node run, that is a meaningful failure probability per day. Run long enough and you *will* get hit.

**NCCL hangs.** One rank stalls during an `all-reduce`. The others sit waiting. No error is raised — the process just freezes. Your GPU utilisation metric drops to 0% but your job shows "running." You only notice when the next checkpoint never arrives.

**Silent node failure.** A host's NVLink degrades or a network card drops. The process does not crash — it just stops making progress. Other ranks block on the next collective forever.

:::why-prod
A single node failure on a vanilla DDP job kills the entire run. On large runs that last days to weeks, this is a near-certainty, not an edge case. Fault tolerance is the difference between finishing a training run and burning your cloud budget on dead-end restarts.
:::

## Elastic training with torchrun

PyTorch ships a fault-tolerant launcher called **`torchrun`** (successor to `torch.distributed.launch`). Under the hood it uses **torchelastic**.

The key idea: instead of a fixed number of workers, you declare a *range*.

```python {title="Elastic torchrun launch" run=false}
# Launch with 4–8 nodes; survive if nodes drop to the minimum
# Local single-machine test: torchrun --nproc_per_node=1 train.py
torchrun \
  --nnodes=4:8 \           # min:max nodes — job continues if >= 4 survive
  --nproc_per_node=8 \     # GPUs per node
  --rdzv_backend=c10d \    # built-in rendezvous; no extra infra needed
  --rdzv_endpoint=MASTER_ADDR:29400 \
  train.py
```

When a node leaves (preemption, crash), the remaining workers **re-rendezvous**: they elect a new coordinator, rebuild the process group, reload from the latest checkpoint, and resume. The job shrinks instead of dying.

When a replacement node becomes available (auto-scaling, new spot slot), it joins the next rendezvous and the job grows back. That is the "elastic" in elastic training.

Your training code needs two small changes to be torchelastic-compatible:

```python {title="torchelastic-compatible training loop" run=false}
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

@record   # captures and surfaces stack traces from all ranks on failure
def main():
    dist.init_process_group(backend="nccl")

    # Always load from checkpoint — not just on first run
    # (Save/load mechanics are in the "Checkpointing" lesson)
    state = load_checkpoint_if_exists("ckpt/latest.pt")

    for step in range(state.step, MAX_STEPS):
        train_step(state)
        if step % CKPT_EVERY == 0:
            save_checkpoint(state, "ckpt/latest.pt")  # frequently!
```

The `@record` decorator is small but vital. Without it, a failure on rank 3 shows up as a silent hang on rank 0, with no stack trace to debug.

## Catching silent failures before they waste hours

:::table {title="Failure modes and how to handle them"}
| Failure type | Default behaviour | Fix |
|---|---|---|
| Spot preemption | Job dies, no warning | Interruption hook → save checkpoint, then exit cleanly |
| NCCL hang | Freezes forever | Set `NCCL_TIMEOUT` (e.g. 1800s); turns freeze into clean error |
| Silent node | Freezes forever | Heartbeat/watchdog thread; kill if no step progress for N minutes |
| Rank OOM | Loud CUDA crash | Gradient checkpointing, smaller micro-batch |
| Master node crash | All workers hang | torchelastic elects a new master automatically |
:::

Set NCCL's timeout so a hang becomes a clean, restartable failure rather than an infinite freeze:

```python {title="NCCL hang detection" run=false}
import os
os.environ["NCCL_TIMEOUT"] = "1800"   # 30 min; tune to your collective size
os.environ["NCCL_DEBUG"] = "WARN"     # surfaces warnings without log spam
```

## Cloud interruption hooks

Spot preemption usually gives you 30–120 seconds of warning via a metadata endpoint. Use it to save before you die.

```python {title="AWS spot interruption watcher (daemon thread)" run=false}
import requests, threading, os

def spot_watcher(checkpoint_fn):
    """Poll AWS metadata every 5 s; save + exit on preemption notice."""
    import time
    while True:
        try:
            r = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/interruption-action",
                timeout=1
            )
            if r.status_code == 200:          # notice received
                print("Spot preemption imminent — saving checkpoint")
                checkpoint_fn()               # your save function here
                os._exit(0)                   # clean exit for torchelastic
        except Exception:
            pass
        time.sleep(5)

# Wire it up in your training setup (rank 0 only is fine):
# t = threading.Thread(target=spot_watcher, args=(save_fn,), daemon=True)
# t.start()
# GCP and Azure expose equivalent metadata endpoints — same pattern.
```

On Kubernetes or SLURM, pair all of this with an **auto-requeue policy**: when a job exits for any reason, the scheduler relaunches it. The job scheduler becomes your outer fault-tolerance loop; torchelastic handles the inner one.

:::gotcha
Saving "on exit" via `atexit` is too late. If NCCL hangs, the process never reaches an exit handler. Save on a fixed step interval *during the run*. Also, saving from all ranks simultaneously causes a thundering-herd write storm on shared storage — save from rank 0 only, or use `torch.distributed.checkpoint` for sharded async saves.
:::

:::interview-line
"We train on spot, so we run elastic torchrun jobs with NCCL timeouts and a spot-watcher thread — a preemption becomes a five-minute checkpoint-reload, not a restart from scratch."
:::

:::qa {q="What is elastic training and when would you use it?"}
Elastic training lets a distributed job resize its worker pool at runtime — nodes can leave or join without killing the job. You'd use it whenever you're training on spot or preemptible instances, or anywhere node availability isn't guaranteed. In practice, that is most large-scale training today. PyTorch's torchelastic (exposed via `torchrun`) implements this with a rendezvous mechanism and automatic process-group rebuilding.
:::

:::qa {q="A training job freezes for hours: GPU utilisation is 0%, processes are still running, no errors logged. What is the likely cause and how do you fix it going forward?"}
This is almost certainly an NCCL hang — one rank stalled during a collective (all-reduce, all-gather) and all others are blocked waiting for it. There is no error because NCCL waits indefinitely by default. The fix is to set the `NCCL_TIMEOUT` environment variable to a finite value (e.g. 1800 seconds). That turns an infinite freeze into a clean exception, which torchelastic can catch and restart from the last checkpoint.
:::

:::drill {type="mcq" q="A 32-node job freezes for 8 hours. GPU utilisation is 0% on all nodes. The job status is still 'RUNNING'. What is the most likely cause?"}
- [ ] Gradient explosion on rank 0 propagated NaN values to other ranks
- [x] An NCCL collective hung — one rank stalled and all others are blocked waiting for it
- [ ] The checkpoint save routine is unusually slow on shared storage
- [ ] The learning rate scheduler raised a divide-by-zero exception
:::

:::drill {type="mcq" q="Which is the correct way to make a torchelastic job survive if two of its eight nodes are preempted?"}
- [ ] Set --nnodes=8 and add a try/except around dist.init_process_group
- [ ] Use --max_restarts=2 with a fixed eight-node launch
- [x] Set --nnodes=6:8 so the job continues with as few as six nodes
- [ ] Set NCCL_TIMEOUT=0 to disable blocking collectives
:::

:::key-takeaway
Fault tolerance in long training runs needs two layers: elastic torchrun (so node failures shrink the job instead of killing it) and frequent in-run checkpoints with interruption hooks (so any restart loses minutes, not days).
:::
