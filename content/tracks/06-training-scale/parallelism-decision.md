---
id: parallelism-decision
track: 06-training-scale
title: "The DDP→FSDP→TP→PP decision (how to choose, out loud)"
badge: HOT
minutes: 9
prereqs: []
tags: [parallelism, DDP, FSDP, tensor-parallel, pipeline-parallel, distributed-training, pytorch]
xp: 60
hot2026: true
---

Your team gets an email at 10 PM: 8 × A100-80GB nodes are yours for the next 72 hours. The model is 30B parameters. Your manager pings: "Which parallelism do we use?" The wrong answer wastes 50,000 rupees of cloud credits and a night of sleep. The right answer is a four-question decision tree you can run in about 90 seconds.

## The four-question decision tree

The parallelism strategies are not alternatives — they are a ladder. You climb only as high as you need to.

**Question 1: Does the model (plus optimizer states) fit on a single GPU?**
If yes → use DDP and stop climbing. DDP is the lowest overhead, the easiest to debug, and the most familiar to every senior engineer who reviews your code. One full model copy per GPU; gradients averaged across GPUs via an all-reduce after each backward pass.

**Question 2: OOM even at batch_size=1?**
That means the model weights alone exceed GPU memory. Climb to FSDP (or ZeRO-3). FSDP shards parameters, gradients, and Adam optimizer states across all GPUs. Each GPU holds only a slice; slices are gathered on-demand during forward and backward, then immediately discarded. A 7B model that needs ~56 GB of optimizer state suddenly lives comfortably across 4 × 40GB GPUs.

**Question 3: Is a single layer still too big for one GPU?**
With very wide models — hidden_dim ≥ 8192, say — even one transformer weight matrix can hit tens of GBs. Now you need Tensor Parallelism (TP): split the weight matrix column-wise or row-wise across GPUs. TP GPUs exchange intermediate activations on every forward/backward pass. This demands NVLink bandwidth (300–600 GB/s inside a node). Ethernet is far too slow. TP must stay *within a single node*.

**Question 4: Is the model still too large even across one full node?**
Climb to Pipeline Parallelism (PP). Split groups of consecutive layers across different nodes. Each node processes its stage, then passes activations to the next node in line. PP can span nodes on slower inter-node links (InfiniBand or even 100 Gb Ethernet). The cost: pipeline bubbles — GPUs idle while waiting for the upstream stage to finish. Micro-batching (splitting one big batch into many small chunks) shrinks the bubble but adds scheduling complexity.

In real large-scale training, strategies combine. Megatron-LM uses DP + TP + PP together (called 3D parallelism). PyTorch-native FSDP + TP is the modern equivalent. But you only add an axis when memory or hardware forces you.

:::why-prod
Choosing the wrong strategy crashes your run, wastes node-hours, and leaves your team debugging at 3 AM instead of analyzing curves. TP on a cluster without NVLink makes training 3–5× slower than PP would have been. Knowing exactly when to escalate — and when NOT to — is one of the clearest signals interviewers use to separate people who have shipped distributed training from people who have only read about it.
:::

:::table {title="Parallelism strategy at a glance"}
| Strategy | What gets sharded | GPU communication | Cross-node? | Climb here when… |
|---|---|---|---|---|
| DDP | Nothing — full copy per GPU | Gradient all-reduce | Yes | Model + optimizer fits on 1 GPU |
| FSDP / ZeRO-3 | Params + gradients + optimizer state | Param gather + scatter | Yes | OOM at batch_size=1 |
| Tensor Parallel | Weight matrices | Activation all-reduce (every layer) | NVLink only (intra-node) | A single layer exceeds GPU VRAM |
| Pipeline Parallel | Layer groups (stages) | Activation send/recv (stage boundaries) | Yes, any interconnect | Model too large for one full node |
:::

```python {title="Quick check — which strategy do I need?" run=false}
# Run locally before spinning up a cluster.
# pip install torch  (free, no GPU needed for this estimate)

import torch

def estimate_memory_gb(param_count: int) -> dict:
    """
    Rough estimate for mixed-precision training with Adam.
    - Weights in bf16: 2 bytes/param
    - Gradients in bf16: 2 bytes/param
    - Adam m, v, master weights in fp32: 4 bytes each → 12 bytes/param
    """
    weights_gb   = (param_count * 2)  / 1e9
    grads_gb     = (param_count * 2)  / 1e9
    optimizer_gb = (param_count * 12) / 1e9
    return {
        "weights_gb": weights_gb,
        "grads_gb": grads_gb,
        "optimizer_gb": optimizer_gb,
        "total_gb": weights_gb + grads_gb + optimizer_gb,
    }

# ---- configure these ----
param_count = 30e9            # 30B model
per_gpu_vram = 80.0           # A100-80GB
num_gpus = 8                  # GPUs in your cluster
# -------------------------

mem = estimate_memory_gb(param_count)
total_cluster_gb = per_gpu_vram * num_gpus

print(f"Weights alone   : {mem['weights_gb']:.1f} GB")
print(f"Gradients       : {mem['grads_gb']:.1f} GB")
print(f"Optimizer states: {mem['optimizer_gb']:.1f} GB")
print(f"Total needed    : {mem['total_gb']:.1f} GB")
print(f"Cluster capacity: {total_cluster_gb:.1f} GB ({num_gpus} × {per_gpu_vram:.0f} GB)")
print()

if mem["total_gb"] <= per_gpu_vram:
    print("→ DDP: fits on a single GPU. Keep it simple.")
elif mem["total_gb"] <= total_cluster_gb:
    print("→ FSDP: shard across all GPUs. Start here before adding TP or PP.")
else:
    print("→ Need TP and/or PP. Check if single layers fit; if not, add TP intra-node.")
```

:::gotcha
FSDP's default `limit_all_gathers=False` pre-fetches the next layer's parameters before the current layer finishes — good for throughput, but it causes a temporary ~2× memory spike mid-forward. If you're already tight on VRAM, set `limit_all_gathers=True` in your FSDP config. You trade a few percent of throughput for not crashing at hour 47 of a 72-hour run.
:::

:::war-story {title="The FSDP migration that still OOM'd"}
A team migrated their 13B model from DDP to FSDP and hit OOM on the first validation step — after celebrating that training was fine. Two hours of frantic debugging later: their custom DataLoader was producing full fp32 activation tensors instead of bf16. FSDP had correctly sharded the *parameters*, but the unsharded intermediate activations ballooned memory during validation (no gradient checkpointing, longer sequences). The fix was a single `.to(torch.bfloat16)` call. The lesson: FSDP shards weights, not activations. Activation memory is your problem — and gradient checkpointing is almost always worth turning on alongside FSDP.
:::

:::interview-line
"I start with DDP. If I'm OOM at batch_size=1, I move to FSDP. If single layers still exceed VRAM, I add Tensor Parallelism within a node. If the model still doesn't fit, I add Pipeline Parallelism across nodes and micro-batch aggressively to shrink the bubble."
:::

:::qa {q="When would you choose FSDP over DDP, and what communication cost does it add?"}
FSDP is the right move when the model plus Adam optimizer states exceed per-GPU VRAM at batch_size=1 — you simply cannot train with DDP at that point. FSDP shards everything across N GPUs, so each GPU holds roughly 1/N of the memory footprint. The cost is that parameters must be gathered from all GPUs before each layer's forward pass and re-sharded immediately after. You minimize this with overlapping communication and computation, which PyTorch FSDP does automatically with `backward_prefetch`.
:::

:::qa {q="Why can't you use Tensor Parallelism across nodes connected by Ethernet?"}
Tensor Parallelism requires an all-reduce of activations at every transformer layer — in both the forward and backward pass. Over NVLink inside a node (300–600 GB/s), that operation takes microseconds. Over 100 Gb Ethernet between nodes, the same data transfer takes 10–50× longer, turning your GPUs from fast compute units into slow network clients. Pipeline Parallelism is designed for cross-node work precisely because it only sends activations once per stage boundary, not on every layer.
:::

:::drill {type="mcq" q="You are training a 70B model on 16 × A100-80GB GPUs (two 8-GPU nodes, connected by InfiniBand). Individual transformer layers fit within 80 GB. Which strategy is most appropriate?"}
- [ ] DDP — 16 × 80 GB = 1.28 TB total, easily enough headroom
- [ ] Tensor Parallelism across both nodes — more bandwidth means more speed
- [x] FSDP across all 16 GPUs — shards weights, gradients, and optimizer states within the available GPU budget
- [ ] Pipeline Parallelism only — split layers evenly across both nodes
:::

:::drill {type="mcq" q="Your colleague says: 'For any model over 7B, just go straight to 3D parallelism (DP + TP + PP).' What is the main problem with this advice?"}
- [ ] 3D parallelism is only available in Megatron-LM, not PyTorch
- [ ] TP and PP are the same thing and cannot be combined
- [x] Each extra parallelism axis adds communication overhead and operational complexity — you should only climb as far as memory forces you
- [ ] DDP cannot be combined with TP or PP under any circumstances
:::

:::key-takeaway
Run the four-question decision tree: DDP first, FSDP when OOM, Tensor Parallelism within a node when a single layer exceeds VRAM, Pipeline Parallelism across nodes when nothing else fits. Every extra axis costs communication and complexity — only pay for what memory actually forces you to.
:::
