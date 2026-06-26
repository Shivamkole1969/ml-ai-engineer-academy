---
id: parallelism-taxonomy
track: 06-training-scale
title: "Parallelism taxonomy: DDP / FSDP / ZeRO / TP / PP"
badge: HOT
minutes: 11
prereqs: []
tags: [distributed-training, ddp, fsdp, zero, tensor-parallelism, pipeline-parallelism, pytorch]
xp: 60
hot2026: true
---

It's a Tuesday evening and your manager drops a message: "We got budget for 8 A100s — can we train the 13B model by Friday?" You've trained on a single GPU before. You've even run DDP on two. But 8 GPUs, a 13B model, and three days? That's a different beast.

You stare at PyTorch docs. Five acronyms stare back: DDP. FSDP. ZeRO. TP. PP.

This lesson gives you a clean mental model for all five — what each one splits, why you'd pick it, and what it costs. The *which one to actually choose* decision tree lives in the next lesson; here we build the vocabulary so that decision makes sense.

## The core question: what are we actually splitting?

Training a neural network has four things that eat memory and compute:

1. **Parameters** — the weights themselves
2. **Gradients** — one gradient tensor per parameter, computed in the backward pass
3. **Optimizer states** — Adam stores two extra tensors per parameter (momentum + variance). That's 3× your parameter count, in float32
4. **Activations** — intermediate outputs cached for the backward pass; scale with batch size

Every parallelism strategy is just a choice about *which of these four things to split*, and *how*.

:::why-prod
At 7B+ parameters, the optimizer states alone exceed 80 GB in float32 — way beyond a single 40 GB A100. You are not "making things faster." You are making them *possible*. Understanding the taxonomy means knowing exactly which bottleneck you're breaking.
:::

## DDP — Data Parallel, nothing shared

**What it splits:** the *data* (batches). The model is fully replicated on every GPU.

Each GPU holds a complete copy of the model and processes its own mini-batch. After the backward pass, an `all-reduce` averages the gradients across GPUs. Everyone updates their identical copy. Repeat.

Simple. Fast. Battle-tested. But the model must fit on *one* GPU. If your LLM needs 80 GB and you have 40 GB cards, DDP won't help.

PyTorch's `DistributedDataParallel` wraps any model in two lines:

```python {title="DDP — the two-line wrapper" run=false}
# pip install torch  — standard PyTorch, free to run locally with torchrun
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Each process already called dist.init_process_group() before this point.
# model is your normal nn.Module
device = torch.device("cuda", local_rank)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# From here, use model exactly like a normal model.
# DDP handles the all-reduce automatically during .backward().
```

:::gotcha
DDP duplicates the optimizer states on every GPU. With Adam on a 1B-param model, each GPU needs ~16 GB just for optimizer states. Eight GPUs = 8× that RAM, wasted. That's the exact waste ZeRO and FSDP fix.
:::

## ZeRO — Sharding the redundancy away

ZeRO (Zero Redundancy Optimizer) is DeepSpeed's answer to the DDP waste problem. Instead of each GPU holding a full copy, ZeRO partitions across three stages:

:::table {title="ZeRO Stage Cheat Sheet"}
| Stage | What gets sharded | Memory saved vs DDP | Communication cost |
|---|---|---|---|
| ZeRO-1 | Optimizer states only | ~4× | Low |
| ZeRO-2 | + Gradients | ~8× | Medium |
| ZeRO-3 | + Parameters | ~64× (linear) | High |
:::

At Stage 3, each GPU only *owns* a slice of the model. When a layer's forward pass needs its weights, a quick `all-gather` broadcasts them; when done, they're discarded. It's memory-efficient but communication-heavy — you want fast interconnects (NVLink or InfiniBand).

## FSDP — PyTorch's native ZeRO-3

FSDP (Fully Sharded Data Parallel) is PyTorch's built-in equivalent of ZeRO Stage 3. Same idea: shard parameters, gradients, and optimizer states. Each GPU holds only its slice; parameters are gathered just-in-time for computation and then discarded.

Why use FSDP over DeepSpeed ZeRO-3? No extra library. Native PyTorch means better ecosystem compatibility (Hugging Face `Trainer`, `torchrun`, etc.). The performance is nearly identical on modern clusters.

```python {title="FSDP — wrapping a transformer" run=false}
# Requires PyTorch >= 2.0; free with: pip install torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
import torch

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap AFTER moving to device; FSDP manages sharding from here.
model = FSDP(
    model,
    mixed_precision=bf16_policy,
    # auto_wrap_policy wraps each transformer block as its own FSDP unit —
    # critical for memory efficiency, otherwise the whole model gathers at once.
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

:::gotcha
Forgetting `auto_wrap_policy` in FSDP is a classic trap. Without it, FSDP treats the entire model as one unit and gathers *all* parameters before any forward pass — spiking memory to nearly DDP levels. Always wrap at the transformer-block granularity.
:::

## TP — Tensor Parallelism (split *within* a layer)

DDP, ZeRO, and FSDP all split *across* layers (or replicate them). Tensor Parallelism goes inside a single layer.

Imagine a massive matrix multiply: `Y = X @ W`. Split `W` column-wise across 4 GPUs. Each GPU does a partial multiply and contributes a shard of `Y`. A single `all-reduce` reassembles the result.

Attention heads are the natural split point: 32 heads across 8 GPUs means each GPU handles 4 heads independently. No synchronisation needed until you aggregate.

**What TP demands:** extremely fast GPU-to-GPU bandwidth. You almost always keep TP *within* a single node (NVLink is 600 GB/s; cross-node Ethernet is 25 GB/s — a death sentence for TP). Requires model surgery — your linear layers have to be written as column-parallel or row-parallel ops. Megatron-LM pioneered this.

## PP — Pipeline Parallelism (split across layers, sequentially)

Pipeline Parallelism assigns different *layers* to different GPUs in sequence. GPU 0 runs the embedding + first 8 transformer blocks; GPU 1 runs blocks 9–16; and so on. Data flows like an assembly line.

The catch: GPU 1 can't start until GPU 0 finishes its forward pass. Naive PP means most GPUs sit idle — the "pipeline bubble." The fix is **micro-batching**: chop one batch into 8 micro-batches. GPU 0 processes micro-batch 2 while GPU 1 processes micro-batch 1. Bubble shrinks to `(k-1)/k` of idle time where `k` is the number of micro-batches.

PP is the only strategy that works across nodes with *low* bandwidth links, because communication is limited to the activations at stage boundaries, not the full parameter set.

:::war-story {title="The FSDP flatten_parameters crash at 4 AM"}
A team training a 30B model on 32 A100s used FSDP but forgot that `torch.save(model.state_dict())` on an FSDP-wrapped model only saves the local shard — not the full weights. Twelve hours in, training crashed on a node failure. They tried to resume from the checkpoint. The loader raised a shape mismatch because each shard had been saved separately with no coordination. The full-parameter checkpoint strategy — using `FSDP.state_dict_type(FULL_STATE_DICT)` with offload to CPU — had been left as a "TODO." They lost the run and restarted from scratch. The next lesson covers how to checkpoint correctly so this never happens to you.
:::

:::interview-line
"DDP replicates everything — model fits one GPU, easy wins. FSDP shards parameters and optimizer states across GPUs — you pick this when the model doesn't fit a single card. TP splits within a layer and needs NVLink; PP splits layers sequentially and tolerates slow links. In practice, large jobs combine all four."
:::

:::qa {q="What is the difference between ZeRO Stage 2 and Stage 3?"}
ZeRO Stage 2 shards optimizer states AND gradients, so each GPU only stores its slice of those — but every GPU still holds a full copy of the parameters. Stage 3 goes further and shards the parameters too, with all-gathers fetching them just-in-time during forward and backward. Stage 3 unlocks training models that are far larger than any single GPU's memory, but at the cost of heavier all-gather communication.
:::

:::qa {q="When would you choose pipeline parallelism over tensor parallelism?"}
When your GPUs are spread across multiple nodes connected by standard network links rather than NVLink. Tensor parallelism requires very high bandwidth because it does all-reduces on activations inside every layer of every forward pass — on slow inter-node links, this becomes the bottleneck. Pipeline parallelism only communicates activations at the layer boundaries between stages, so the bandwidth requirement is much lower and it scales across nodes more gracefully.
:::

:::qa {q="Why does FSDP need auto_wrap_policy set at the transformer-block level?"}
FSDP works by defining "FSDP units" — groups of parameters that are gathered and discarded together. If you don't set a wrap policy, the whole model becomes one giant unit. Before any forward pass, FSDP must gather every parameter at once, spiking memory to DDP levels and defeating the purpose. Wrapping each transformer block separately means only that block's parameters are gathered at any one time, keeping peak memory proportional to one block rather than the whole model.
:::

:::drill {type="mcq" q="You're training a 7B model. Each A100 has 40 GB VRAM. The model + optimizer states in float32 require ~120 GB. Which strategy is the minimum change to make training possible?"}
- [ ] DDP across 4 GPUs — full model replica per GPU, so 120 GB per card
- [x] FSDP (ZeRO-3 equivalent) — shards parameters and optimizer states across GPUs, peak per-GPU memory drops below 40 GB
- [ ] Tensor parallelism only — splits weight matrices but doesn't shard optimizer states across the node
- [ ] Gradient checkpointing only — trades compute for activation memory, does not reduce parameter or optimizer-state memory
:::

:::drill {type="mcq" q="A team uses 4 pipeline stages with 16 micro-batches. What fraction of each forward pass is wasted as pipeline bubble?"}
- [ ] 25% — one stage idle out of four at all times
- [ ] 50% — half the time in steady state
- [x] ~19% — bubble ≈ (p-1)/m where p=4 stages and m=16 micro-batches, so 3/16 ≈ 18.75%
- [ ] 0% — micro-batching eliminates the bubble entirely
:::

:::key-takeaway
DDP replicates everything (model must fit one GPU). FSDP/ZeRO shards parameters and optimizer states (enabling models larger than one GPU). TP splits within a layer (needs NVLink, inside a node). PP splits layers sequentially (tolerates slow cross-node links). Production runs at scale nearly always combine at least two of these.
:::
