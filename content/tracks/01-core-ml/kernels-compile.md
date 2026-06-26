---
id: kernels-compile
track: 01-core-ml
title: "Kernels, fusion, torch.compile, FlashAttention"
badge: HOT
minutes: 10
prereqs: []
tags: [gpu, kernels, kernel-fusion, torch-compile, flash-attention, performance, training]
xp: 60
hot2026: true
---

Your training run is humming. Loss is ticking down. You feel good. Then you check `nvidia-smi` — GPU utilization: **34%**. The card cost you (or your company) a small fortune, and it's idle two-thirds of the time. Not because the model is wrong. Not because the data pipeline is slow. Because every tiny PyTorch operation is launching a separate kernel, bouncing data through the slowest memory on the chip, then doing it all over again for the next op.

This lesson gives you the mental model and the tools to fix it: **kernel fusion**, **`torch.compile`**, and **FlashAttention**.

## What Is a GPU Kernel, and Why Should You Care?

A **kernel** is a function that runs on thousands of GPU threads simultaneously. When you write `x = torch.relu(h)`, PyTorch launches a CUDA kernel behind the scenes. Simple.

The problem is overhead. Each kernel launch costs roughly 5–20 µs of fixed tax — just to *start* the thing. Chain 500 ops together and you've burned up to 10 ms before a single useful multiply happens.

But launch overhead is only half the story. The bigger villain is **memory bandwidth**.

Your GPU has two kinds of memory:
- **HBM** (High Bandwidth Memory) — the "main" GPU RAM. Slow to access. A100 can move ~2 TB/s, which sounds fast until you realize your kernels are doing millions of tiny reads per second.
- **SRAM** — tiny on-chip cache, maybe 20 MB total, but ~10× faster than HBM.

Most individual ops are **memory-bandwidth bound**: they read a tensor from HBM, do a few multiplies, write the result back to HBM. The next op reads that result right back from HBM. Over and over.

:::why-prod
On an A100 cluster, a training run that takes 4 days instead of 2 days costs double the cloud bill — easily ₹2–5 lakh extra per run. Fixing kernel inefficiency isn't micro-optimization; it's direct business impact. Interviewers at Sarvam, Microsoft India, and well-funded AI startups ask about this.
:::

## Kernel Fusion: Stop Paying the Middleman

Fusion merges multiple ops into one kernel. Instead of: load from HBM → `linear` → store to HBM → load from HBM → `add bias` → store to HBM → load from HBM → `relu` → store to HBM, you do: load once from HBM → run all three ops in SRAM → store once.

:::table {title="Fused vs Unfused: HBM Round-Trips"}
| Operation | Unfused | Fused |
|---|---|---|
| Linear (matmul) | Read X, W → write Y | Read X, W → stay in SRAM |
| Add Bias | Read Y → write Y+b | Compute in SRAM |
| ReLU | Read Y+b → write output | Compute in SRAM → write output |
| **Total HBM trips** | **6** | **2** |
:::

Same FLOP count. Dramatically less memory traffic. Faster wall-clock time.

## torch.compile: Fusion Without Rewriting Anything

PyTorch 2.x ships a compiler that handles fusion automatically. You get it in one line:

```python {title="torch.compile — drop-in speedup" run=false}
import torch
import torch.nn.functional as F

model = MyTransformer().cuda().to(torch.bfloat16)

# Wrap once — PyTorch 2.x traces your model, fuses kernels via TorchInductor,
# and generates optimized Triton/CUDA code on the first call.
model = torch.compile(model)

# First forward pass: slow (~30–90 s for large models) — this is the compile step.
out = model(x)

# All subsequent passes use the compiled, fused graph.
out = model(x)   # now measurably faster

# For inference only, try mode="reduce-overhead" for extra aggression.
# If your input shapes vary a lot: torch.compile(model, dynamic=True)
```

The compilation is one-time per process. Typical training throughput gain: **1.3×–2×** with zero architecture changes. The compiler traces your ops, builds a computation graph, identifies fusable clusters, and generates tight kernels — all automatically.

:::gotcha
`torch.compile` silently falls back to eager (uncompiled) mode when it hits unsupported patterns: data-dependent control flow, ops with dynamic shapes that change every step, or certain custom C++ extensions. You pay the compilation cost and get *zero* speedup. Set `TORCH_LOGS="+inductor"` to see exactly what got compiled vs skipped. Always benchmark before and after to confirm the gain.
:::

## FlashAttention: When the N×N Matrix Becomes the Enemy

Standard self-attention computes `softmax(QKᵀ / √d) · V`. The catch: `QKᵀ` produces a full **N × N** matrix. At sequence length 32k, that's 32k × 32k × 2 bytes (fp16) = **2 GB per layer per batch item**. OOM is guaranteed before you've done any useful work.

FlashAttention (Dao et al., 2022) rewrites attention from scratch to be **IO-aware**:

1. Tile Q, K, V into small blocks that fit in SRAM.
2. Compute softmax *incrementally* across tiles using the online softmax trick.
3. On the backward pass, **recompute** the attention scores from Q, K, V instead of storing them.

Result: **O(N) HBM memory** instead of O(N²). You can run 100k-token sequences on the same card that OOM'd at 8k tokens before.

In PyTorch ≥ 2.0, calling `F.scaled_dot_product_attention` automatically dispatches to FlashAttention when your setup supports it (CUDA, fp16/bf16, causal or no mask). You usually get it for free if your attention code already uses this function.

:::war-story {title="48 hours → 26 hours, six lines of code"}
A team training a 7B model on 8×A100s saw GPU utilization stuck at 42% for days. The model was correct; the bill was growing. Someone added `model = torch.compile(model)` and replaced the hand-rolled attention with `F.scaled_dot_product_attention`. Utilization jumped to 68%. The training run that was projected to take 48 hours finished in 26. The saving on that single run: roughly ₹80,000 in compute credits. No architecture change. No retraining from scratch. Six lines.
:::

:::interview-line
"torch.compile fuses kernel launches and cuts HBM round-trips; FlashAttention rewrites attention to be O(N) in memory instead of O(N²) — together they're the first two levers I pull for training throughput before touching the architecture."
:::

:::qa {q="What does 'memory-bandwidth bound' mean, and why does kernel fusion help?"}
A kernel is memory-bandwidth bound when it spends more time reading/writing HBM than doing actual math. Fusion helps because instead of each op reading and writing back to slow HBM, fused ops keep intermediate tensors in fast on-chip SRAM across the entire chain — drastically reducing total memory traffic without changing the computation.
:::

:::qa {q="When would you NOT use torch.compile?"}
During active debugging (compiled traces are harder to inspect), when your model has heavy data-dependent control flow that changes every batch, or when input shapes shift every step and you haven't enabled `dynamic=True`. Also avoid it for short-lived inference servers where the one-time compilation latency matters more than throughput. Always measure: a fallback-heavy compile run is strictly worse than eager.
:::

:::drill {type="mcq" q="Why does FlashAttention reduce peak GPU memory for long sequences?"}
- [ ] It quantizes the attention matrix to int8 during the forward pass
- [ ] It prunes low-scoring attention pairs before computing softmax
- [x] It tiles Q, K, V to fit in SRAM and recomputes attention during the backward pass rather than storing the full N×N matrix in HBM
- [ ] It uses gradient checkpointing to offload attention weights to CPU
:::

:::drill {type="mcq" q="You add torch.compile to your model. Epoch 1 runs slower than before. Epoch 2 is faster. What explains this?"}
- [ ] torch.compile only optimizes even-numbered epochs
- [x] The first forward pass triggers JIT compilation; the compiled fused graph is cached and used for all subsequent passes
- [ ] PyTorch is warming up CUDA cores during epoch 1
- [ ] The optimizer needs one epoch to align with the compiled graph
:::

:::key-takeaway
GPU kernels are cheap to write but expensive to launch repeatedly with HBM round-trips. Kernel fusion (automated by `torch.compile`) and IO-aware attention (FlashAttention via `F.scaled_dot_product_attention`) cut that overhead — often delivering 1.5×–2× faster training with no changes to your model design.
:::
