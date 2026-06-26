---
id: gpu-memory-math
track: 01-core-ml
title: "GPU memory math & the 16-bytes/param rule"
badge: HOT
minutes: 10
prereqs: []
tags: [gpu, memory, training, adam, mixed-precision, vram]
xp: 60
hot2026: true
---

Interviewer: *"We want to fine-tune a 7B model. Will it fit on one A100 with 80GB?"*

Most people guess. **You** are going to answer in five seconds with a number — and watch the
interviewer's eyebrow go up. The trick is one little rule.

## The 16×N rule (memorize this, thank me later)

For **full training** of a model with **N** parameters using **Adam** in **mixed precision**,
you need roughly:

$$\text{VRAM} \approx 16 \times N \text{ bytes}$$

Why 16? Because training keeps **five copies** of model-ish data floating around. Imagine each
parameter has an entourage:

:::table {title="Where the 16 bytes/param go (Adam, mixed precision)"}
| Item | Bytes/param | What it's for |
|---|---|---|
| fp16 weights | 2 | the fast working copy |
| fp16 gradients | 2 | this step's update direction |
| fp32 master weights | 4 | the precise copy that actually learns |
| Adam momentum (m) | 4 | running average of gradients |
| Adam variance (v) | 4 | running average of gradient² |
| **Total** | **16** | |
:::

So a **7B** model in full training ≈ `16 × 7e9` = **112GB**. That does *not* fit on one 80GB
GPU. Now you know — and you can say *why*.

:::why-prod
This single number decides your whole infra plan: one GPU vs eight, whether you need FSDP/ZeRO
sharding, whether QLoRA on a free Colab T4 is the move. Get it wrong and you either over-provision
(burn money) or crash mid-run (burn time).
:::

Try the scenarios — flip between training and inference and watch it fit or blow up:

:::widget {name="vram"}
:::

## Inference is a totally different (cheaper) story

At **inference** you drop the entourage — no gradients, no optimizer states. You just need the
weights:

- fp16 → **2 bytes/param** (7B ≈ 14GB)
- int8 → **1 byte/param** (7B ≈ 7GB)
- int4 → **0.5 bytes/param** (7B ≈ 3.5GB — fits on a gaming GPU!)

:::gotcha
Inference VRAM isn't *just* the weights — you also pay for the **KV cache**, which grows with
context length and concurrency. The weights are the floor, not the ceiling. (See the KV-cache
lesson.)
:::

:::war-story {title="The training run that died at hour 9"}
An engineer kicked off a 13B full fine-tune on a single 80GB GPU, walked away, came back to a
crash. 13B full training ≈ `16 × 13e9` ≈ 208GB — there was never a universe where it fit. Two
minutes of napkin math up front would have sent them straight to QLoRA (4-bit base + tiny LoRA
adapters), which *would* have fit. The lesson: do the 16×N math **before** you launch, not after.
:::

:::interview-line
"Full training is about 16 bytes per parameter with Adam — 2+2 for fp16 weights and grads, 4 for
fp32 master, and 4+4 for Adam's m and v. Inference is just 2 for fp16, or 0.5 for int4."
:::

:::qa {q="Why is training ~8× heavier than fp16 inference for the same model?"}
Inference needs only the fp16 weights (2 bytes/param). Training adds fp16 gradients, an fp32
master copy, and Adam's two optimizer states — about 16 bytes/param total, roughly 8× more.
:::

:::qa {q="Your 70B model won't fit for training. First move?"}
Shard the states across GPUs with FSDP or DeepSpeed ZeRO (and/or offload the optimizer to CPU).
If you only need adaptation, QLoRA freezes a 4-bit base and trains tiny adapters — far less memory.
:::

:::drill {type="mcq" q="Roughly how much VRAM to FULLY train a 7B model with Adam in mixed precision?"}
- [ ] ~14 GB
- [ ] ~28 GB
- [x] ~112 GB
- [ ] ~7 GB
:::

:::drill {type="mcq" q="A 7B model at int4 inference needs about…"}
- [x] ~3.5 GB (0.5 bytes/param)
- [ ] ~14 GB
- [ ] ~112 GB
:::

:::key-takeaway
Full training ≈ **16 × N bytes** (fp16 w+g, fp32 master, Adam m+v). Inference is cheap:
fp16 ≈ 2N, int8 ≈ N, int4 ≈ 0.5N — plus KV cache on top. Do the math *before* you launch.
:::
