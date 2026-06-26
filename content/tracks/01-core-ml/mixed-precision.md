---
id: mixed-precision
track: 01-core-ml
title: "Mixed precision: fp32/fp16/bf16/fp8 (why bf16 won)"
badge: HOT
minutes: 9
prereqs: [gpu-memory-math]
tags: [mixed-precision, bf16, fp16, fp8, pytorch, amp, autocast, training, numerics]
xp: 60
hot2026: true
---

It's 11 PM. Your fine-tune of a 7B model has been cooking for six hours. Loss curve looks clean. Then — epoch 9 — the loss goes `nan`. You stare at it. You rerun. `nan` again. You blame your dataset, scrub it, wait another six hours. `nan` again.

Three days later someone points out you copied a 2021 tutorial that used `fp16` without a `GradScaler`. One line of a config change — switch to `bf16` — and the NaN never comes back.

That's the whole lesson, really. But let's understand *why*.

## What a float format actually is

Every number your model touches — weights, activations, gradients — is stored as bits. A floating-point number carves those bits into three fields:

- **Sign** — positive or negative (1 bit, always)
- **Exponent** — the scale, the "how big is this number" field
- **Mantissa** — the precision, the "how many decimal places" field

More exponent bits → wider range of values you can represent.
More mantissa bits → finer precision within that range.

fp32 gets 32 bits total. Half-precision formats get 16 bits. The question is how you split those 16 bits.

## Why mixed precision at all?

Pure fp32 training is safe but expensive. Every weight, gradient, and activation eats 4 bytes. On a big model that blows your GPU budget fast (the sibling lesson on GPU memory math has the exact numbers).

The trick: most of the heavy compute — the forward pass, the backward pass — does not actually need fp32 precision. You just need the **optimizer's master copy of weights** to stay in fp32, so tiny gradient updates don't get rounded away. Everything else can run in 16-bit.

That's mixed precision: low-precision arithmetic for speed, fp32 master weights for correctness.

## fp16 vs bf16 — the exponent is everything

This is the crux. Both use 16 bits. But they slice those bits differently:

- **fp16**: 5 exponent bits, 10 mantissa bits. Max value: ~65,504.
- **bf16**: 8 exponent bits, 7 mantissa bits. Max value: ~3.4×10³⁸ (same as fp32).

During training, gradients can be large — especially early on, or after a sudden loss spike. With fp16, a gradient that exceeds 65,504 overflows to `inf`. `inf` poisons the next multiply. You get `nan`. Your run dies.

bf16 keeps fp32's 8 exponent bits. Same dynamic range. Gradients never overflow. The mantissa is coarser (7 bits vs 10 in fp16), but for training that barely matters — you don't need gradients to be precise to 10 decimal places, you just need them to not be infinity.

**That is why bf16 won.** Google Brain designed it specifically for training. No overflow, no GradScaler, no late-night NaN archaeology.

## What about fp8?

fp8 (one byte per value) is the next frontier, shipped with the H100. Two variants: E4M3 (used for activations, more mantissa) and E5M2 (used for gradients, more exponent). It needs careful per-tensor scaling and is still non-default in most frameworks. Keep it on your radar, not in your config yet.

:::why-prod
bf16 cuts activation memory roughly 2× vs fp32 with no extra code and no NaN risk — that's real money saved on A100/H100 cloud jobs. At ₹500/hr for a GPU node, a 10-hour training run costs half as much at the same batch size when you're not constrained by fp32 memory.
:::

:::table {title="Float format cheat-sheet"}
| Format | Bytes | Exp bits | Mantissa bits | Max value | Training safe? | Use it for |
|---|---|---|---|---|---|---|
| fp32 | 4 | 8 | 23 | ~3.4e38 | Yes | Master weights, optimizer state |
| fp16 | 2 | 5 | 10 | ~65,504 | Risky | Inference, forward pass (with scaler) |
| bf16 | 2 | 8 | 7 | ~3.4e38 | Yes | Training on Ampere+ GPUs — default choice |
| fp8 E4M3 | 1 | 4 | 3 | ~448 | Needs scaling | Activations on H100 |
| fp8 E5M2 | 1 | 5 | 2 | ~57344 | Needs scaling | Gradients on H100 |
:::

```python {title="fp16 vs bf16 in PyTorch AMP" run=false}
import torch
from torch.cuda.amp import autocast, GradScaler

# ── Option A: fp16 — needs a GradScaler ───────────────────────────────────────
# Scaler multiplies loss before backward so gradients stay in fp16 range,
# then un-scales before the optimizer step. Fragile but works on older GPUs.

scaler = GradScaler()

with autocast(dtype=torch.float16):
    loss = model(inputs)                   # forward in fp16

scaler.scale(loss).backward()              # scaled backward pass
scaler.step(optimizer)                     # un-scales, then steps
scaler.update()                            # adjusts scale factor for next iter

# ── Option B: bf16 — just wrap and go ─────────────────────────────────────────
# No scaler needed. Same dynamic range as fp32.
# Requires Ampere or newer (A100, RTX 3090+). Free on Kaggle/Colab A100 notebooks.

with autocast(dtype=torch.bfloat16):
    loss = model(inputs)                   # forward in bf16

loss.backward()                            # clean backward, no scaling
optimizer.step()                           # master weights stay fp32 internally

# Check support before assuming:
# print(torch.cuda.is_bf16_supported())
```

:::gotcha
bf16 is not supported on all GPUs. V100s, T4s, and older consumer cards (10xx, 20xx series) do not have native bf16 hardware. If you request bf16 on an unsupported device, PyTorch may silently fall back to fp32 — your memory won't shrink and you'll be confused why. Always run `torch.cuda.is_bf16_supported()` in your training script and raise an explicit error if it returns `False` rather than letting it silently degrade.
:::

:::war-story {title="The NaN that ate three days of a startup's GPU budget"}
A small ML team in Bangalore was fine-tuning Mistral-7B for a legal-document use case. They bootstrapped off a well-shared fp16 AMP tutorial from 2021 — no GradScaler, because the original author had forgotten it too. Runs on small data looked fine. When they scaled up to their full 50K document set, gradients spiked at epoch 8 and the loss went NaN. They blamed data quality, ran cleaning pipelines, and repeated twice. On day three, a senior engineer audited the training config, spotted the missing scaler, and suggested bf16 instead. Five-minute fix. The next run trained clean through epoch 30.
:::

:::interview-line
"We default to bf16 for all training — same dynamic range as fp32, half the memory, and zero loss-scaling complexity."
:::

:::qa {q="Why does fp16 training require a GradScaler but bf16 doesn't?"}
fp16's 5 exponent bits cap its maximum representable value at around 65,504. During training, gradient magnitudes can easily exceed that, causing overflow to `inf` and then `nan`. A GradScaler compensates by scaling the loss up before the backward pass — pushing gradients into the representable range — then un-scaling before the optimizer step. bf16 keeps 8 exponent bits (identical to fp32), so it can represent the same huge values. No overflow, no scaler needed.
:::

:::qa {q="You're deploying a model for inference on a server with V100 GPUs. Which precision format do you choose and why?"}
fp16 is the right call here. V100s do not support native bf16, but they have Tensor Cores optimised for fp16 inference. For inference there are no gradient overflow concerns — inputs and activations are bounded — so fp16's limited dynamic range isn't a problem. Run with `torch.half()` or `autocast(dtype=torch.float16)` and you get 2× memory efficiency with fast Tensor Core throughput.
:::

:::drill {type="mcq" q="A colleague's training run produces NaN loss starting at epoch 4 with fp16 AMP. What is the most likely fix?"}
- [ ] Switch the entire model to fp32
- [x] Add a GradScaler, or switch autocast dtype to torch.bfloat16
- [ ] Reduce the learning rate by 10×
- [ ] Clip gradients to a maximum norm of 1.0
:::

:::drill {type="mcq" q="What specifically makes bf16 more stable than fp16 for training?"}
- [ ] More mantissa bits, giving higher numerical precision
- [ ] Smaller memory footprint per value than fp16
- [x] The same number of exponent bits as fp32, giving the same dynamic range
- [ ] Native support on all CUDA-capable GPUs including V100
:::

:::key-takeaway
bf16 won because it kept fp32's 8 exponent bits — and therefore fp32's dynamic range — while halving memory. On any Ampere or newer GPU, `autocast(dtype=torch.bfloat16)` is your default; reach for fp16 only when the hardware demands it.
:::
