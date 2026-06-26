---
id: peft-family
track: 10-fine-tuning
title: "PEFT Family: LoRA, QLoRA, Adapters, Prefix/Prompt Tuning"
badge: HOT
minutes: 11
prereqs: []
tags: [peft, lora, qlora, adapters, fine-tuning, llm, parameter-efficient]
xp: 60
hot2026: true
---

It's a Friday afternoon. Your manager drops a task: "Fine-tune Llama 3 8B on our support ticket dataset. We want it to reply in our brand voice." You spin up a notebook, load the model — and immediately hit a wall. Full fine-tuning a 8B-parameter model needs ~80 GB of GPU VRAM just for the weights and optimizer states. Your team's budget? One A100 (40 GB). Game over?

Not even close. This is exactly why **PEFT** (Parameter-Efficient Fine-Tuning) exists — and why every ML engineer working with LLMs in 2025 needs to know it cold.

## Why Full Fine-Tuning Is Overkill (Almost Always)

Pre-training a large language model costs millions of dollars. The base model already "knows" language, reasoning, structure. All you usually need to do is nudge its behavior for a specific task or style.

Full fine-tuning updates every single parameter — billions of weights, plus optimizer states (Adam needs 2× more). Your GPU screams and your cloud bill follows.

PEFT flips the script: **freeze almost everything, train only a tiny slice, and get 90%+ of the quality at 1–5% of the cost.**

:::why-prod
In production, you're often serving 10 different fine-tunes of the same base model (customer support, code, sales, docs). PEFT lets you swap small adapter weights at inference time without loading 10 separate 7B+ models. That's the difference between one A10G and a rack of GPUs.
:::

## The PEFT Family, One by One

### Adapters — The OG Approach

Proposed in 2019, adapters insert tiny trainable **bottleneck layers** between each transformer block. The base weights are frozen; only the adapters train.

Think of it like fitting a custom nozzle to a hose — you don't rebuild the hose, you just attach a small piece at the end. Adapters add a small inference-time overhead because you're running extra layers, but they proved the concept works.

### LoRA — The Production Default

**LoRA** (Low-Rank Adaptation, Hu et al. 2021) is the go-to approach today. Here's the key insight:

Weight updates during fine-tuning tend to live in a **low-dimensional subspace**. A 4096×4096 weight matrix doesn't need a full-rank update — a rank-8 approximation captures most of the signal.

So LoRA decomposes the weight update ΔW into two small matrices:

```
ΔW = B × A
where B is (d_out × r), A is (r × d_in), and r << d_in, d_out
```

You freeze W, train only A and B, and at inference time merge them back: `W' = W + α × (B × A)`.

**The win:** A 4096×4096 layer has 16.7M parameters. With rank 8, you train 4096×8 + 8×4096 = 65K parameters — about **0.4% as many**. That's the magic.

:::table {title="PEFT method comparison"}
| Method | Trainable Params | Extra VRAM vs base | Inference overhead | Best use case |
|---|---|---|---|---|
| Full fine-tune | 100% | 3–4× (optimizer) | None | Unlimited compute |
| Adapters | ~1–3% | Low | Small (extra layers) | Classic approach |
| LoRA | ~0.1–1% | Low | None (merge weights) | Production default |
| QLoRA | ~0.1–1% | Very low (4-bit base) | Tiny (dequant) | GPU-poor, large models |
| Prefix tuning | <0.1% | Minimal | Small (virtual tokens) | Very tight budgets |
| Prompt tuning | <0.01% | Minimal | Small (soft tokens) | Smallest possible delta |
:::

```python {title="LoRA config with PEFT + transformers (free to run on Colab T4)" run=false}
# pip install peft transformers bitsandbytes accelerate
# Free tier: Google Colab T4 GPU is enough for 7B with QLoRA

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model (frozen by default with PEFT)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",  # use 1B for free-tier testing
    torch_dtype="auto",
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,               # rank — higher = more capacity, more params
    lora_alpha=32,      # scaling factor; effective scale = alpha / r
    target_modules=["q_proj", "v_proj"],  # which weight matrices to adapt
    lora_dropout=0.05,  # optional regularisation
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Wrap model — only LoRA params will have requires_grad=True
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# prints: trainable params: 1,310,720 || all params: 1,236,047,872 || trainable%: 0.11%

# After training, merge back into base weights for zero-overhead inference:
# merged = model.merge_and_unload()
```

### QLoRA — LoRA on a Diet

QLoRA (Dettmers et al. 2023) stacks two ideas on top of LoRA:

1. **Quantize the frozen base model to 4-bit** (NF4 format — designed for normally-distributed neural weights) so it fits in far less VRAM.
2. Keep the LoRA adapters themselves in bf16 for training precision.

The gradient still flows through the frozen weights during the backward pass — the dequantization is differentiable. You lose a tiny bit of quality vs. full-precision LoRA, but you can fine-tune a 65B model on a single consumer GPU. That tradeoff is almost always worth it.

### Prefix Tuning & Prompt Tuning — Soft Tokens

These approaches don't touch the weight matrices at all. Instead, they prepend **trainable virtual tokens** to the input.

- **Prefix tuning** prepends virtual tokens at every transformer layer — more expressive.
- **Prompt tuning** only prepends to the embedding layer — simpler, fewer parameters.

Both are extremely lightweight but fall short of LoRA in quality for most tasks. You'll still see them in research papers, and occasionally on inference endpoints where even adapter weights are too large to swap.

## Choosing Your Weapon

The mental model is simple:

- **Default choice:** LoRA. Excellent quality, mergeable, no inference overhead.
- **Tight on VRAM (< 24 GB for 7B+):** QLoRA. Almost no quality drop.
- **Serving 20+ fine-tunes of the same base:** LoRA — merge and serve, or hot-swap adapters.
- **Prefix/prompt tuning:** Only when the task is extremely simple and you can't afford even adapters.

:::gotcha
The most common LoRA mistake: setting `r` too high thinking it gives "more power." In practice, rank 8–16 outperforms rank 64 on most tasks because higher rank adapters overfit on small datasets. Start low, watch your validation loss, and only increase rank if you're clearly underfitting.
:::

:::war-story {title="The OOM That Launched a QLoRA Sprint"}
A team in Hyderabad was fine-tuning Mistral 7B for a legal-document classification task. They kicked off full fine-tuning overnight on a 24 GB A10G — woke up to CUDA OOM errors at epoch 1 step 3. The next attempt used LoRA (r=8) but the base model alone ate 18 GB in bfloat16, leaving barely room for the batch. Final fix: QLoRA with 4-bit NF4 base. The model fit comfortably in 12 GB, trained in 4 hours, and hit 94% accuracy on the eval set — within 1 point of the full fine-tune they'd benchmarked externally.
:::

:::interview-line
"LoRA decomposes the weight update into two low-rank matrices, so we train 0.1–1% of parameters and merge them back for zero inference overhead — QLoRA adds 4-bit quantisation of the frozen base to cut VRAM by another 3–4×."
:::

:::qa {q="What does the rank hyperparameter r control in LoRA, and how do you choose it?"}
Rank r sets the dimension of the two adapter matrices (B and A), controlling how many trainable parameters you add. Higher r = more expressive adapter but more parameters and higher overfitting risk on small datasets. In practice, r=8 or r=16 is the right starting point for most fine-tuning tasks. Go higher only if your training loss is clearly higher than your val loss suggests you're underfitting.
:::

:::qa {q="What's the difference between LoRA and QLoRA, and when would you choose QLoRA?"}
LoRA trains small low-rank adapters on top of a full-precision frozen base. QLoRA does the same but first quantizes the frozen base model to 4-bit (NF4), which cuts VRAM by roughly 3–4×. You pick QLoRA when you can't fit even the frozen base in full precision — for example, fine-tuning a 13B+ model on a single consumer or mid-tier cloud GPU (≤24 GB VRAM).
:::

:::qa {q="Can you use LoRA adapters at inference time without any overhead?"}
Yes, if you call merge_and_unload() after training. This bakes the adapter weights (α × B × A) directly into the original weight matrices and removes the adapter modules entirely. The merged model is identical in size and speed to the original base model. If you need to swap multiple fine-tunes at runtime, you can also keep adapters unmerged and hot-swap them — slight overhead, but enables serving many fine-tunes from one base.
:::

:::drill {type="mcq" q="You want to fine-tune a 13B model on a single 24 GB GPU with minimal quality loss. Which approach is most practical?"}
- [ ] Full fine-tuning with gradient checkpointing
- [ ] Prefix tuning only
- [x] QLoRA (4-bit quantized base + LoRA adapters)
- [ ] Prompt tuning with 50 soft tokens
:::

:::drill {type="mcq" q="After LoRA training, a colleague says 'let's deploy the adapter separately so inference is faster.' What's wrong with this reasoning?"}
- [ ] Nothing — separate adapters are always faster
- [ ] LoRA adapters can't be deployed without the base model anyway
- [x] Merging adapter weights into the base (merge_and_unload) removes inference overhead entirely; keeping them separate adds a small overhead
- [ ] Inference speed depends only on quantisation, not adapter placement
:::

:::drill {type="mcq" q="Which of these is NOT a benefit of LoRA over full fine-tuning?"}
- [ ] Much lower VRAM during training
- [ ] Far fewer trainable parameters
- [x] Higher final accuracy on every task regardless of data size
- [ ] Easy to version and swap multiple fine-tunes per base model
:::

:::key-takeaway
LoRA is the production default for fine-tuning LLMs: freeze the base, train tiny low-rank matrices on the attention weights, and merge them for zero inference overhead. When VRAM is the bottleneck, add 4-bit quantization (QLoRA) to go even further — same technique, smaller footprint.
:::
