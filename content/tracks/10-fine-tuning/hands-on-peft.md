---
id: hands-on-peft
track: 10-fine-tuning
title: "Hands-on: transformers + peft + trl (SFTTrainer), 4-bit"
badge: HOT
minutes: 12
prereqs: []
tags: [peft, lora, qlora, sft, trl, transformers, bitsandbytes, fine-tuning]
xp: 60
hot2026: true
---

Your PM sends a Slack at 6 PM: "Can we make the assistant answer exactly like our internal docs — concise, structured, no hallucinated steps?" You have one T4 GPU on a Colab Pro notebook and a 600-row JSONL file. Full fine-tuning of a 7B model would need ~60 GB VRAM for weights + gradients + optimizer state. You have 16 GB. This is the exact moment QLoRA + SFTTrainer was built for.

## The Four-Library Stack

The sibling lesson covers the theory of LoRA and QLoRA. Here we focus on wiring the libraries together correctly — because the gotchas are in the glue, not the concepts.

:::why-prod
Production fine-tuning runs at 4-bit not because it's trendy but because it's the only way a startup can iterate on a 7B+ model without a six-figure GPU bill. The adapter files are also ~20 MB — trivially versioned in Git and swappable per customer.
:::

:::table {title="Library responsibilities at a glance"}
| Library | What it does | Key object you touch |
|---|---|---|
| `transformers` | Loads base model + tokenizer | `AutoModelForCausalLM`, `AutoTokenizer` |
| `bitsandbytes` | 4-bit NF4 quantization (the "Q" in QLoRA) | `BitsAndBytesConfig` |
| `peft` | Injects trainable LoRA adapters into frozen weights | `LoraConfig`, `get_peft_model` |
| `trl` | Training loop optimized for LLM SFT | `SFTTrainer`, `SFTConfig` |
:::

## The Code, Step by Step

The full pipeline below fits on one T4 (16 GB). Comments explain every non-obvious choice.

```python {title="QLoRA fine-tune with SFTTrainer" run=false}
# pip install transformers peft trl bitsandbytes datasets accelerate -q

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ── 1. 4-bit quantization ──────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 preserves outlier weights better than int4
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4-bit
    bnb_4bit_use_double_quant=True,      # quantize the quant constants too (~0.4 GB saved)
)

model_id = "meta-llama/Llama-3.2-1B-Instruct"  # swap to 3B/7B on bigger GPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # <-- CRITICAL (see gotcha below)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",       # spreads across GPU/CPU if needed
)
model.config.use_cache = False  # must disable when using gradient checkpointing

# ── 2. LoRA adapter config ─────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                              # rank: higher = more capacity = more VRAM
    lora_alpha=32,                     # scaling; rule of thumb: alpha = 2 * r
    target_modules=["q_proj", "v_proj"],  # which attention projections to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 2,097,152 || all params: 1,235,814,400 || trainable%: 0.17

# ── 3. Dataset — ChatML / messages format ─────────────────────────────────
# Each JSONL row: {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")

# ── 4. Training ────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch size = 2 * 4 = 8
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=1024,
    dataset_text_field="messages",   # tells SFTTrainer which column to use
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# ── 5. Save adapters only (not the full 2 GB base model) ──────────────────
model.save_pretrained("./lora-adapters")   # ~20 MB on disk
tokenizer.save_pretrained("./lora-adapters")
```

## What Each Knob Actually Does

**Rank `r`** controls how many parameters the adapter adds. `r=8` is lean and fast. `r=64` is rich but approaches full fine-tune territory. For most SFT tasks on <1 K rows, `r=16` is the starting point.

**`lora_alpha`** scales the adapter output before adding it to the frozen layer. The effective learning rate of the adapter is `alpha / r`, so doubling `r` while doubling `alpha` keeps the same effective scale. Just set `alpha = 2 * r` and stop thinking about it.

**`target_modules`** decides which Linear layers receive adapters. Query and value projections (`q_proj`, `v_proj`) give the best quality-per-parameter ratio for most tasks. You can add `k_proj`, `o_proj`, and the MLP layers if you need more capacity.

**`gradient_accumulation_steps`** is your memory lever. If you hit OOM, halve the batch size and double accumulation steps — the math is identical, the peak VRAM is not.

:::gotcha
Two silent killers in every first QLoRA run:

1. **Missing `pad_token`.** LLaMA-family tokenizers ship without a `pad_token`. Without `tokenizer.pad_token = tokenizer.eos_token`, SFTTrainer silently pads with `None`, gives you NaN loss after step 1, and you spend 40 minutes re-reading the LoRA paper thinking you got the math wrong.

2. **Forgetting `model.config.use_cache = False`.** KV-cache and gradient checkpointing are mutually exclusive. Leave `use_cache=True` (the default) and you get a cryptic warning then corrupted gradients. Set it to `False` before training, flip it back to `True` before inference.
:::

:::war-story {title="The NaN That Wasn't LoRA's Fault"}
A team in Pune spent an afternoon debugging "unstable QLoRA training" — loss went to NaN at step 1 every single run. They tried lower learning rates, different ranks, even swapped the base model. The actual culprit: their JSONL had two rows where the `assistant` turn was an empty string `""`. SFTTrainer computed a cross-entropy loss over zero tokens and produced NaN, which poisoned the entire batch. A one-line filter — `dataset.filter(lambda x: len(x["messages"][-1]["content"]) > 0)` — fixed it instantly. Always validate your dataset before you touch a hyperparameter.
:::

:::interview-line
"QLoRA lets me fine-tune a 7B model on a single T4 by quantizing frozen base weights to 4-bit NF4 and only training tiny rank-16 LoRA adapters — about 0.2% of parameters — then saving just the 20 MB adapter file, not the full model."
:::

:::qa {q="Why do you freeze the base model weights during LoRA fine-tuning?"}
Because the base model is quantized to 4-bit, which makes gradients through it numerically unstable and slow. LoRA sidesteps this entirely: gradients flow only through the small adapter matrices, which stay in full precision (bf16). You get targeted task adaptation without touching the frozen weights at all.
:::

:::qa {q="How would you load the LoRA adapters for inference after training?"}
Load the original base model normally (no quantization needed for inference if you have VRAM), then call `PeftModel.from_pretrained(base_model, './lora-adapters')`. Alternatively, merge the adapters permanently into the base weights with `model.merge_and_unload()` — this produces a single standard model file and removes the PEFT dependency at inference time.
:::

:::drill {type="mcq" q="After QLoRA training, `model.print_trainable_parameters()` shows 0.17% trainable. What does this mean for the saved adapter file size vs the base model?"}
- [ ] The adapter is 0.17% smaller than the base model
- [x] Only the adapter weights are saved, so the file is roughly 0.17% the size of the full model
- [ ] The base model is also saved but compressed to 0.17% of its original size
- [ ] Nothing — `save_pretrained` always saves all parameters
:::

:::drill {type="mcq" q="Your loss is NaN at step 1. You have verified the learning rate and rank are sane. What is the FIRST thing to check?"}
- [ ] Increase gradient_accumulation_steps to stabilize gradients
- [ ] Switch from NF4 to int8 quantization
- [x] Inspect the dataset for empty assistant turns or malformed rows
- [ ] Reduce lora_alpha below r
:::

:::key-takeaway
The QLoRA + SFTTrainer pipeline is four imports and ~50 lines. The power is in the defaults — freeze + quantize the base, train tiny adapters, save 20 MB. The failures are almost never in the math; they are in dataset quality and two forgotten config flags: `pad_token` and `use_cache=False`.
:::
