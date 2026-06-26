---
id: sft-instruction
track: 10-fine-tuning
title: "SFT & instruction tuning: data format & data quality"
badge: HOT
minutes: 10
prereqs: []
tags: [sft, instruction-tuning, data-quality, llm, fine-tuning, chat-template]
xp: 60
hot2026: true
---

It's your first week at a Pune-based AI startup. The CTO demos the fine-tuned Llama 3 they trained on 50,000 customer support tickets. It should answer product questions crisply. Instead, every response starts: *"We have escalated your issue to our L2 team. You will hear back in 3–5 business days."* Someone asked how to reset a password. The model learned the data format perfectly — unfortunately, 60% of the corpus was escalation templates.

That's SFT in one cautionary tale. The model does exactly what the data says. So the data had better say the right thing.

## What SFT actually is

**SFT** stands for Supervised Fine-Tuning. You take a pretrained base model — one that already knows grammar, facts, and code structure — and show it paired examples: *prompt → ideal response*. The model trains to predict the response tokens, conditioned on the prompt. Standard next-token prediction, just on curated pairs instead of raw internet text.

**Instruction tuning** is SFT where the prompts are instructions. "Summarize this paragraph." "Write a SQL query that..." "Explain gradient descent to a 10-year-old." The base model already knows how to write text; you're teaching it *which* text belongs after a human instruction.

Two things come out of this step:
1. The model follows instructions reliably.
2. It replies in the tone and format your product needs.

Both live or die on your data quality.

:::why-prod
A base model given a prompt is likely to continue it, not answer it — it treats everything as raw text to autocomplete. SFT is the step that makes a model "chat-ready." In production, the gap between a base model and an instruction-tuned one is the gap between "weird autocomplete" and "deployable product."
:::

## The three data formats you'll meet in the wild

Every SFT library expects a specific shape. Mix them up and your training silently breaks — no error, just a model that was trained on malformed input.

:::table {title="Common SFT data formats"}
| Format | Structure | Common in |
|---|---|---|
| Alpaca | `instruction`, `input`, `output` keys in a dict | Classic fine-tune scripts, older HF community examples |
| ShareGPT | `conversations` list → each item has `from` and `value` | Vicuna, many community datasets on HF Hub |
| ChatML / messages | `messages` list → each item has `role` and `content` | Mistral, Llama-3, Qwen, TRL's SFTTrainer |
:::

**ChatML / messages is now the standard.** If you're starting fresh in 2025, use it — TRL's `SFTTrainer` expects it by default, and every major open model uses it under the hood.

The critical step most tutorials skip: **applying the chat template**. Every instruction model has special tokens that wrap each conversation turn — `[INST]`, `<|im_start|>`, `<|eot_id|>`, and so on. These tokens tell the model where instructions end and responses begin. Skip them during training, and the model never learns that boundary. Inference then breaks in subtle, very-hard-to-debug ways.

```python {title="Format data and apply the chat template" run=false}
from datasets import Dataset
from transformers import AutoTokenizer

# Works with Mistral, Llama-3, Qwen, Zephyr — swap to any instruction model.
# Free: run on Colab T4 or locally with a small model like zephyr-7b-beta.
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

raw = [
    {
        "instruction": "Explain what a transformer is in one sentence.",
        "response": "A transformer uses self-attention to relate every token to every other token in the sequence.",
    },
    # ... add more examples
]

def to_chat_format(example):
    messages = [
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    # apply_chat_template injects the model's special tokens correctly.
    # tokenize=False → returns a plain string ready for SFTTrainer's `text` column.
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list(raw)
dataset = dataset.map(to_chat_format)

# Output looks like:
# <s>[INST] Explain what a transformer... [/INST] A transformer uses...</s>
print(dataset[0]["text"])
```

## Data quality is the actual job

500 clean, consistent, diverse examples outperform 50,000 noisy ones. The InstructGPT paper that launched this whole era used ~13,000 human-curated examples. Quality, not volume.

Here is what "quality" actually means in practice:

- **Consistent format** — same system prompt style, same response length expectation, no mixed Alpaca/ChatML in the same run.
- **Correct responses** — obvious, but: a wrong answer trains the model to be *confidently* wrong. That is worse than no example.
- **Diversity** — cover all the behaviors you want. A dataset of only easy questions produces a model that fumbles the hard ones.
- **No format leakage** — strip boilerplate like "Dear Customer," ticket IDs, HTML tags, forwarded-email chains, internal names.
- **Healthy length distribution** — if 90% of your responses are one sentence, the model learns terseness everywhere, even where a detailed answer is needed.

A fast audit: sample 50 random examples and read them as if you're a new hire getting a style guide. If you'd be embarrassed, fix the data before touching the GPU.

:::gotcha
There are actually two separate calls to `apply_chat_template` — one for training, one for inference — and they differ. During training, you pass the full conversation including the assistant response. At inference, you want the model to *start* writing the response, so you pass `add_generation_prompt=True`: `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. Forgetting this flag at inference time makes the model either echo the prompt or output endless garbage.
:::

:::war-story {title="The compliance hallucination audit"}
A fintech team in Bangalore shipped a fine-tuned model for internal regulatory Q&A. First week in production, auditors flagged it citing regulation section numbers that do not exist. Root cause: 800 of the 4,000 training examples came from a legal blog that mixed real citations with illustrative hypotheticals marked only by context — no structural difference. The model learned the *pattern* of citing confidently, including the made-up ones. Two days of data cleaning, a re-run, and an eval harness fixed it. Lesson: garbage in, hallucinations out.
:::

:::interview-line
"Data quality is the actual fine-tuning job. Hyperparameters are a rounding error compared to fixing your training set."
:::

:::qa {q="What is instruction tuning and how does it differ from pretraining?"}
Pretraining teaches a model language by predicting the next token across a massive corpus — no task structure, just raw text. Instruction tuning (a form of SFT) then shows the model paired instruction → ideal response examples, training it to follow directions and reply in a specific style. The base model supplies the knowledge; SFT shapes the behavior on top of that knowledge.
:::

:::qa {q="Why does the chat template matter in SFT, and what happens if you skip it?"}
Every instruction model uses special tokens to mark the boundary between user turns and assistant turns. `apply_chat_template` inserts those tokens correctly. If you skip it during training, the model never learns where instructions end — at inference it may continue the prompt instead of answering it, loop indefinitely, or refuse to stop generating. The bug is silent during training and loud during serving.
:::

:::drill {type="mcq" q="You have 2,000 carefully written instruction-response pairs and 40,000 auto-scraped pairs of unknown quality. What is the best first step?"}
- [ ] Train on all 42,000 — more data always helps a fine-tune.
- [x] Audit a sample of the 40,000 and filter aggressively before deciding whether to combine them.
- [ ] Use only the 2,000 — adding noisy data can only hurt.
- [ ] Mix them 50/50 and watch the validation loss to decide.
:::

:::drill {type="mcq" q="What does `tokenizer.apply_chat_template(messages, tokenize=False)` return?"}
- [ ] A list of integer token IDs ready to pass directly to the model.
- [ ] A JSON object preserving the role/content structure.
- [x] A plain string with the model's special tokens correctly wrapped around each conversation turn.
- [ ] A padded tensor batch at the model's max sequence length.
:::

:::key-takeaway
SFT teaches a pretrained model to follow instructions — and the data format plus data quality are the entire game. Always apply the chat template, audit your examples before the GPU run, and remember: 500 clean examples beat 50,000 messy ones every time.
:::
