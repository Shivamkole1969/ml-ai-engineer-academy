---
id: cost-toolkit
track: 13-monitoring-cost
title: "Cost toolkit: distillation, pruning, quantization, caching, routing"
badge: HOT
minutes: 11
prereqs: []
tags: [cost-optimization, quantization, distillation, pruning, caching, routing, inference, llm]
xp: 60
hot2026: true
---

Your startup's inference bill just hit ₹14 lakhs for a single month. The CTO drops a calendar invite — "Cost Review, urgent" — with no agenda and a pained emoji. You have 24 hours to come back with a concrete reduction plan that does not break the product.

Five tools. One goal: cheaper inference without wrecking quality. Here they are, in the order you should actually reach for them.

## The five tools at a glance

Every technique trades *something* for lower cost. The skill is knowing what you can safely trade given your specific workload.

:::why-prod
Inference cost scales with every user. A model that costs ₹0.80 per call is manageable at 10,000 calls a day and catastrophic at 10 million. These tools are how you stay profitable as traffic grows — or how you survive a surprise spike before your billing alert fires.
:::

:::table {title="Cost toolkit: five techniques compared"}
| Technique | What you trade | Typical saving | Effort to ship |
|---|---|---|---|
| Quantization | A tiny bit of precision | 2–4× GPU cost | Low — one config change |
| Caching | Freshness on repeated prompts | 30–70% of calls | Medium — cache infra |
| Routing | Quality on simpler queries | 40–60% per-token cost | Medium — need a classifier |
| Distillation | Upfront training time | 5–10× at inference | High — train a student model |
| Pruning | Accuracy + engineering time | 10–30% model size | High — rarely worth it alone |
:::

## Quantization: the fastest win

Your model stores every weight as a float32 — four bytes per number. INT8 stores it in one byte. INT4 in half a byte.

That is a 4–8× reduction in memory. A 13B-parameter model that needed two A100 GPUs now fits comfortably on one. GPU hours drop. Latency drops. The model still answers nearly as well — typically 95–99% of original quality on standard benchmarks.

The most common path for HuggingFace models is **4-bit NF4 quantization** via `bitsandbytes`. It takes three lines of config and zero changes to your inference code.

```python {title="Load a 7B model in 4-bit — fits on a single 16 GB GPU" run=false}
# pip install bitsandbytes transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality/size tradeoff
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "mistralai/Mistral-7B-v0.1"  # ~14 GB in fp16 → ~4 GB in 4-bit

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # spread across whatever GPUs or CPU you have
)

# Inference is identical — no other changes needed
inputs = tokenizer("Explain gradient descent simply.", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

No GPU? GGUF files via `llama.cpp` do the same thing on CPU. Runs on a MacBook. Free.

## Caching: never pay for the same answer twice

If your users ask similar questions repeatedly — FAQ bots, search assistants, report generators — a large slice of your calls are near-duplicates. You are paying full inference cost to generate the same answer, slightly rephrased, over and over.

Two levels of caching:

**Exact-match cache.** Hash the prompt string. On a hit, return the stored response instantly. Zero inference cost. Works when prompts are truly identical — same system prompt, same user text.

**Semantic cache.** Embed the incoming prompt. Find the nearest cached embedding in a vector store. If cosine similarity is above your threshold (try 0.97 first), return the cached answer. "What is your refund policy?" and "How do I get a refund?" can both resolve to the same cached response. Tools like `GPTCache` or Redis + pgvector make this straightforward.

At scale, a well-tuned semantic cache eliminates 40–60% of LLM calls outright.

## Routing: match the model to the question

Not every query needs your biggest, most expensive model. "Translate this sentence to Hindi" does not need the same model as "Audit this 60-page contract for liability clauses."

A router sits in front of your model fleet and decides:

- **Easy query** → cheap, fast small model (saves 5–10× per call)
- **Hard query** → powerful large model (full quality where it counts)

Simple routers use heuristics: prompt length, keyword detection, user tier. Smarter routers train a tiny binary classifier on "easy vs hard" examples labelled from your own traffic. The classifier itself costs essentially nothing to run — microseconds and a few KB.

## Distillation and pruning: the heavy artillery

**Distillation** trains a small *student* model to mimic a large *teacher* model — not just copying right/wrong labels, but learning from the teacher's full probability distribution over tokens. The student absorbs richer signal. DistilBERT came out 40% smaller with 97% of BERT's quality using this technique.

The catch: you need labelled training examples, a compute budget for the training run, and a task that is stable enough to be worth specialising for. Distillation pays off at millions of calls per day. At thousands, the build time rarely justifies the savings.

**Pruning** removes weights that contribute least to the output — near-zero activations that are essentially noise. Structured pruning removes whole neurons or attention heads. Unstructured pruning removes individual weights. Both shrink the model, but hurt quality more than quantization for the equivalent size reduction. In practice: quantize first, always. Prune only if you have exhausted every other option and have a team willing to manage the accuracy regression.

:::gotcha
Do not reach for distillation first. It has the highest upfront cost and only pays off at massive scale. Most teams distill before trying quantization and caching, then discover they could have hit their budget target in a day instead of a sprint. The right order is: quantize → cache → route → distill. Follow it.
:::

:::war-story {title="The search widget that bankrupted a weekend"}
A Pune fintech team shipped a "smart search" feature on a Friday afternoon. Users loved it. The problem: every keystroke fired a full LLM call to rerank results. By Sunday morning, 40,000 keystrokes had generated ₹65,000 in inference spend before anyone's billing alert fired. The fix was not a smaller model — it was a 200ms debounce (free) plus a semantic cache for the top 500 recurring queries (one afternoon of work). Monday's inference cost was 11% of Sunday's. Lesson: always profile call frequency before assuming you need a model change.
:::

:::interview-line
"We quantize first — one config flag that typically cuts GPU spend 2–4× with negligible quality loss. Then we layer a semantic cache for repeated prompts and a router to send simple queries to a cheap model. Distillation is the last resort: powerful, but a multi-week project only justified at millions of calls a day."
:::

:::qa {q="What is model distillation and when would you use it?"}
Distillation trains a small student model to match a large teacher model's output probability distributions — not just hard labels, but the full softmax over tokens. The student learns richer signal and ends up surprisingly capable for its size. You reach for it when you have a stable, high-volume task (millions of calls per day), enough labelled training data, and the compute budget for the training run. DistilBERT is the canonical example: 40% smaller, 97% of BERT quality. At lower volumes, quantization and caching get you there faster with far less work.
:::

:::qa {q="How does a semantic cache differ from an exact-match cache, and what are the risks?"}
An exact-match cache hashes the prompt and returns a stored answer only on identical input — deterministic and safe, but low hit rate on natural language. A semantic cache embeds the prompt and finds the nearest cached result by cosine similarity, so paraphrases of the same question can all hit the same cached answer. The risk is threshold tuning: set similarity too low and you return a cached answer to a genuinely different question; set it too high and you miss valid hits. Always monitor cache-hit quality with a sample of real traffic before fully trusting it in production.
:::

:::drill {type="mcq" q="A team hosts a 13B LLM on two A100 GPUs and wants to cut GPU cost by roughly 4× with minimal code changes and minimal quality loss. What should they try first?"}
- [ ] Train a 3B student model via distillation
- [x] Apply 4-bit NF4 quantization with bitsandbytes
- [ ] Run structured pruning on attention heads
- [ ] Add an exact-match cache for all prompts
:::

:::drill {type="mcq" q="Spend analysis shows 55% of your LLM API calls are near-duplicate customer support questions, slightly rephrased each time. Which technique most directly reduces this cost?"}
- [ ] Routing all queries to a smaller model
- [ ] INT8 quantization on the hosted model
- [x] Semantic caching with a tuned cosine similarity threshold
- [ ] Distilling a support-specific student model from the API responses
:::

:::key-takeaway
Quantize first — it is a single config change that cuts GPU cost 2–4× with near-zero quality loss. Layer semantic caching and query routing on top for further wins. Distillation is powerful but expensive to build; only reach for it when the simpler tools have already been maxed out.
:::
