---
id: sampling-parameters
track: t3-llms-finetuning
title: "Sampling: temperature, top-k, top-p"
badge: HOT
minutes: 9
prereqs: []
tags: [llm, inference, temperature, top-k, top-p, nucleus-sampling, generation]
xp: 60
hot2026: true
---

Imagine your team ships a customer-support chatbot. In the first week, users complain it gives "robotic, repetitive" answers. Your manager asks engineering to "make it more creative." Someone bumps the temperature from 0.1 to 1.8 — and suddenly the bot starts inventing refund policies that do not exist. You roll back at 2 a.m. Sound familiar? This is a sampling parameter incident, and it happens constantly. Understanding these three knobs — temperature, top-k, and top-p — is the difference between shipping a useful model and shipping a liability.

## How an LLM actually picks the next word

Every time an LLM generates a token (roughly a word or word-piece), it produces a long list of scores — one per token in the vocabulary. Those raw scores are called **logits**. A function called **softmax** converts them into a probability distribution: numbers that sum to 1.0.

The model does NOT just pick the most likely token every time. Instead it **samples** from that distribution. Sampling parameters control the shape of that distribution before the draw happens.

Three parameters are doing most of the work in every major API you'll call:

:::why-prod
These parameters control output quality, cost, and safety simultaneously. Set them wrong and your model hallucinates, loops, or produces legally risky text — all without any code change, just a bad config. Every production LLM deployment needs an intentional default and a rationale for each value.
:::

## Temperature

Temperature is the simplest and most powerful lever. It works by dividing all logits by a single number before the softmax step.

- **Low temperature (0.1–0.5):** The highest-scoring tokens get even more dominant. Output becomes focused, predictable, and repetitive. Good for factual Q&A, JSON extraction, code generation.
- **Temperature = 1.0:** No change. The model behaves exactly as trained.
- **High temperature (1.2–2.0):** The distribution flattens. Low-probability tokens get much more likely. Output becomes creative, surprising — and increasingly incoherent.

A simple mental model: temperature = 0 means "always pick the top token" (greedy), temperature = infinity means "pick any token at random."

:::gotcha
Temperature 0 sounds safe but it causes **repetition loops**. Greedy decoding can get stuck repeating the same phrase forever once it lands in a local cycle. Always use at least a tiny non-zero value (e.g. 0.01) and set a `max_tokens` cap.
:::

## Top-k sampling

Top-k cuts the vocabulary down to the **k most likely tokens** before sampling. All other tokens are set to zero probability.

If k = 50, the model only ever considers the 50 best options for the next token. This prevents it from picking a bizarre, low-probability word that temperature might otherwise let through.

The catch: k is a fixed number. In some positions the model is very confident and the top 3 tokens have 99% of the probability — k = 50 is wasteful. In other positions probability is spread across 200 tokens — k = 50 cuts off too much.

## Top-p (nucleus sampling)

Top-p solves the fixed-k problem by being dynamic. Instead of picking the top k tokens, it picks the **smallest set of tokens whose cumulative probability reaches p**.

If p = 0.9, the model keeps adding tokens (sorted by probability, highest first) until the running total hits 90%. On a confident position that might be just 3 tokens. On an uncertain position it might be 150 tokens.

This is called **nucleus sampling** because you're always sampling from the high-confidence "nucleus" of the distribution regardless of how spread out the rest is.

:::table {title="Parameter quick-reference"}
| Parameter | What it controls | Safe default | When to tune |
|---|---|---|---|
| temperature | spread of the whole distribution | 0.7 for chat, 0.0–0.2 for structured tasks | creativity vs precision trade-off |
| top-k | hard cap on vocabulary size | 40–50 | rarely; top-p usually supersedes it |
| top-p | dynamic vocabulary by cumulative prob | 0.9–0.95 | fine-grained quality tuning |
| max_tokens | hard stop on output length | task-dependent | always set — cost + loop prevention |
:::

## Putting it together in code

```python {title="Sampling parameters with the Anthropic SDK" run=false}
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# --- FACTUAL / STRUCTURED use-case ---
# Low temperature, tight nucleus — want predictable JSON output
structured_response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=256,
    temperature=0.1,        # near-deterministic
    top_p=0.9,              # still avoids pathological tokens
    messages=[
        {"role": "user", "content": "Extract the invoice total from: 'Total due: $1,234.50'"}
    ]
)

# --- CREATIVE use-case ---
# Higher temperature, slightly looser nucleus
creative_response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=512,
    temperature=0.9,        # more variety
    top_p=0.95,             # wider nucleus
    messages=[
        {"role": "user", "content": "Write a punchy tagline for an AI-powered calendar app."}
    ]
)

print(structured_response.content[0].text)
print(creative_response.content[0].text)

# NOTE: Run locally — `pip install anthropic` then set ANTHROPIC_API_KEY.
# For a free alternative, use ollama (https://ollama.com) with the
# same parameter names in its /api/generate endpoint.
```

## When to use top-k vs top-p

In practice, most production teams pick **one or the other**, not both. Top-p has mostly superseded top-k because it adapts to the model's confidence on each token. Many hosted APIs (including Claude) only expose top-p and temperature. If an API exposes both, a common pattern is: use top-p as the primary filter, set top-k as a hard safety ceiling (e.g. k = 100) to prevent extreme outliers.

:::war-story {title="The 2 a.m. sampling rollback"}
A fintech team deployed a document-summarization pipeline using temperature = 1.5 because the outputs "felt too stiff" during demos. Everything looked fine for three days. On day four the pipeline started producing summaries that confidently stated wrong interest rates and fabricated clause numbers — real documents, hallucinated details. An on-call engineer traced it back to sampling config, not a model update. Rolling temperature to 0.2 fixed the hallucinations immediately. The incident went into their runbook as "sampling parameters are a safety surface, not just a UX dial."
:::

:::interview-line
"Temperature controls how much probability mass you give to unlikely tokens — low for precision, high for creativity, but high temperature is also a hallucination accelerator, so every production deployment needs an explicit, justified default."
:::

:::qa {q="What happens when you set temperature = 0?"}
The model always picks the single most likely token at each step — called greedy decoding. Output is fully deterministic and often high quality, but it can get stuck in repetition loops. In practice, a very low value like 0.01 is safer than exactly 0.
:::

:::qa {q="What is the difference between top-k and top-p?"}
Top-k hard-codes the number of candidate tokens (e.g. always consider the top 50). Top-p dynamically adjusts the candidate set so that their cumulative probability reaches a target (e.g. 90%). Top-p is generally preferred because it adapts to the model's confidence on each position — when the model is very sure, top-p naturally restricts to a small set; when it is uncertain, top-p stays broad.
:::

:::qa {q="Should you always lower temperature to reduce hallucinations?"}
Lower temperature makes the model more deterministic and reduces invention — so yes, it helps with factual tasks. But temperature does not remove hallucinations entirely. A model that has incorrect beliefs will express them confidently at temperature = 0. Reducing hallucinations ultimately requires better grounding (RAG, tool use) and evaluation, not just lower temperature.
:::

:::drill {type="mcq" q="Your pipeline extracts structured data (dates, amounts) from scanned invoices. Which setting is most appropriate?"}
- [ ] temperature = 1.5, top-p = 0.99
- [x] temperature = 0.1, top-p = 0.9
- [ ] temperature = 0.7, top-p = 0.5
- [ ] temperature = 1.0, top-k = 500
:::

:::drill {type="mcq" q="You set top-p = 0.1. What is the most likely result?"}
- [ ] The model becomes more creative and varied
- [ ] The model refuses to generate any tokens
- [x] The model considers only the very top token(s), producing near-greedy, repetitive output
- [ ] The model samples uniformly from the entire vocabulary
:::

:::drill {type="mcq" q="Why does top-p generally outperform top-k in production?"}
- [ ] top-p is faster to compute
- [ ] top-k was deprecated by most LLM providers
- [x] top-p adapts the candidate set size to the model's confidence, while top-k uses the same fixed count regardless of context
- [ ] top-p guarantees no hallucinations
:::

:::key-takeaway
Temperature controls how random the output is; top-p controls how wide the candidate pool is. For factual or structured tasks use low temperature (0.0–0.2) and tight top-p (0.9). For creative tasks raise both modestly. Always treat sampling parameters as a **safety surface** — document your defaults and the reasoning behind them.
:::
