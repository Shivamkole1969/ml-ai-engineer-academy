---
id: few-shot
track: t3-llms-finetuning
title: "Few-shot, chain-of-thought & structured output"
badge: HOT
minutes: 9
prereqs: []
tags: [prompting, few-shot, chain-of-thought, structured-output, json-mode, llm]
xp: 60
hot2026: true
---

Your sentiment classifier has been in production for three weeks. It works fine on clean product reviews. Then one morning you get paged: the model is tagging support tickets as "positive" because customers say things like "great, another outage." Sarcasm. You don't have time to fine-tune. You have a prompt and ten minutes. What do you do? You reach for few-shot examples and chain-of-thought — and you fix it before the stand-up.

## What few-shot prompting actually is

A language model predicts the next token based on everything in its context window. Few-shot prompting exploits that: you put worked examples *right there in the prompt*, so the model pattern-matches against them instead of relying purely on its pre-training instincts.

- **Zero-shot**: no examples. Just instructions. Works for simple tasks; breaks on edge cases.
- **One-shot**: one example. Establishes format but limited signal.
- **Few-shot**: two to eight examples. Enough to cover the distribution of tricky cases you care about.

The sweet spot in production is usually **three to five examples**. More than eight and you're burning tokens for diminishing returns — and you risk a model that over-fits to your examples and fails on anything slightly different.

:::why-prod
Few-shot is your fastest escape hatch when a model misbehaves in production and retraining is not an option. A well-chosen example set can shift accuracy by 15–30 points on a classification task in under an hour of engineering work.
:::

:::table {title="Zero-shot vs Few-shot vs Fine-tune — quick comparison"}
| Approach | Setup time | Token cost | Best when |
|---|---|---|---|
| Zero-shot | Minutes | Low | Task is simple, instructions are clear |
| Few-shot | < 1 hour | Medium (examples in every call) | Edge cases, format control, fast fix |
| Fine-tune | Days–weeks | Low at inference | High volume, stable task, quality ceiling hit |
:::

## Chain-of-thought: make the model show its work

Chain-of-thought (CoT) prompting tells the model to reason out loud before giving its final answer. The classic nudge is `"Think step by step."` It sounds almost too simple — and yet it reliably improves performance on tasks that need multi-step logic: math, coding, diagnosis, legal reasoning.

Why does it work? Forcing the model to generate intermediate tokens lets it "use" those tokens as scratch space. Each token it produces becomes context for the next prediction. An answer it would get wrong in one shot it may get right when given room to reason.

Two flavours you'll actually use:

**Zero-shot CoT** — append `"Think step by step."` or `"Reason through this before answering."` to any existing prompt. No examples needed. Cheap, fast, surprisingly effective.

**Manual CoT** — write out a full reasoning chain in your few-shot examples. The model sees *how* you want it to think, not just what answer you want.

:::gotcha
CoT can produce a confident, fluent, completely wrong chain of reasoning. The model sounds authoritative even when it makes a logical error halfway through. Always validate CoT outputs on a held-out eval set before shipping. A plausible-looking scratchpad is not a guarantee of a correct answer.
:::

```python {title="Few-shot + CoT prompt builder" run=false}
# pip install openai  (free tier available at openai.com)
# Or swap for any OpenAI-compatible endpoint (Ollama, Together AI, etc.)

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

SYSTEM = """You are a sentiment classifier for customer support tickets.
Classify each ticket as POSITIVE, NEGATIVE, or NEUTRAL.
Think step by step before giving your final label on its own line as: LABEL: <value>"""

EXAMPLES = [
    {"role": "user", "content": "Ticket: 'Wow, great job breaking the login page again.'"},
    {"role": "assistant", "content": (
        "Step 1: The word 'great' is positive on its face.\n"
        "Step 2: 'again' implies this is a repeated failure.\n"
        "Step 3: The overall tone is sarcastic — the customer is frustrated.\n"
        "LABEL: NEGATIVE"
    )},
    {"role": "user", "content": "Ticket: 'Finally! The export feature works perfectly now.'"},
    {"role": "assistant", "content": (
        "Step 1: 'Finally' suggests prior frustration but current relief.\n"
        "Step 2: 'works perfectly' is a strong positive signal.\n"
        "Step 3: Overall sentiment is satisfaction.\n"
        "LABEL: POSITIVE"
    )},
]

def classify(ticket: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}] + EXAMPLES
    messages.append({"role": "user", "content": f"Ticket: '{ticket}'"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap; swap for any model you have access to
        messages=messages,
        temperature=0,        # deterministic output for classifiers
    )
    return response.choices[0].message.content

print(classify("Oh brilliant, my invoice disappeared after the update."))
# Expected: LABEL: NEGATIVE  (sarcasm handled correctly now)
```

## Structured output: getting JSON you can actually use

Classification is one thing. But most production systems need the model's output to feed into downstream code — a database write, an API call, a UI component. Free-form prose breaks everything. You need **structured output**.

Three tiers of reliability, from worst to best:

**1. "Please return JSON" in the prompt** — Works sometimes. Fails under load, with long reasoning chains, or when the model gets creative. Do not use in production.

**2. JSON mode** — Many providers (OpenAI, Anthropic, Groq) offer a parameter that forces the model to only emit valid JSON. Still does not guarantee your *schema* is correct.

**3. Constrained decoding / tool-calling with schema validation** — The gold standard. You define a schema (often via Pydantic), the model fills it, the library validates it and retries automatically. Libraries like `instructor` make this one-liner easy.

```python {title="Structured output with Pydantic + instructor" run=false}
# pip install openai instructor pydantic

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

class TicketClassification(BaseModel):
    label: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0–1")
    reason: str = Field(description="One-sentence rationale")

# instructor patches the client to add schema enforcement + auto-retry
client = instructor.from_openai(OpenAI())

def classify_structured(ticket: str) -> TicketClassification:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=TicketClassification,   # instructor enforces this
        messages=[
            {"role": "system", "content": "Classify support tickets. Be precise."},
            {"role": "user", "content": f"Ticket: '{ticket}'"},
        ],
        temperature=0,
        max_retries=2,  # instructor will retry if schema validation fails
    )

result = classify_structured("My dashboard has been down for 2 hours.")
print(result.label)       # "NEGATIVE"
print(result.confidence)  # 0.97
print(result.reason)      # "Customer reports extended service outage."
# result is a typed Pydantic object — pass it straight to your DB layer
```

:::war-story {title="The JSON that ate Saturday morning"}
A team shipped a pipeline that asked GPT-4 to return a JSON array of product recommendations. In testing, it always worked. In production, with long product descriptions, the model occasionally wrapped the JSON in a markdown code fence — ` ```json ... ``` ` — which silently crashed the `json.loads()` call downstream. The error was swallowed by a broad `except` block, and the recommendations table just stopped updating. They discovered it on a Saturday when a stakeholder noticed the widget showing stale data for 18 hours. Fix: switch to JSON mode + schema validation. The markdown fence problem vanished instantly because the model could no longer emit anything except raw JSON matching the schema.
:::

:::interview-line
"Few-shot buys you quick wins; chain-of-thought unlocks multi-step reasoning; structured output makes both usable in real systems — use all three together."
:::

:::qa {q="When would you prefer few-shot prompting over fine-tuning?"}
When you need a fast fix, have limited labeled data, or the task volume doesn't justify fine-tuning cost. Few-shot is also preferable when the task changes frequently — you update examples in a config file, not a training pipeline. Fine-tuning wins when quality has plateaued and you're running millions of calls a day.
:::

:::qa {q="What is zero-shot chain-of-thought and when does it help most?"}
It means appending a phrase like "Think step by step" to your prompt without providing example reasoning chains. It helps most on tasks that require sequential logic — math word problems, multi-hop reasoning, code debugging — where answering in one shot is too cognitively demanding even for a large model. It adds tokens but improves accuracy enough to be worth it on hard tasks.
:::

:::qa {q="Why is 'please return JSON' unreliable in production?"}
Because nothing in the model's decoding process enforces it. The model might add a preamble, wrap the JSON in a code fence, or produce subtly invalid JSON (trailing commas, missing quotes) that passes a quick glance but breaks `json.loads()`. JSON mode or constrained decoding with schema validation enforces the format at the token level, making failures near-impossible.
:::

:::drill {type="mcq" q="You add 10 few-shot examples to fix a tricky edge case. What is the most likely production downside?"}
- [ ] The model ignores examples longer than 5 items
- [ ] Few-shot examples make the model output longer text
- [x] Every API call now costs more tokens because examples are sent each time
- [ ] Few-shot disables JSON mode on most providers
:::

:::drill {type="mcq" q="Which approach gives the strongest guarantee that the model's output matches your Pydantic schema?"}
- [ ] Appending 'return only JSON' to the system prompt
- [ ] Using JSON mode alone (no schema)
- [ ] Asking the model to validate its own output
- [x] Constrained decoding via a library like instructor that enforces schema and auto-retries
:::

:::key-takeaway
Few-shot gives examples, chain-of-thought gives reasoning room, and structured output makes the result machine-readable — together they turn a raw LLM into a reliable production component without touching the model weights.
:::
