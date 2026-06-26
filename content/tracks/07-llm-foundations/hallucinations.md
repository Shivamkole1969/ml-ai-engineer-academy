---
id: hallucinations
track: 07-llm-foundations
title: "Where hallucinations originate & layered mitigations"
badge: HOT
minutes: 9
prereqs: []
tags: [hallucination, rag, grounding, evals, reliability, genai]
xp: 60
hot2026: true
---

Your startup just launched a customer-facing chatbot. First day in production, a user asks about your refund policy. The bot answers — confidently, fluently, with zero hedging — and it's completely wrong. The return window it cited doesn't exist. You spend the next morning apologising in support tickets and the CTO is asking why the model "lied."

It didn't lie. It did exactly what it was built to do. That's the uncomfortable truth about hallucinations.

## What the model is actually doing

Large language models are **next-token predictors**. At every step, the model looks at the tokens so far and asks: *what token is most statistically likely to come next, given everything I saw during training?*

That's it. There's no lookup table of facts. No internal Wikipedia. No "am I sure about this?" gate. The model compresses billions of training examples into weights, and at inference time it generates text that *looks like* a plausible continuation of the conversation.

Plausible ≠ true.

When the model's training data doesn't contain the right answer, the weights find the nearest pattern and continue from there — fluently, confidently, and incorrectly. That confident tone isn't arrogance; it's just how next-token prediction sounds.

:::why-prod
Hallucinations are the #1 production incident type in GenAI apps. A wrong confident answer erodes user trust immediately. In finance, legal, or health contexts it can also create real liability. You cannot ship an LLM product without understanding this and building mitigations into the architecture — not just the prompt.
:::

## Why the model sounds so certain

Here's the part that throws everyone off: the model has no idea it's wrong. Confidence is a property of *fluency*, not *accuracy*. A model trained on polished text learns that authoritative-sounding sentences are what coherent text looks like. So it generates authoritative-sounding sentences — even when it's making things up.

The four root causes are worth knowing cold:

:::table {title="Hallucination root causes"}
| Root cause | What happens | Typical example |
|---|---|---|
| Knowledge gap | Fact was never in training data | Real-time prices, your internal docs, events after cutoff |
| Conflation | Two similar facts get merged | Quote from person A attributed to person B |
| Context drift | Model loses track of an earlier constraint as the context grows long | Early "don't mention competitors" forgotten 8000 tokens later |
| Prompt ambiguity | Vague question leaves a gap the model fills creatively | "Summarise our policy" when no policy was provided |
:::

## The layered mitigation stack

No single fix eliminates hallucinations. You layer defences. Think of it like production security: depth beats any one silver bullet.

**Layer 1 — Prompt hygiene (free, always try first)**

Force the model to cite its sources and to say "I don't know" explicitly:

```python {title="Citation-enforcing system prompt" run=false}
SYSTEM_PROMPT = """
You are a helpful assistant for AcmeCorp support.

Rules (never break these):
1. Only answer using the CONTEXT block below. Do not use outside knowledge.
2. If the context doesn't contain the answer, respond with exactly:
   "I don't have that information. Please contact support@acmecorp.in"
3. When you answer, end with: "Source: [quote the exact sentence you used]"

CONTEXT:
{retrieved_docs}
"""

# Quick sanity check — did the model actually cite something?
def has_citation(response: str) -> bool:
    return "Source:" in response

# Run it
response = llm.complete(SYSTEM_PROMPT.format(retrieved_docs=docs) + user_question)
if not has_citation(response):
    # Flag for review or fall back to human agent
    log_to_review_queue(user_question, response)
```

**Layer 2 — Retrieval (RAG) for knowledge gaps**

If the model doesn't know, give it the facts at runtime. Covered in depth in the LoRA vs RAG lesson — the short version: RAG doesn't eliminate hallucinations but it shrinks the gap between "what the model has to work with" and "what's actually true." Combine with a citation check (Layer 1) for the best result.

**Layer 3 — Constrained / structured outputs**

Restrict the output space. Instead of free-form text, ask for JSON with a fixed schema. If the field you need is `is_eligible: true/false`, the model can't smuggle in a wrong sentence — it either parses or it doesn't.

```python {title="Structured output to constrain the answer space" run=false}
from pydantic import BaseModel
from openai import OpenAI  # swap for your provider; same pattern on any SDK

class RefundDecision(BaseModel):
    eligible: bool
    reason: str          # 1 sentence, grounded in context
    confidence: str      # "high" | "medium" | "low"

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",   # or any model that supports structured output
    messages=[{"role": "user", "content": prompt}],
    response_format=RefundDecision,
)

decision = response.choices[0].message.parsed
# Now you can gate on confidence before showing to the user:
if decision.confidence == "low":
    escalate_to_human(decision)
```

**Layer 4 — Self-consistency sampling**

Ask the model the same question 3–5 times (different temperatures). If the answers disagree, that's a signal of high uncertainty — surface a "I'm not sure, please verify" to the user rather than picking one answer at random.

**Layer 5 — Evals and observability**

Catch regressions before users do. Build a small golden set (50–200 question/answer pairs) and run it on every model or prompt change. Track faithfulness (did the answer match the retrieved doc?) and answer rate (how often did it say "I don't know" when it should have). Tools like RAGAS or a simple LLM-as-judge script work well here.

:::gotcha
Adding "do not hallucinate" or "only give accurate answers" to your system prompt does almost nothing measurable. The model doesn't know when it's wrong — that's the entire problem. What actually works is constraining *where* the model can draw its answer from (RAG + citation enforcement) and constraining *what shape* the output can take (structured outputs). Move from soft instructions to hard architectural constraints.
:::

:::war-story {title="The confident wrong refund window"}
A Pune-based SaaS startup added an LLM chatbot to their B2B product in Q1. The system prompt said "answer questions about our refund policy." No context was provided — just training data and vibes. The model had seen thousands of SaaS refund policies during pretraining, so it confidently answered "30 days" (the industry modal). The actual policy was 14 days. Three enterprise customers claimed refunds under the "30-day policy the chatbot promised." Support escalations, a legal review, and two weeks of remediation. The fix took an afternoon: inject the actual policy PDF as context and add a citation requirement. The lesson wasn't "LLMs are bad" — it was "you cannot skip Layer 1 and Layer 2."
:::

:::interview-line
"Hallucinations come from next-token prediction without ground truth — I treat them as an architecture problem, not a prompt problem: RAG to ground knowledge, structured outputs to constrain the answer space, citation enforcement to make gaps visible, and evals to catch regressions before they hit users."
:::

:::qa {q="Why do LLMs hallucinate even on topics they were heavily trained on?"}
Heavy training on a topic doesn't guarantee factual precision — it guarantees fluent-sounding text about that topic. If training data contained contradictions or errors, those patterns are baked into the weights. The model also conflates similar facts (people, dates, statistics) because they share similar surrounding context in text. More training data helps at the margins but doesn't solve the root cause.
:::

:::qa {q="What's the difference between a hallucination and a model being outdated?"}
Knowledge cutoff is one *cause* of hallucinations (the knowledge gap row in the table above), not a separate phenomenon. The model doesn't know what it doesn't know — so when asked about post-cutoff events, it extrapolates from old patterns and produces confident-sounding fiction. RAG is the right fix for both: give the model the current facts at runtime rather than relying on baked-in training knowledge.
:::

:::qa {q="How do you evaluate hallucination rate in production?"}
Two signals: faithfulness and answer rate. Faithfulness asks "does the response contradict the retrieved context?" — you can score this with an LLM judge at low cost. Answer rate tracks how often the model says "I don't know" vs. invents an answer; if it spikes down after a prompt change, your new prompt is probably over-confident. Run both on a golden eval set before every deployment, and log a sample of live traffic for periodic audit.
:::

:::drill {type="mcq" q="A model confidently gives a wrong patient medication dose despite the correct dose being in the system prompt. What is the most likely root cause?"}
- [ ] The model's training data contained no medical information
- [ ] The model intentionally ignored the system prompt
- [x] Context drift or prompt ambiguity caused the model to draw on training priors instead of the provided context
- [ ] The temperature was set too low, making the model conservative
:::

:::drill {type="mcq" q="Which mitigation has the highest impact-to-effort ratio as a first step when your RAG chatbot keeps hallucinating details not in the retrieved documents?"}
- [ ] Switch to a larger model with a more recent knowledge cutoff
- [ ] Fine-tune the model on your domain documents
- [x] Add a citation requirement to the system prompt and reject responses that don't quote the source
- [ ] Enable self-consistency sampling across 5 parallel calls
:::

:::key-takeaway
Hallucinations are an architectural risk, not a prompt bug. Layer your defences: ground the model with retrieved context, constrain the output shape, enforce citations, and run evals on every change — because the model will never tell you when it's making something up.
:::
