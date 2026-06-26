---
id: llm-monitoring
track: 13-monitoring-cost
title: "LLM-specific monitoring: token spend, hallucination rate, silent decay"
badge: HOT
minutes: 10
prereqs: []
tags: [llm, monitoring, hallucination, token-cost, silent-decay, production, rag]
xp: 60
hot2026: true
---

Your LLM-powered support bot has been running smoothly for two months. Tickets are resolved, users seem happy. Then one Friday, your PM messages you: users are complaining the bot is citing the wrong refund policy. You check your dashboards — zero errors, latency normal, model API healthy. No alerts fired. But the bot has been quietly hallucinating for days, and nobody noticed.

This is the defining challenge of LLM monitoring. The scariest failures don't trip alarms. They just silently erode trust until a user (or a compliance audit) finds them first.

## Why LLM monitoring is a different beast

Traditional ML models output a number. A wrong number leaves a detectable fingerprint: accuracy drops, distributions shift, you get an alert. LLMs output *text*. Text can be confident, grammatically perfect, and completely wrong — and your existing error-rate dashboard will shrug and say "all good."

Three metrics matter here that simply don't exist in classical ML:

1. **Token spend** — your cost, and a surprisingly good proxy for behavior change upstream
2. **Hallucination rate** — how often the model produces unfaithful or factually invented output
3. **Silent decay** — gradual quality degradation where no single metric goes red

:::why-prod
A hallucination in a customer-facing product can mean wrong medical guidance, incorrect financial advice, or a broken legal disclosure — with zero error logs and zero on-call pages. Token spend spikes are often the *first* signal that something upstream changed (longer context, a retrieval bug, a new feature gone wrong). These three metrics are the gap between "the model is up" and "the model is actually working."
:::

## Token spend: the canary in the cost mine

Every LLM API call has two costs: **input tokens** (your prompt + context) and **output tokens** (the model's response). Output tokens cost more per token on most providers and are harder to predict. A single bug — a RAG retriever stuffing duplicate chunks, a prompt template gone recursive — can turn a ₹5 request into a ₹500 one.

Track these four things per request and push them to your metrics store:

```python {title="Per-request token cost logger" run=false}
# pip install tiktoken
# Works locally; swap the print() for your metrics client in prod
import tiktoken

# Update these for your model — prices are illustrative
PRICE_PER_1K = {
    "input":  0.003,   # USD per 1K input tokens
    "output": 0.015,   # USD per 1K output tokens
}

def log_llm_call(prompt: str, response_text: str, model: str = "gpt-3.5-turbo") -> dict:
    enc = tiktoken.encoding_for_model(model)
    n_input  = len(enc.encode(prompt))
    n_output = len(enc.encode(response_text))

    cost_usd = (n_input / 1000)  * PRICE_PER_1K["input"] + \
               (n_output / 1000) * PRICE_PER_1K["output"]

    record = {
        "input_tokens":  n_input,
        "output_tokens": n_output,
        "cost_usd":      round(cost_usd, 6),
    }
    print(f"[LLM] {record}")  # In prod: push to Prometheus, Datadog, CloudWatch, etc.
    return record

# Example:
# log_llm_call(system_prompt + user_message, model_reply)
```

A sudden jump in median `input_tokens` — say, from 800 to 4,000 — usually means your context pipeline started retrieving more chunks than expected. Catch it early, before the billing alarm fires at month's end.

:::gotcha
Don't alert only on monthly total spend. Set a **p99 alert on per-request token count** in your metrics store. A single runaway request with 100,000 tokens (bad retrieval loop, prompt injection, a malformed doc in your vector DB) will be invisible in a monthly aggregate until the invoice arrives.
:::

## Hallucination rate: catching confident lies

Hallucination is the model generating output that contradicts source documents, invents facts, or fabricates citations — all with complete confidence. There is no exception thrown. Nothing in your logs goes red.

Four practical detection approaches, ranked by effort:

:::table {title="Hallucination detection methods"}
| Method | How it works | Effort | Best for |
|---|---|---|---|
| User signals | Thumbs-down, corrections, re-asks | Free | Any user-facing product |
| Regex / factual checks | Rule-based extraction of critical fields | Low | Structured output (dates, numbers, codes) |
| LLM-as-judge | A second LLM scores faithfulness to source | Medium | RAG Q&A, document assistants |
| Reference eval on gold set | Score against labeled gold answers | Medium | When you have annotated test data |
:::

For RAG systems specifically, the key metric is **faithfulness score**: does every claim in the answer appear in the retrieved context, or did the model invent it? Libraries like RAGAS and DeepEval automate this. Run it on a random 5% sample of live traffic — not just your test set — so you catch real-world edge cases your golden examples don't cover.

## Silent decay: the failure mode with no alert

Silent decay is gradual quality degradation that trips no hard metric. Latency is fine. Error rate is fine. Token spend is flat. Users are getting subtly worse answers every week, and nobody notices until the product review or a churn spike.

It happens because:

- Small edits to **prompt templates** accumulated over sprints and changed model behavior
- The **retrieval layer** started returning worse chunks (embedding model drift, index corruption)
- The **provider quietly updated the underlying model** — yes, `gpt-4` and `claude-3-sonnet` are not frozen snapshots

The fix is **behavioral monitoring**: run a fixed set of golden prompts through your full pipeline on a weekly cron and score the outputs. If quality scores drop 10% week-over-week, something changed. This is LLM regression testing in production, and it's cheap to set up.

:::war-story {title="The 'be concise' edit that triggered a compliance incident"}
A Pune-based fintech team was running a KYC document assistant. One engineer added "be concise" to the system prompt to reduce output tokens — a reasonable cost optimisation. Outputs got shorter; token spend dropped. No alarms. Three weeks later, the compliance team found that mandatory disclosure language was being silently dropped from every summary. Error rate: zero. Latency: normal. Token count: actually better. The only catch was a weekly golden-set eval that had been quietly tracking faithfulness score, which had fallen from 0.91 to 0.71 over those three weeks. Without that eval, the issue would have hit an RBI audit.
:::

:::interview-line
"For LLMs, I track three things standard ML doesn't need: per-request token count to catch cost anomalies and upstream changes, a faithfulness score on sampled RAG outputs to catch hallucinations, and a weekly golden-set behavioral eval to catch silent decay — because none of these show up in error-rate or latency dashboards."
:::

:::qa {q="How do you detect hallucinations in a production RAG system?"}
I run faithfulness scoring on a random sample of live queries — checking whether each answer's claims are grounded in the retrieved chunks, not invented by the model. Tools like RAGAS or a lightweight LLM-as-judge pipeline handle this automatically. I also monitor user correction signals (thumbs-down, immediate re-asks) as a free leading indicator. And I maintain a small gold eval set that runs weekly to catch regressions before users do.
:::

:::qa {q="What is silent decay and how is it different from model drift?"}
Classic model drift is a distributional shift in your *inputs* that causes output degradation — detectable with standard drift monitoring. Silent decay is subtler: the input distribution can be perfectly stable, but output quality degrades because of prompt template edits, retrieval quality degradation, or unannounced provider model updates. You can't catch it with feature statistics alone. You need behavioral monitoring — scoring a fixed golden-prompt set over time and alerting on quality regression.
:::

:::drill {type="mcq" q="Your RAG chatbot's error rate and latency are normal, but user satisfaction scores have been declining for two weeks. No new features shipped. What should you check first?"}
- [ ] Whether the LLM API provider is throttling requests
- [ ] Whether input token count has crossed the context window limit
- [x] Whether faithfulness scores on sampled outputs have dropped, signalling hallucination or retrieval decay
- [ ] Whether the vector database index is corrupted
:::

:::drill {type="mcq" q="Which alert would catch a runaway per-request token bug BEFORE the monthly cloud bill arrives?"}
- [ ] A monthly spend threshold alert in the cloud console
- [ ] A weekly golden-set faithfulness score alert
- [x] A p99 alert on per-request input_tokens in your metrics store
- [ ] A latency p95 alert
:::

:::key-takeaway
LLMs fail silently. Track per-request token spend to catch cost spikes and upstream changes, run faithfulness scoring on a live sample of RAG outputs to catch hallucinations, and run a weekly golden-set behavioral eval to catch silent decay — your standard error-rate and latency dashboards will not see any of these.
:::
