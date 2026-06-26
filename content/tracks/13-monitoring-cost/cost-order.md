---
id: cost-order
track: 13-monitoring-cost
title: "A sane cost-reduction order of operations"
badge: HOT
minutes: 8
prereqs: []
tags: [cost, optimization, llm, inference, production, routing, caching]
xp: 60
hot2026: true
---

It's 10am on a Tuesday. The CTO forwards you a Slack message from Finance: "Our inference bill jumped from ₹8L to ₹40L this month. What happened?" Everyone in the thread is panicking. Someone suggests distilling the model. Another person wants to rewrite the whole pipeline in C++.

You take a breath. You've seen this before. Most cost explosions have boring, fast fixes — if you attack them in the right order. Skip ahead and you'll spend three weeks distilling a model only to discover the root cause was a 9,000-token system prompt that nobody had trimmed in six months.

The order of operations is the skill.

## Profile before you cut

Rule zero: never optimize what you haven't measured. Before touching anything, run a cost audit — break down your spend by call type, component, and user segment.

You're looking for the 80/20 break: which 20% of call patterns drive 80% of your bill?

```python {title="Quick token-cost audit" run=false}
# Run against your prod request logs (no GPU needed — pure analysis)
# Adapt the log_path and cost-per-token to your provider.

import json, collections

LOG_PATH = "prod_requests.jsonl"
COST_PER_1K_INPUT  = 0.005   # USD — swap for your model's rate
COST_PER_1K_OUTPUT = 0.015

endpoint_totals = collections.defaultdict(lambda: {"in": 0, "out": 0, "calls": 0})

with open(LOG_PATH) as f:
    for line in f:
        r = json.loads(line)
        ep = r.get("endpoint", "unknown")
        endpoint_totals[ep]["in"]    += r["prompt_tokens"]
        endpoint_totals[ep]["out"]   += r["completion_tokens"]
        endpoint_totals[ep]["calls"] += 1

print(f"{'Endpoint':<30} {'Calls':>8} {'$Input':>10} {'$Output':>10} {'$Total':>10}")
print("-" * 70)
for ep, t in sorted(endpoint_totals.items(), key=lambda x: -x[1]["in"]):
    cost_in  = t["in"]  / 1000 * COST_PER_1K_INPUT
    cost_out = t["out"] / 1000 * COST_PER_1K_OUTPUT
    print(f"{ep:<30} {t['calls']:>8} {cost_in:>10.2f} {cost_out:>10.2f} {cost_in+cost_out:>10.2f}")
```

Run this, rank endpoints by total cost, then work top-down. That ranking IS your work queue.

:::why-prod
Every minute you spend optimizing the wrong thing is money you're not saving. A 5-minute audit script has saved teams weeks of wrong-direction engineering. Profile first, always.
:::

## The seven-step order

Here is the sequence that gives you the most savings for the least effort. Always exhaust a step before moving to the next.

:::table {title="Cost-reduction order of operations"}
| Step | What you do | Effort | Typical saving | Time to ship |
|---|---|---|---|---|
| 1. Audit | Break down spend by endpoint, component, user | 1 day | Directional clarity | Immediate |
| 2. Trim tokens | Shorten system prompts, strip unused context, cut verbose output instructions | 1–2 days | 30–70% on prompt cost | < 1 week |
| 3. Cache aggressively | Semantic cache for repeated queries; prefix cache for shared system prompts | 2–5 days | 40–80% on cacheable traffic | < 2 weeks |
| 4. Route by difficulty | Cheap model for easy queries, expensive model for hard ones | 3–7 days | 50–80% on routable traffic | 2–3 weeks |
| 5. Batch + async | Group low-urgency requests; avoid one-call-per-user-event patterns | 3–5 days | 20–40% on batch-eligible traffic | 2–3 weeks |
| 6. Quantize / smaller model | 4-bit quant or swap to a smaller open-weight model for your use case | 1–3 weeks | 50–90% on serving cost | 3–6 weeks |
| 7. Distill | Train a student model on your prod traffic | 4–12 weeks | 70–95%, but quality risk | 2–4 months |
:::

Steps 2–5 are often enough. Most teams that reach step 7 skipped steps 2–4.

## Why this order is right

Each step is cheap to undo. If trimming prompts hurts quality, you can roll back in an hour. If distillation hurts quality, you've lost months.

Each step also unlocks the next. A cached system prompt (step 3) makes prefix caching in step 3 cheaper. Routing (step 4) means your expensive model handles fewer calls, so quantizing it (step 6) has a smaller blast radius.

And critically: steps 2–4 are ops/prompt changes, not model changes. They ship in days and don't require ML infra.

:::gotcha
The most dangerous move is jumping to distillation to "fix the bill." Distillation trains a student model on your current prod outputs — including any bugs, hallucinations, and edge-case failures. If your prompts are bloated and your outputs are noisy, you're baking that noise into the student. Fix the prompts first, then distill on the clean traffic.
:::

:::war-story {title="The system prompt that ate ₹30L"}
A fintech team in Pune had a `/summarise` endpoint costing ₹30L/month. The on-call engineer opened the system prompt for the first time in eight months: 8,400 tokens of accumulated instructions, half of which contradicted each other. Six sprints of "just add a line to the prompt" had compounded silently. An afternoon of trimming got it to 380 tokens. No model change. No infra change. Bill dropped by 82% the next day. The team had been planning a three-month distillation project.
:::

:::interview-line
"My first move on a cost spike is always an audit — find the 20% of calls driving 80% of the bill, then work the cheapest fix first. Usually it's prompt bloat or a missing cache, not the model."
:::

## When to skip steps

Two legitimate reasons to skip ahead:

**You have latency SLAs that rule out caching.** If every query must be fresh (live financial data, real-time personalisation), caching is off the table. Go straight to routing.

**You're already on the smallest sensible model.** If you're running Mistral-7B and quality is at the floor, there's nowhere smaller to go. Distillation becomes the only lever.

Otherwise: don't skip.

:::qa {q="How do you decide which model to route an 'easy' query to?"}
Start simple: use output length as a proxy. Queries answered in fewer than 100 tokens rarely need a frontier model. You can also train a tiny classifier on your existing logs — label calls by the model that actually answered correctly, then use that classifier as the router. Keep a shadow evaluation set to catch quality regressions before they hit users.
:::

:::qa {q="Our CTO wants to distill now because a competitor did it. How do you push back?"}
Frame it in money and risk: distillation costs 4–12 weeks of ML engineering plus GPU training time. If we haven't measured token spend by endpoint, we might distill our way into a worse model that still costs too much. Ask for two weeks to run the audit and exhaust steps 2–4 first. If cost is still above target after that, distillation becomes the obvious next move and everyone's aligned.
:::

:::drill {type="mcq" q="Your audit shows the `/chat` endpoint costs 3x more than all other endpoints combined. Prompt tokens average 6,000 — mostly a static system prompt. What is the highest-leverage first move?"}
- [ ] Distill a student model on `/chat` traffic
- [ ] Add a GPU node to reduce latency
- [x] Trim the system prompt and add prefix caching for the shared preamble
- [ ] Route `/chat` queries to a smaller model immediately
:::

:::drill {type="mcq" q="You've trimmed prompts (step 2) and added caching (step 3). Cost is still 40% above target. What is the correct next step in the order of operations?"}
- [ ] Start a distillation run immediately
- [ ] Quantize the production model to 4-bit
- [x] Implement a difficulty-based router to send easy queries to a cheaper model
- [ ] Rewrite the serving layer in a lower-level language
:::

:::key-takeaway
Profile first, trim prompts second, cache third — most cost problems die before you reach the model. The order of operations isn't optional; it's the difference between a two-day fix and a three-month project.
:::
