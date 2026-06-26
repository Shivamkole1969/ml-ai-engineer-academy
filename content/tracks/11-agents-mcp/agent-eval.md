---
id: agent-eval
track: 11-agents-mcp
title: "Agent evaluation & observability (traces, tool-success, task success)"
badge: HOT
minutes: 9
prereqs: []
tags: [agents, evaluation, observability, tracing, llm-as-judge, tool-success, task-success]
xp: 60
hot2026: true
---

It's Monday morning. Your research agent shipped on Friday. Slack is lighting up — users say it "just stops working." You pull the logs. The agent ran. Fifty spans. No Python exceptions. No HTTP 500s. But the answers are wrong. Or missing. Or the agent stopped halfway. You stare at the logs trying to reconstruct what happened, but they're flat — a wall of timestamps. You have no idea which step went sideways, which tool call failed silently, or whether the agent even tried the right approach. This is the agent observability problem, and it bites every team that skips it.

## Why Agent Observability Is a Different Beast

With a normal REST endpoint, observability is simple: latency, error rate, p99. One request, one response.

An agent run is a *tree* of decisions. A single user task might unfold into 3 LLM calls, 5 tool calls, 2 retries, and 90 seconds of wall time. "No errors" just means no Python exceptions were raised. The agent could have called the wrong tool, looped three extra times burning tokens, or returned a confident hallucinated answer — and your standard infra metrics would look perfectly fine.

You need three measurement layers, each catching different failure modes.

:::why-prod
In production, a broken agent often looks *healthy* on standard infrastructure metrics — it responds, it doesn't crash. But it might burn 10x the expected tokens on circular retries, or return wrong answers with high confidence. You cannot catch this without agent-native observability: traces, not just logs.
:::

## Layer 1: Traces — The Foundation

A trace is a tree of spans. For an agent run:

- **Root span** — the full agent run (user query → final answer, with total latency and token cost)
- **Child spans** — each LLM reasoning call, each tool call, each planning step

Every span records inputs, outputs, latency, token counts, errors, and a status flag. This is what separates real observability from flat logs. With traces you can replay exactly what the agent did and pinpoint the span where things went sideways.

**Free local option:** Arize Phoenix is open-source, runs entirely on your laptop, and auto-instruments LangChain, LlamaIndex, and OpenAI clients with two lines of code.

```python {title="Trace an agent with Arize Phoenix (free, local)" run=false}
# pip install arize-phoenix openinference-instrumentation-openai opentelemetry-sdk
# Start the UI: python -m phoenix.server.main serve  →  http://localhost:6006

import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor

px.launch_app()            # connects to local Phoenix server
OpenAIInstrumentor().instrument()
# Every OpenAI call your agent makes is now auto-traced.
# In the Phoenix UI: filter span_kind="TOOL" to see tool-level success rates.

# For your own custom tools, emit a span manually:
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

def call_search_tool(query: str) -> list:
    with tracer.start_as_current_span("web_search") as span:
        span.set_attribute("tool.input", query)
        result = my_search_api(query)                        # your actual call
        success = bool(result)                               # non-empty = success
        span.set_attribute("tool.success", success)          # key attribute
        span.set_attribute("tool.result_count", len(result) if result else 0)
        if not success:
            span.set_status(trace.StatusCode.ERROR, "empty result")
        return result or []
# Aggregate: tool_success_rate = count(tool.success=True) / count(spans where kind=TOOL)
```

## Layer 2: Tool-Success Rate

For every tool call in your traces, record whether it produced a usable result. "Success" means: returned without error, result is non-empty, and for data-fetching tools, the data is actually fresh and relevant.

Aggregate this *per tool*, not just globally. If `web_search` shows 40% success and `calculator` shows 99%, your search integration is the problem — not the agent's reasoning or the LLM. This precision is only possible with span-level tracking.

Threshold alert: set a simple monitor that fires when any tool's success rate drops below 70% over a 15-minute window. This single alert would have saved countless teams hours of confused debugging.

## Layer 3: Task-Success Rate

Did the agent actually complete the user's goal, end to end?

This is the hardest to measure. Three practical approaches:

**Deterministic check.** If the task has a clear ground truth ("find the order ID for invoice #4421"), compare programmatically. Fast and free.

**LLM-as-judge.** For open-ended tasks, prompt a fast judge model (Claude Haiku, GPT-4o-mini) with the user's original goal and the agent's final output. Ask: "Did the agent successfully accomplish this goal? Score 1–5 and explain." Scales cheaply to thousands of runs.

**Human spot-check.** For high-stakes domains (finance, legal, medical), sample 5–10% for human review. Use the LLM-judge for the rest and calibrate the judge against human scores periodically.

:::table {title="Three evaluation layers — what each catches and where it falls short"}
| Layer | Metric | What it catches | Blind spot |
|---|---|---|---|
| Tool calls | Tool-success rate | API failures, bad schemas, empty results, rate limits | Agent calling the *right* tool with wrong arguments |
| Agent path | Trajectory quality | Unnecessary loops, wrong tool choice, wasted steps | Correct path that still produces a wrong final answer |
| End to end | Task-success rate | Whether the user's goal was actually met | *Why* it failed (need traces for that) |
:::

:::gotcha
Don't use the same model family as your agent to judge its own outputs. If your agent runs on GPT-4o, judging with GPT-4o inflates scores — the judge shares the same blind spots and hallucination patterns. Use a cross-family judge: if your agent is GPT-4o, judge with Claude. If it's Claude, judge with Gemini. Different weights, different failure modes.
:::

:::war-story {title="The silent retry loop that tripled the bill"}
A fintech team deployed a research agent that fetched live stock data to answer portfolio questions. In production, their data provider started rate-limiting at 60 requests per minute. The agent's tool-success rate on `get_stock_price` quietly dropped to 30%. But their task-success eval only checked the quality of the final answer text — and the agent, to its credit, kept trying. It would loop 8–12 times per query burning tokens on failed retries, then eventually return a stale cached answer that read fine to the LLM judge. Nobody noticed for six days. The monthly API bill came in at 3x budget. A single alert on tool-success rate below 70% for any tool over a 15-minute window would have fired within 20 minutes of the rate-limiting starting.
:::

:::interview-line
"We track three layers: tool-success rate per tool call in traces, trajectory quality via LLM-as-judge on the agent path, and end-to-end task-success rate. That tells us not just if the agent failed, but exactly where and why."
:::

:::qa {q="How is evaluating an agent different from evaluating a regular LLM output?"}
A regular LLM eval asks: given input X, was output Y correct? An agent eval must also check the *process* — did it call the right tools, in what order, with what arguments, and did it stop at the right time? A correct final answer reached via 15 unnecessary tool calls is still a production failure: it's slow, expensive, and fragile. You need traces and trajectory evaluation, not just output quality scoring.
:::

:::qa {q="What is LLM-as-judge and when do you use it for agent tasks?"}
You prompt a capable model with the user's original goal and the agent's final output, then ask it to score whether the goal was met. Use it when tasks are open-ended and you can't write a deterministic checker — for example, "research this company and summarize the top 3 risks." It scales to thousands of runs at low cost. The critical rule: use a cross-family judge (different model vendor than your agent) to avoid shared blind spots inflating your scores.
:::

:::drill {type="mcq" q="Your agent's infra metrics look normal — p99 latency is fine, error rate is 0%. But users are reporting wrong answers. What's the most likely missing piece?"}
- [ ] The base LLM needs to be upgraded to a larger model
- [ ] You need to increase the agent's max_iterations limit
- [x] You have no trace-level observability — you can't see what the agent actually did step by step
- [ ] Your task-success eval threshold is set too high
:::

:::drill {type="mcq" q="Your agent uses GPT-4o. Which judge setup gives the most reliable task-success scores?"}
- [x] A cross-family model like Claude Haiku, which has different failure modes
- [ ] GPT-4o judging its own outputs — same model, so it understands the task best
- [ ] A deterministic regex check on the final answer string
- [ ] Always use human evaluators — LLM judges are never trustworthy
:::

:::key-takeaway
Agent evaluation needs three layers: tool-success rate (did each tool actually work?), trajectory quality (did it take a sensible path?), and task-success rate (did the goal get met?). Traces give you all three in one shot. Without them, you're flying blind — and your first sign of a problem will be the bill.
:::
