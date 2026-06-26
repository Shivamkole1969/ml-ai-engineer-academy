---
id: agents-in-prod
track: 11-agents-mcp
title: "Designing agents that survive production"
badge: HOT
minutes: 11
prereqs: []
tags: [agents, production, reliability, circuit-breaker, idempotency, human-in-the-loop]
xp: 60
hot2026: true
---

It is 11:47 PM on a Tuesday. Your agent is supposed to send a weekly digest to 200 newsletter subscribers. Instead, it loops. By the time the PagerDuty alert wakes you up, it has called the `send_email` tool 4,800 times. Twenty-four copies of the same email, per subscriber. Twitter is not kind. Your CEO is in the Slack thread.

This is not a hypothetical. Agents in production fail in ways that feel embarrassing in retrospect but were completely predictable. The goal of this lesson is to make you the engineer who sees those failure modes coming — and designs the guardrails before someone gets 24 emails.

## What makes agents different from normal APIs

A REST endpoint is stateless and bounded. One request, one response, done.

An agent is neither. It runs a loop of unpredictable length, calls external tools with real side effects, and makes decisions in the middle. Each of those properties is a new blast radius.

When a normal API call fails, you retry it. When an agent loops forever, it keeps spending money and causing damage. When a tool call has a side effect (email sent, order placed, row deleted), retrying it blindly makes things worse.

Production agent design is fundamentally about **containing blast radius** — limiting how far things can go wrong before a human or a circuit breaker intervenes.

:::why-prod
Agents introduce unbounded execution time, compounding tool side effects, and decision points that models can get subtly wrong. Without explicit guardrails, a single bug or a single bad model output can cause cascading, expensive, and sometimes irreversible damage at 3 AM when no one is watching.
:::

## The five levers of a production-safe agent

Think of these as layers you stack on top of any agent framework, regardless of whether you are using LangGraph, CrewAI, or raw tool-calling loops.

**1. Hard budget limits.** Cap total LLM tokens, tool calls, and wall-clock seconds per task. Not soft warnings — hard stops that raise an exception and land the agent in a "failed" terminal state.

**2. Idempotency keys on every destructive tool.** Before any tool that writes, sends, or deletes, generate a UUID for that specific task + action. Check if it was already executed. If yes, skip. This is the only reliable defense against the duplicate-email nightmare above.

**3. Human-in-the-loop checkpoints.** Before any irreversible action with broad impact (deleting records, sending comms, posting publicly), pause and emit a `requires_approval` event. A human or a rules engine approves or rejects. The agent resumes or aborts.

**4. Scoped permissions per agent role.** An agent that reads customer data to draft a support reply has no business calling your `delete_account` tool. Use capability lists and fail loudly if an agent tries to call a tool outside its declared scope.

**5. Idempotent, logged, replayable state.** Checkpoint agent state after each step so you can replay or resume from a failure without re-running side effects. A simple serialized JSON blob in Redis or a DB row is enough.

:::table {title="Failure mode → guardrail mapping"}
| Failure mode | What goes wrong | Guardrail |
|---|---|---|
| Infinite loop | Agent keeps calling tools, burns budget | Step limit + wall-clock timeout |
| Duplicate side effects | Email/order fires multiple times | Idempotency key per action |
| Runaway cost | Long context + many steps = ₹500 surprise bill | Token + cost hard cap |
| Privilege escalation | Agent calls a tool it shouldn't | Scoped tool allowlist per agent |
| Irreversible action | Row deleted, can't undo | Human-in-the-loop checkpoint |
| Silent failure | Tool returns error, agent halts silently | Structured error state + alert |
:::

Here is a minimal Python pattern that wires budget limits and idempotency together. Drop this wrapper around any tool call in your agent loop.

```python {title="Budget guard + idempotency wrapper" run=false}
import uuid, time, redis

r = redis.Redis()  # or any fast store

def safe_tool_call(agent_run_id: str, tool_name: str, tool_fn, *args, **kwargs):
    """
    Wraps a tool call with:
      - Idempotency: will not re-execute if already succeeded
      - Step budget: raises if we've exceeded allowed calls
    Free local dev: pip install redis && docker run -p 6379:6379 redis
    """
    idempotency_key = f"tool:{agent_run_id}:{tool_name}:{hash(str(args)+str(kwargs))}"

    # Already ran successfully — return cached result
    cached = r.get(idempotency_key)
    if cached:
        return cached.decode()

    # Check step budget
    step_key = f"steps:{agent_run_id}"
    steps_used = r.incr(step_key)
    r.expire(step_key, 3600)  # auto-cleanup after 1 hour
    MAX_STEPS = 20
    if steps_used > MAX_STEPS:
        raise RuntimeError(f"Agent {agent_run_id} exceeded {MAX_STEPS}-step budget. Halting.")

    # Execute tool
    result = tool_fn(*args, **kwargs)

    # Cache success — TTL of 24h prevents stale replays
    r.setex(idempotency_key, 86400, str(result))
    return result
```

:::gotcha
Do not use the agent's LLM-generated reasoning as the idempotency key. Models rephrase. Use a deterministic hash of the actual tool name + input arguments, scoped to the run ID. If you hash the model's "explanation" of what it's doing, you'll get duplicates every time it rewords itself.
:::

:::war-story {title="The order-placement loop that cost $11,000 in 40 minutes"}
A fintech startup built a trading agent to rebalance a portfolio. The agent called a `place_order` tool, but the brokerage API was slow and returned a 504. The agent interpreted silence as "order not placed" and retried. Eleven thousand dollars in duplicate buy orders later, the circuit breaker finally triggered — not the one they built (there wasn't one), but the brokerage's own fraud detection. The fix took four lines of code: an idempotency key and a 504 → "assume success, log and halt" policy. The post-mortem took four hours.
:::

:::interview-line
"Every destructive tool in my agent gets an idempotency key and a scoped permission check before it fires — those two lines are cheaper than any post-mortem."
:::

:::qa {q="How do you prevent an agent from running forever and burning your LLM budget?"}
Set hard numeric limits at the start of every agent run: a maximum number of steps, a token budget, and a wall-clock timeout. When any limit is hit, raise an exception that transitions the agent to a failed terminal state and fires an alert. Soft warnings are not enough — agents do not read warnings.
:::

:::qa {q="A teammate says 'just retry on failure, agents are resilient.' What's wrong with that advice?"}
Retrying is safe only for idempotent operations. If a tool has a side effect — sends a message, places an order, modifies a record — blind retries multiply the damage. You need an idempotency key to detect whether the action already succeeded before re-executing it. Resilience without idempotency is just a more reliable way to duplicate side effects.
:::

:::drill {type="mcq" q="An agent's email-sending tool gets a 503 timeout on step 7 of 10. What should the agent do next?"}
- [ ] Immediately retry the tool call with the same arguments
- [x] Check an idempotency record; if absent, retry once with backoff; if present, skip and continue
- [ ] Restart the entire agent run from step 1
- [ ] Log a warning and proceed to step 8 without sending
:::

:::drill {type="mcq" q="Which of these is the strongest reason to use human-in-the-loop checkpoints in a production agent?"}
- [ ] It makes the agent faster by parallelising decisions
- [ ] It lets you avoid writing tool descriptions
- [x] Irreversible actions need a human gate because model errors compound and cannot always be undone programmatically
- [ ] It reduces token usage by short-circuiting the reasoning loop
:::

:::key-takeaway
Production agents need four non-negotiables before going live: a hard step/cost budget, an idempotency key on every side-effecting tool, scoped tool permissions per agent role, and a human checkpoint before any irreversible action. Everything else is tuning.
:::
