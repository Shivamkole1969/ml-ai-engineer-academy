---
id: tools-memory-planning
track: 11-agents-mcp
title: "Tool calling, memory, planning, multi-agent — capabilities & failures"
badge: HOT
minutes: 11
prereqs: []
tags: [agents, tool-calling, memory, planning, multi-agent, llm, production]
xp: 60
hot2026: true
---

It's 4 PM on a Friday. Your product manager Slack-pings you: "The new support agent handled 500 tickets today — awesome! But it's been calling the `refund_order` API for questions that just needed a status update. We've refunded ₹3 lakh we didn't have to."

You dig into the trace. The agent understood the *intent* correctly. It just picked the *wrong tool*. The planning was fine; the tool selection was broken. Welcome to the four ways agents go sideways in production: bad tools, leaky memory, derailed plans, and multi-agent miscommunication. This lesson is your map.

---

## Tool calling — how agents reach into the world

An LLM on its own is a text transformer. It can't check stock, update a DB, or call your internal API. Tool calling (also called function calling) gives it hands.

The pattern is simple: you give the model a list of tool **schemas** — name, description, typed parameters. The model reads the conversation and emits a structured JSON object when it decides to call one. Your code executes it, sends the result back, and the model continues.

The schema description is everything. If you write `refund_order: processes an order`, the model will call it for *any* order-related question. Write `refund_order: issues a monetary refund for a completed order — ONLY use when customer explicitly requests money back` and errors drop dramatically.

:::why-prod
Tool call errors compound. A wrong tool midway through a 10-step plan poisons every downstream step. At 0.95 accuracy per step over 10 steps you're at ~60% end-to-end success — and bad tool selection is one of the fastest ways to tank that number in real workflows.
:::

```python {title="Minimal tool schema for OpenAI-style function calling" run=false}
# Works with any OpenAI-compatible endpoint (Together, Groq, local Ollama with /v1/chat)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Returns the current status and estimated delivery date of an order. "
                "Use this when the customer asks WHERE their order is or WHEN it arrives. "
                "Do NOT use for cancellations or refunds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID shown in the customer's confirmation email."
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": (
                "Issues a full or partial monetary refund for a completed order. "
                "ONLY call when the customer explicitly requests a refund or return."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount_inr": {"type": "number", "description": "Refund amount in INR."}
                },
                "required": ["order_id", "amount_inr"]
            }
        }
    }
]

# The key insight: the model reads "description" fields as instructions.
# Treat them like a mini-prompt. Be explicit about when NOT to use a tool.
```

:::gotcha
Never expose a destructive tool without a description that names the exact trigger condition. "Use this only when…" wording dramatically reduces mis-fires. Add a human-in-the-loop confirmation step for any tool with financial, data-deletion, or external-send side effects.
:::

---

## Memory — short-term, long-term, and what leaks

Agents have four distinct memory modes. Mixing them up is a classic design mistake.

:::table {title="Four agent memory types"}
| Type | Where it lives | Capacity | Typical use |
|---|---|---|---|
| In-context | The active prompt / KV cache | ~128k–1M tokens, then gone | Current task, recent tool results |
| External / retrieval | Vector DB, key-value store | Unlimited (with search) | User history, docs, past decisions |
| In-weights | Model parameters | Fixed at training | General world knowledge |
| In-cache | KV cache snapshots | Session-level | Shared system prompt for cost savings |
:::

The failure mode is context overflow. An agent running a long agentic loop silently drops early messages once the context window fills. It "forgets" what the user originally asked. Fix: summarize-and-compress old turns, or write key facts to an external store and retrieve them explicitly.

Another failure: **stale external memory**. If you embedded product docs six months ago and the return policy changed, the agent confidently cites the old policy. Timestamp your chunks and invalidate aggressively.

:::why-prod
Production agents almost always need external memory for anything longer than one session. In-context alone will not cut it — and when context overflows silently, you get hallucinations that look like correct behavior until a customer complains.
:::

---

## Planning — decomposing goals into steps

Planning is how an agent bridges a vague goal ("book me a trip to Goa") into concrete tool calls. There are two broad styles:

**ReAct (Reason + Act)**: The model alternates between writing a reasoning trace (`Thought: I need to check available flights first`) and taking an action (`Action: search_flights(...)`). It's transparent and easy to debug because the thinking is visible.

**Plan-then-execute**: A planner model creates a full task list upfront; executor sub-agents run in parallel. Faster, but the plan can be wrong before any execution starts and there is no mid-course correction.

Planning failures tend to be:
- **Infinite loops**: The agent keeps re-checking the same resource because it doesn't recognize that the state has changed.
- **Wrong decomposition**: Breaking "send email to all users who signed up this month" into one call per user instead of one bulk API call.
- **Goal drift**: After 15 steps the agent is solving a sub-problem and has abandoned the original objective.

:::gotcha
Always set a hard maximum on agent steps or tool calls — something like `max_steps=25`. Without it, a looping agent can rack up thousands of API calls before anyone notices. Log every step so you can replay and diagnose.
:::

---

## Multi-agent — power and new failure modes

Single agents plateau on complex tasks. Multi-agent systems add parallelism and specialization: an orchestrator decomposes work, sub-agents handle domains (search agent, code agent, email agent), results flow back and merge.

The new failure modes are communication failures:
- **Semantic mismatch**: Agent A passes `{"date": "26/06/2026"}` but Agent B expects ISO `"2026-06-26"` and silently mis-parses it.
- **Trust without verification**: The orchestrator blindly uses a sub-agent's output as fact. One hallucinating sub-agent corrupts the whole result.
- **Race conditions**: Two parallel agents both write to the same shared state (a DB row, a file) and the last writer wins incorrectly.

The fix is to treat inter-agent messages like API contracts: schema-validate them, add checksums or confidence scores, and make shared writes idempotent.

:::war-story {title="The email-blast incident"}
A multi-agent marketing tool at a B2B SaaS had a "draft agent" and a "send agent". The orchestrator passed draft IDs as integers; the send agent cast them to strings for its API call. A mismatch in ID format caused the send agent to resolve ambiguous IDs incorrectly and dispatched a promotional email to 4,000 opted-out users, triggering GDPR complaints. The root cause: no schema contract between agents, no opt-out check in the send agent itself. Defense-in-depth — each agent should validate its own preconditions, not trust upstream.
:::

---

:::interview-line
"The biggest agent failure modes I watch for are tool mis-selection from weak descriptions, silent context overflow wiping memory, and inter-agent type mismatches — I put schema validation and step caps on every production agent."
:::

:::qa {q="How do you prevent an agent from calling a destructive tool when it shouldn't?"}
Write the tool description to explicitly state the exact trigger condition ("ONLY when the customer explicitly requests a refund") and add a guardrail layer — either a cheap classifier that re-checks the intent before execution, or a human-approval step for high-risk actions. Logging every tool call with its reasoning trace lets you audit and retrain.
:::

:::qa {q="What happens when an agent's context window fills up mid-task?"}
By default, the model either errors out or silently truncates older messages — meaning it loses the original goal or early tool results. The production fix is a rolling summary: compress old turns into a short state snapshot and append it as a system message. For long-running agents, write key state to external memory and retrieve it on demand rather than relying on the context window alone.
:::

:::qa {q="Why would you use multiple agents instead of one big agent with many tools?"}
Parallelism and specialization. A single agent processes tool calls sequentially; multiple agents can work simultaneously on independent sub-tasks, cutting wall-clock time. Specialization also helps — a dedicated retrieval agent tuned on your knowledge base will outperform a generalist. The trade-off is inter-agent communication complexity, which introduces its own failure modes like schema mismatches and trust issues.
:::

:::drill {type="mcq" q="Your agent keeps calling `search_docs` in a loop even after finding the answer. What is the most likely root cause?"}
- [ ] The model's temperature is set too high
- [ ] The external vector DB has stale embeddings
- [x] The agent has no termination condition and doesn't recognise that the retrieved result already satisfies the goal
- [ ] The context window has overflowed
:::

:::drill {type="mcq" q="Which memory type would you use to persist a user's preferences across multiple sessions?"}
- [ ] In-context memory — keep it in the system prompt
- [ ] In-weights memory — fine-tune the model on user data
- [x] External memory — store in a key-value or vector store and retrieve at session start
- [ ] In-cache memory — reuse the KV cache snapshot
:::

:::drill {type="mcq" q="Two parallel sub-agents both attempt to update the same order record. What property prevents data corruption?"}
- [ ] Schema validation on the inter-agent message
- [ ] A higher max_steps limit
- [x] Idempotent writes with optimistic locking or atomic compare-and-swap
- [ ] Giving the orchestrator a higher-capacity context window
:::

:::key-takeaway
Tool descriptions are agent instructions — write them like prompts. Pair them with step caps, schema contracts between agents, and a memory strategy that survives context overflow, and you've addressed the four biggest production failure modes before they cost you.
:::
