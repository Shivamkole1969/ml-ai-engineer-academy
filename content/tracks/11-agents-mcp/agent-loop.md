---
id: agent-loop
track: 11-agents-mcp
title: "The Agent Loop: reason → tool → observe → loop/stop (and where it breaks)"
badge: HOT
minutes: 10
prereqs: []
tags: [agents, llm, tool-calling, production, loops, failures]
xp: 60
hot2026: true
---

You ship a "smart research assistant" for a client in Pune. It works beautifully in the demo — searches the web, summarises results, done. You deploy it over the weekend. Monday morning you get a Slack message: the agent looped 47 times, called the search tool on every iteration, and burned through half the month's API budget before hitting a token limit. No answer. Just a bill.

That's not bad luck. That's you not knowing exactly how the agent loop works — and where it can get stuck. Let's fix that.

## The four-phase loop

Every AI agent — whether it's built with LangGraph, AutoGen, or raw API calls — runs the same core cycle:

1. **Reason**: The LLM reads the system prompt, conversation history, and available tools. It decides what to do next.
2. **Tool call**: It emits a structured tool-use request (e.g., `search_web(query="RAG eval metrics")`).
3. **Observe**: Your code runs the tool and appends the result back to the message history.
4. **Loop or stop**: The LLM sees the result and decides — call another tool, or emit a final text answer?

That's it. That's the whole loop. Everything else — memory, planning, multi-agent handoffs — is built on top of this.

The loop exits when the model signals `stop_reason == "end_turn"` (Anthropic SDK) or an equivalent "no more tools needed" signal in other frameworks. If that signal never comes, the loop keeps going — until you or your `max_steps` guard kills it.

:::why-prod
Production agents almost never fail in demos; they fail at scale and at the edges. A missing stop condition, a tool that always returns an error, or an overly cautious model that "keeps checking" will silently chew through your token budget. You need to understand the loop so you can instrument and cap it.
:::

```python {title="Minimal agent loop — Anthropic SDK" run=false}
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def run_agent(user_message: str, tools: list, max_steps: int = 10) -> str:
    """
    Bare-metal agent loop.
    Install: pip install anthropic
    Run locally: python agent_loop.py
    """
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        # --- STOP phase ---
        if response.stop_reason == "end_turn":
            # Model is done; grab final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "(no text in final response)"

        # --- TOOL + OBSERVE phase ---
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = call_tool(block.name, block.input)   # your dispatch fn
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        # Append model turn + tool results back into history (the "observe" step)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Safety net — never let the loop run forever
    return "ERROR: max_steps reached without a final answer"


def call_tool(name: str, inputs: dict) -> str:
    """Wire your real tools here."""
    if name == "search_web":
        return f"[stub] results for: {inputs.get('query')}"
    return f"[unknown tool: {name}]"
```

The key insight: the message history grows with every loop iteration. The model reads the entire history each step. That is why runaway loops are expensive — you pay tokens on every pass.

## Where the loop breaks

The loop is clean on paper. In production it breaks in predictable, annoying ways.

:::table {title="Common loop failure modes"}
| Failure | What happens | Fix |
|---|---|---|
| No stop condition | Loop hits token limit or `max_steps`, returns nothing useful | Always set `max_steps`; log when it triggers |
| Tool always errors | Model retries the same tool over and over | Return a clean error string from tools; add retry logic with backoff |
| Context overflow | History grows past the context window | Summarise or truncate old observations |
| Hallucinated tool args | Model invents argument values | Use strict JSON schemas in your tool definitions |
| Goal drift | Model starts solving a sub-problem and forgets the original task | Pass the original goal in system prompt; re-state it on each call |
:::

:::gotcha
The most common mistake: forgetting to append both the assistant's tool-call message AND the tool result back into `messages`. If you only add one, the model loses track of what it asked for — and either crashes or hallucinates the answer. The append order must be: assistant turn → user turn with tool results.
:::

:::war-story {title="The infinite Google loop that cost ₹40,000 in one night"}
A team built a competitive-intel agent that called a search API for pricing data. Their tool always returned an HTTP 429 (rate-limited) but they serialised it as an empty string `""` instead of a clear error message. The model saw an empty result and figured it needed to search again — with a slightly different query. It looped 200 times over six hours before the on-call engineer noticed the Datadog spike. The fix was three lines: return `"ERROR: rate limited, retry in 60s"` from the tool and add a `max_steps=15` guard. Always let the model know why a tool failed.
:::

:::interview-line
"An agent loop is just reason-tool-observe repeated under a max-steps guard — the hard part is making every tool return a signal the model can act on, especially on failure."
:::

:::qa {q="Walk me through what happens inside an agent on a single step."}
The model receives the full message history (system prompt + conversation + previous tool results) and generates a response. If it needs more information, it emits a structured tool-call. Your code executes the tool, appends the result to history, and calls the model again. This repeats until the model signals it's done or you hit a step limit.
:::

:::qa {q="Why is context growth a problem in long-running agents?"}
Every loop iteration appends the tool call and its result to the message history. The model re-reads the entire history on each pass — so token cost grows roughly linearly with the number of steps. For a 20-step agent with 500-token tool results, you can easily hit 10k–20k input tokens per call by step 15. Solutions include summarising old observations, using a sliding window, or offloading to an external memory store.
:::

:::qa {q="How do you prevent an agent from looping forever in production?"}
Set a hard `max_steps` limit and log every time it triggers — a frequent trigger is a signal your tools are returning bad data or your prompt is ambiguous. Also define a clear success criterion in the system prompt so the model knows when to stop, and make sure every tool returns a structured, model-readable signal on both success and failure.
:::

:::drill {type="mcq" q="An agent loop calls a tool 12 times and then errors with 'context length exceeded'. What is the MOST LIKELY root cause?"}
- [ ] The model's temperature was set too high
- [ ] The system prompt was too short
- [x] Tool results accumulated in history without any truncation or summarisation
- [ ] The tool was returning results in the wrong format
:::

:::drill {type="mcq" q="After executing a tool call, which sequence correctly updates the message history?"}
- [ ] Append only the tool result as a user message
- [ ] Append only the assistant's tool-call message
- [ ] Append the tool result first, then the assistant's tool-call message
- [x] Append the assistant's tool-call message first, then a user message containing the tool result
:::

:::drill {type="mcq" q="Your agent stops returning answers after 10 steps but the task is clearly unfinished. The logs show stop_reason='max_tokens' on every final call. What should you investigate first?"}
- [ ] Increase max_steps to 50
- [x] Check whether tool results are bloated — large raw responses growing context each step
- [ ] Switch to a smaller model
- [ ] Disable tool calling and use pure reasoning
:::

:::key-takeaway
The agent loop is just reason → tool → observe repeated — but every failure mode in production traces back to either runaway iteration (missing step cap) or bad tool signals (errors the model can't understand). Get those two right first.
:::
