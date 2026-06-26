---
id: llm-app-fundamentals
track: 09-genai-rag
title: "LLM app fundamentals: APIs, prompting, structured outputs, tool calling"
badge: HOT
minutes: 11
prereqs: []
tags: [llm, api, prompting, structured-output, tool-calling, genai]
xp: 60
hot2026: true
---

Your manager drops a message on a Friday evening: "Can you prototype a feature that reads inbound support emails, extracts the issue and urgency, and auto-creates a Jira ticket? Monday morning demo." Your weekend just got interesting.

Four things stand between you and that demo: knowing how to call an LLM API, writing prompts that actually work, getting structured data out reliably, and wiring up tool calls so the model can *do* things — not just talk. This lesson covers all four. Get these right and you can build almost any LLM-powered feature.

## 1. The anatomy of an LLM API call

Every modern LLM API — Claude, GPT-4o, Gemini, Mistral — follows the same shape. You send a **messages array**. You get a **completion** back.

Each message has a **role** that tells the model who is speaking:

:::table {title="Message roles explained"}
| Role | Who sends it | What it's for |
|---|---|---|
| `system` | You, the developer | Sets persona, constraints, output format — before any user input |
| `user` | Your app or end user | The actual request or data to process |
| `assistant` | A previous model turn | Replays prior context in multi-turn conversations |
:::

Important: the model reads the *entire* conversation on every call. There is no persistent memory built in. You reconstruct the history yourself. That is why context length and cost management matter — but those are later-lesson problems.

:::why-prod
A vague system prompt is your biggest risk surface. Under edge-case inputs you never tested, the model will drift toward whatever makes intuitive sense to it — not what your product needs. A tight system prompt is the cheapest guardrail you have.
:::

Here is the minimal working call using the Anthropic Python SDK:

```python {title="Minimal Claude API call" run=false}
# pip install anthropic
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-5",   # swap to claude-opus-4-5 for harder reasoning tasks
    max_tokens=512,              # ALWAYS set this — prevents runaway cost
    system="You are a support triage assistant. Be concise and structured.",
    messages=[
        {"role": "user", "content": "My login has been broken since yesterday's deploy."}
    ],
)

print(response.content[0].text)
# → "Auth regression likely. Priority: High. Recommend rollback check on auth service."
```

Key parameters to know cold:
- `max_tokens` — hard cap on output length. Runaway completions = runaway bills.
- `temperature` — set to `0` for analytical or extraction tasks; `0.7–1.0` for creative tasks.
- `stop_sequences` — the model stops generating when it hits one of these strings. Very useful for parseable output formats.

## 2. Prompting that actually works

A prompt is just instructions. But the quality of those instructions is the difference between a demo and a production feature.

**Be explicit about format.** "Respond in JSON" is not enough. Show the exact shape you want. LLMs are pattern-matchers — give them the pattern.

**Put data last.** The system prompt sets the rules. The user message delivers the content to process. Do not mix them. Mixing them opens you up to prompt injection (a user-supplied value overriding your instructions).

**Add one worked example.** One-shot beats zero-shot for extraction tasks. Two shots rarely beats one for simple tasks — it just costs more tokens.

:::gotcha
"Respond only in JSON" often fails silently. The model prepends "Sure! Here's the JSON:" before the object, and your `json.loads()` throws an exception in production at 2am. Use structured output modes (next section) rather than relying on prompt wording alone.
:::

## 3. Structured outputs — reliable JSON every time

Asking politely for JSON is fragile. The right fix is to use the API's native structured output enforcement.

```python {title="Schema-enforced structured output" run=false}
import json, anthropic

client = anthropic.Anthropic()

SYSTEM = """
Extract info from the support email. Respond with ONLY valid JSON matching exactly:
{
  "issue_summary": "<one sentence>",
  "urgency": "low | medium | high | critical",
  "affected_component": "<component name or null>"
}
No prose. No markdown fences. JSON only.
"""

email = """
Hi team, payment gateway has been returning 500s since 14:00 IST.
Three enterprise clients are blocked. Need this fixed immediately.
"""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    system=SYSTEM,
    messages=[{"role": "user", "content": email}],
    stop_sequences=["```"],   # prevent model from wrapping output in markdown
)

data = json.loads(response.content[0].text)
print(data)
# → {"issue_summary": "Payment gateway returning 500s since 14:00 IST",
#    "urgency": "critical", "affected_component": "payment-gateway"}
```

For OpenAI-compatible endpoints, `response_format={"type": "json_object"}` enforces JSON at the API level — stronger than a prompt instruction. Anthropic's tool-use feature (next section) gives you full JSON Schema validation. Use whichever your provider exposes.

## 4. Tool calling — when the model needs to act

Tool calling (also called function calling) is how you let a model trigger real-world actions: query a database, create a Jira ticket, fire a webhook.

The flow has four steps:
1. You define tools with a name, description, and JSON Schema for their parameters.
2. The model decides *whether* to call a tool and returns a structured `tool_use` block.
3. **Your code** executes the actual function.
4. You send the result back; the model uses it to form its final reply.

The model never directly touches your systems. It *asks* you to call the function. You stay in control of execution. This is by design.

```python {title="Tool calling: auto-create a Jira ticket" run=false}
import anthropic, json

client = anthropic.Anthropic()

tools = [
    {
        "name": "create_jira_ticket",
        "description": "Creates a Jira ticket in the support project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary":   {"type": "string", "description": "One-line issue summary"},
                "priority":  {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                "component": {"type": "string", "description": "Affected system component"},
            },
            "required": ["summary", "priority"],
        },
    }
]

email = "Payment gateway 500s since 14:00 IST. Enterprise clients blocked. Fix ASAP."

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    tools=tools,
    system="You triage support emails and create Jira tickets for actionable issues.",
    messages=[{"role": "user", "content": email}],
)

for block in response.content:
    if block.type == "tool_use":
        print(f"Model wants to call: {block.name}")
        print(json.dumps(block.input, indent=2))
        # → Model wants to call: create_jira_ticket
        # → {"summary": "Payment gateway 500s since 14:00 IST",
        #    "priority": "Critical", "component": "payment-gateway"}

        # Your real code executes here — the model never calls this directly
        # result = jira_client.create_issue(**block.input)
```

:::war-story {title="The tool-call infinite loop at 3am"}
A team built an agent that queried their analytics DB to answer user questions. The system prompt said "keep trying until you find a confident answer." No max iteration guard. The model hit an ambiguous result and kept calling the same query tool in a loop. By morning: 50,000 DB queries, a $400 API bill, one very unhappy on-call engineer, and a postmortem that just said "add max_iterations=10." Always cap your agent loops.
:::

:::interview-line
"Tool calling is what separates a chat toy from a production agent — the model decides *what* to do, but your code stays in control of *execution*, which is where safety and reliability actually live."
:::

:::qa {q="What is the difference between a system prompt and a user message?"}
The system prompt is written by the developer at deploy time — it sets permanent rules, persona, and output format. The user message is runtime input: the actual data or question being processed. Mixing them risks prompt injection, where a user-supplied value silently overrides your instructions.
:::

:::qa {q="Why not just tell the model to 'respond in JSON' in the prompt instead of using structured output APIs?"}
Prompt-level JSON instructions are fragile. The model can prepend prose, wrap output in markdown code fences, or hallucinate schema fields — all of which break `json.loads()` silently in production. Structured output APIs enforce JSON at the transport level and validate against your schema, making parsing reliable under real traffic.
:::

:::drill {type="mcq" q="Which message role is set by the developer before any user interaction and defines the model's persona and constraints?"}
- [ ] `user`
- [x] `system`
- [ ] `assistant`
- [ ] `developer`
:::

:::drill {type="mcq" q="In a tool-calling flow, which statement is correct?"}
- [ ] The model calls the external API directly using the credentials you provided
- [ ] The model generates Python code that your runtime eval()s to call the function
- [x] The model returns a tool_use block; your code executes the actual function and returns the result
- [ ] Tool calling only works when temperature is 0
:::

:::key-takeaway
Four primitives unlock every LLM feature: the messages API (roles + context), a tight system prompt (your cheapest guardrail), structured output enforcement (not just prompt instructions), and tool calling (model decides, your code executes). Everything else in this track builds on top of these.
:::
