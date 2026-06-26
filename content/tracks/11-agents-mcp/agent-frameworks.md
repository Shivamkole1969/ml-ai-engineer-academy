---
id: agent-frameworks
track: 11-agents-mcp
title: "Frameworks: LangGraph vs CrewAI vs AutoGen"
badge: HOT
minutes: 9
prereqs: []
tags: [agents, langgraph, crewai, autogen, multi-agent, production]
xp: 60
hot2026: true
---

Your team just greenlit an AI agent for automating customer refund decisions. The PM wants it done in two weeks. You open a browser, search "Python agent framework", and immediately hit three names: LangGraph, CrewAI, AutoGen. Every blog post says a different one is "best." You pick wrong, and in month two you're fighting the framework instead of shipping features.

This lesson is the cheat sheet. Know when to reach for which one — and exactly why it matters when your agent starts misbehaving at 2am.

## The Three Contenders

Think of these frameworks like three kinds of project managers.

**LangGraph** is the meticulous engineer who draws a flowchart before writing a single line. It models your agent as a *directed graph* — nodes are actions, edges are transitions, and a shared *state* object flows through every step. You decide exactly when to loop, when to call a tool, when to stop. It's verbose. It's also the most production-battle-tested of the three.

**CrewAI** is the team lead who assigns roles. You define agents with job titles — a Researcher, a Writer, a QA Reviewer — and a *crew* runs them in sequence or in parallel. Less boilerplate, faster prototypes. The flow is more opaque though; you're trusting the framework to handle hand-offs.

**AutoGen** (Microsoft) is the Slack channel that never ends. Agents are conversational actors who message each other until a task is done. It shines for code-generation loops: a Coder agent writes Python, an Executor agent runs it and reports back, and they iterate. `GroupChat` wires up N agents in a round-robin or custom speaker pattern.

:::why-prod
Your framework choice sets your ceiling. LangGraph lets you add checkpointing, human-in-the-loop approvals, and conditional retry logic at the graph level — things you'll need in production. CrewAI and AutoGen are faster to start but harder to instrument when something goes wrong at scale.
:::

:::table {title="Framework Decision Matrix"}
| Criterion | LangGraph | CrewAI | AutoGen |
|---|---|---|---|
| Flow control | Explicit, graph-based | Role-based, semi-auto | Conversation-driven |
| State management | Built-in, persistent | Basic | Message history only |
| Best use case | Stateful production pipelines | Multi-role task crews | Code-gen & eval loops |
| Human-in-the-loop | First-class support | Workaround needed | Possible, but manual |
| Observability | LangSmith native | Limited | Limited |
| Learning curve | Medium–High | Low | Medium |
| Maturity (2026) | High | Medium | Medium |
:::

## LangGraph: The Production Choice

LangGraph treats your agent as a state machine. You define a `TypedDict` for state, add nodes (plain Python functions), and wire edges between them. The runner handles retries, checkpointing to Postgres or Redis, and streaming — out of the box.

```python {title="Minimal LangGraph agent skeleton" run=false}
# pip install langgraph langchain-openai
# Works locally with any OpenAI-compatible endpoint — Ollama is free

from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Shared state — every node reads and writes this dict
class AgentState(TypedDict):
    query: str
    tool_result: str
    final_answer: str

# 2. Node: reasoning step (swap for any LLM provider)
def llm_node(state: AgentState) -> AgentState:
    # In real code: call Claude / GPT / local Ollama here
    print(f"Reasoning over: {state['query']}")
    state["tool_result"] = "mock tool output"
    return state

# 3. Node: format the answer
def answer_node(state: AgentState) -> AgentState:
    state["final_answer"] = f"Answer based on: {state['tool_result']}"
    return state

# 4. Wire the graph
builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("answer", answer_node)
builder.set_entry_point("llm")
builder.add_edge("llm", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

# 5. Run — state flows through each node in order
result = graph.invoke({"query": "What is the refund policy?", "tool_result": "", "final_answer": ""})
print(result["final_answer"])
```

The key insight: every node is just a Python function. You can unit-test each one in isolation. That's rare in agent-land and worth a lot when debugging production failures at a specific step.

## CrewAI: Quick Multi-Role Prototyping

CrewAI lets you declare agents with roles and goals, then assign them tasks. The framework routes outputs between agents automatically. It's the fastest way to stand up a "research → draft → review" pipeline where the hand-off logic is linear.

The downside: when an agent misbehaves, the framework's abstraction makes it hard to pinpoint *which step* failed. You end up reading raw LLM logs rather than a clean state diff.

## AutoGen: When Agents Need to Write and Run Code

AutoGen's standout feature is the `UserProxyAgent` — it actually *executes* Python in a subprocess and feeds stdout back into the conversation. For evaluation pipelines, data-wrangling agents, or any workflow where "write code → run it → fix bugs" is the loop, AutoGen is the fastest path.

One hard rule: always set `max_turns` and a token budget. AutoGen's conversation model re-sends the full message history on every turn, so costs compound fast with no ceiling.

:::gotcha
Don't choose CrewAI or AutoGen for your first *production* agent just because the README demo looks clean. Both hide state transitions inside the runtime. When your agent starts looping at midnight and you need to checkpoint, replay, or surgically restart a failed step — you'll wish you'd used LangGraph. Demos are deceptive; production is not.
:::

:::war-story {title="The CrewAI loop that ate ₹12,000 of API credits overnight"}
A Pune-based startup shipped a content-generation crew on a Friday evening. The Researcher agent kept failing silently on a rate-limited search API — returning empty strings instead of raising an exception. CrewAI's retry logic re-queued the whole crew, the Researcher failed again, and the loop ran all weekend. By Monday, 8,000+ API calls later, the bank account was drained and the task queue was full of empty drafts. The same pipeline in LangGraph would have been one conditional edge: `if tool_result == "": goto human_review`. Always fail loud, never fail silent in agent pipelines.
:::

:::interview-line
"I reach for LangGraph when I need production guarantees — checkpointing, observability, and surgical control over flow. CrewAI is my rapid prototype tool, and AutoGen is my go-to for code-execution loops."
:::

:::qa {q="When would you choose LangGraph over CrewAI?"}
LangGraph is the right call when your agent needs to run in production with persistence, human-in-the-loop steps, or complex conditional branching. It gives you explicit state management and first-class checkpoint support so you can replay from any step. CrewAI gets you to a demo faster but becomes painful to debug when agents misbehave mid-run in production.
:::

:::qa {q="What is AutoGen's main strength compared to LangGraph?"}
AutoGen excels at code-generation loops where an agent writes code, executes it, reads the output, and iterates — its `UserProxyAgent` handles sandboxed execution natively. LangGraph can do this too, but you'd wire it manually. The trade-off: AutoGen's conversation model re-sends full history every turn, so token costs compound quickly without a hard cap.
:::

:::qa {q="How does LangGraph handle a failure mid-run in a long agent pipeline?"}
LangGraph supports checkpointing to a persistence layer — Postgres, Redis, or in-memory. If a node fails, you replay the graph from the last successful checkpoint rather than restarting from scratch. For long-running agents this is critical: restarting from zero means re-spending money and time on steps that already succeeded.
:::

:::drill {type="mcq" q="Your team is building a customer support agent that needs human approval before issuing refunds. Which framework handles this most naturally out of the box?"}
- [ ] CrewAI, because role-based agents can simulate a human reviewer
- [x] LangGraph, because it has first-class human-in-the-loop support at graph edges
- [ ] AutoGen, because its conversational model naturally pauses for input
- [ ] All three handle this equally well
:::

:::drill {type="mcq" q="An AutoGen agent is running a code-writing loop with no max_turns set. What is the most likely production risk?"}
- [ ] The agent will refuse to write code after a few rounds
- [ ] State will be lost between agent turns
- [x] Token costs will balloon because every turn re-sends the full conversation history as context
- [ ] AutoGen does not support code execution, making this setup invalid
:::

:::drill {type="mcq" q="What is the key architectural difference between LangGraph and CrewAI?"}
- [ ] LangGraph is conversation-based; CrewAI is graph-based
- [ ] CrewAI supports tool calling; LangGraph does not
- [x] LangGraph uses an explicit graph with shared typed state; CrewAI uses role-based agents with implicit orchestration
- [ ] They are architecturally identical — just different APIs
:::

:::key-takeaway
LangGraph for production, CrewAI for quick multi-role prototypes, AutoGen for code-execution loops. When in doubt, pick LangGraph — explicit state and checkpointing are what saves you at 2am.
:::
