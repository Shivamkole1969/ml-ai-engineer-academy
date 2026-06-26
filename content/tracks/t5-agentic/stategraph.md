---
id: stategraph
track: t5-agentic
title: "LangGraph: stateful graphs & cycles"
badge: HOT
minutes: 9
prereqs: []
tags: [langgraph, agents, stateful, graphs, cycles, langchain, agentic-ai]
xp: 60
hot2026: true
---

Imagine you've shipped a customer-support agent. It can look up orders, check inventory, and issue refunds. Day one goes great. Then a user sends: "My refund didn't arrive. Can you check the order, verify the bank account, and try again?" The agent calls the order lookup tool — and then forgets what it found before calling the bank tool. By the time it tries the refund, it has already lost the thread. It either hallucinates an answer or starts the whole thing over. That's what happens when your agent has no persistent state. LangGraph exists to fix exactly this.

## What LangGraph actually is

LangGraph is a library built on top of LangChain that lets you model an agent as a **directed graph**. Each **node** is a Python function. Each **edge** is a transition — "after this node, go to that node." The killer feature: edges can **loop back**, creating cycles.

Why does that matter? Because agents aren't pipelines. A pipeline runs once, top to bottom, done. An agent needs to:

- Call a tool, see the result, decide whether to call another tool or stop.
- Retry on failure.
- Ask a clarifying question and wait for the user's reply before continuing.

All of that requires cycles — the ability to revisit a node after you've been somewhere else. Traditional chains can't do that. Graphs can.

## The state object: the agent's shared memory

The centerpiece of every LangGraph app is a **State** — a typed dictionary that every node reads from and writes to. Think of it as a whiteboard that all nodes share. When a node runs, it receives the current state and returns a patch (the fields it wants to update). LangGraph merges that patch back into the state automatically.

```python {title="Minimal LangGraph agent loop" run=false}
# pip install langgraph langchain-openai
# Free alternative: use langchain-community + Ollama for a local model

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Define what state looks like ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # add_messages merges, not overwrites
    iteration: int
    done: bool

# --- 2. Write nodes (plain Python functions) ---
def call_model(state: AgentState) -> dict:
    """Ask the LLM what to do next."""
    # In real life, bind tools here and call your LLM
    last_msg = state["messages"][-1].content
    reply = AIMessage(content=f"[Model thinking about: {last_msg}]")
    return {"messages": [reply], "iteration": state["iteration"] + 1}

def should_continue(state: AgentState) -> str:
    """Routing function: returns the name of the next node."""
    if state["iteration"] >= 3 or state["done"]:
        return "end"          # go to END node
    return "model"            # loop back to call_model

# --- 3. Build the graph ---
graph = StateGraph(AgentState)

graph.add_node("model", call_model)
graph.add_edge(START, "model")               # entry point

graph.add_conditional_edges(               # <-- this creates the cycle
    "model",
    should_continue,
    {"model": "model", "end": END},        # map return values to node names
)

app = graph.compile()

# --- 4. Run it ---
result = app.invoke({
    "messages": [HumanMessage(content="Check my refund status")],
    "iteration": 0,
    "done": False,
})
print(result["messages"][-1].content)
```

Three things to notice:
1. `AgentState` is your whiteboard. Every node gets it, every node can update it.
2. `add_messages` is a **reducer** — it appends new messages instead of replacing the list. Reducers are how LangGraph safely merges partial updates.
3. `add_conditional_edges` is where the cycle lives. The routing function `should_continue` inspects state and returns a string that maps to the next node — including looping back to `"model"`.

:::why-prod
In production, stateful graphs let you checkpoint an agent mid-run, resume it after a crash, and audit exactly which node failed and with what state. Without this, a long-running agent failure means starting over — and explaining to the user why their 10-minute task vanished.
:::

## Nodes, edges, and the routing function

:::table {title="LangGraph building blocks"}
| Building block | What it is | Production use |
|---|---|---|
| Node | Python function `(state) -> dict` | One unit of work: call LLM, run tool, validate output |
| Edge | Hard transition from node A to node B | Guaranteed next step (e.g. always log after tool call) |
| Conditional edge | Router function picks next node | Branch on tool result, error, or iteration count |
| Reducer | Merge function for a state field | Safely accumulate messages, tool results, or scores |
| Checkpointer | Persists state to DB between nodes | Resume on crash; human-in-the-loop approval flows |
:::

## Cycles without guardrails are just infinite loops

:::gotcha
The most common LangGraph mistake in production: forgetting a termination condition inside the routing function. If your router always returns `"model"` — maybe because your LLM never emits a "done" signal — the graph runs forever, burning tokens and money. Always add a hard cap: `if state["iteration"] >= MAX_STEPS: return "end"`. Set `MAX_STEPS` to something conservative (5–10) and raise it deliberately.
:::

## Checkpointing: pause, resume, approve

One of LangGraph's killer production features is the **checkpointer**. Attach a SQLite or PostgreSQL checkpointer and every node transition is saved. If the process crashes at node 7 of 12, you resume from node 7 — not from scratch.

The same mechanism enables **human-in-the-loop** (HITL) flows: the graph pauses at an "approval" node, a human reviews the proposed action in a UI, clicks approve, and the graph continues. This is how you build agents that can draft a refund but require a human sign-off before money actually moves.

:::war-story {title="The $800 loop"}
A team shipped a research agent on a Friday. The routing function checked whether the LLM's response contained the word "FINAL" — the LLM's signal that it was done. Over the weekend, the model started occasionally returning responses where "FINAL" appeared inside a quoted block of text the LLM was analyzing. The router saw "FINAL" in the wrong place, routed to END immediately, producing garbage output. Another code path had no iteration cap, so a slightly different prompt pattern triggered an infinite loop through a paid search API. By Monday morning: 40,000 search calls, an $800 bill, and a very awkward Slack thread. The fix was two lines — a hard iteration cap and a regex that matched only the sentinel at the start of the LLM reply. Ship the cap first, ask questions later.
:::

:::interview-line
"LangGraph models agents as stateful directed graphs where cycles are first-class — that's what lets you build loops, retries, and human-in-the-loop flows without hacking around a linear chain."
:::

:::qa {q="What is the purpose of a reducer in LangGraph state?"}
A reducer tells LangGraph how to merge a partial state update from a node back into the shared state. Without a reducer, each node would overwrite the field entirely. The built-in `add_messages` reducer, for example, appends new messages to the existing list rather than replacing it — which is exactly what you want in a conversation that grows over multiple node invocations.
:::

:::qa {q="How does LangGraph differ from a simple LangChain chain for building agents?"}
A LangChain chain is a linear directed acyclic graph — data flows one way, no loops. An agent needs to cycle: call a tool, evaluate the result, decide to call another tool or stop, then possibly retry on error. LangGraph adds first-class cycle support through conditional edges and routing functions, plus persistent state that threads through every node. That combination is what makes complex, multi-step agents reliable enough to run in production.
:::

:::qa {q="When would you use a LangGraph checkpointer?"}
Any time an agent task runs longer than a single HTTP request, or when you need a human to approve an action mid-run. The checkpointer saves graph state to a database after every node, so you can resume after a crash, implement time-outs with graceful recovery, or pause the graph and wait for asynchronous input — like a user clicking "Approve" in a dashboard — before continuing.
:::

:::drill {type="mcq" q="You add `add_conditional_edges('model', router, {'model': 'model', 'end': END})`. What is the minimum extra safeguard you must add to avoid an infinite loop?"}
- [ ] Set `max_concurrency=1` on the compiled graph
- [ ] Use a PostgreSQL checkpointer instead of the in-memory default
- [x] Add an iteration or step counter to state and have the router return `'end'` when it exceeds a threshold
- [ ] Replace conditional edges with hard edges so the graph always terminates
:::

:::drill {type="mcq" q="A LangGraph node function signature is `def my_node(state: AgentState) -> dict`. What should the returned dict contain?"}
- [ ] The complete, fully updated AgentState object
- [ ] A LangChain `RunnableOutput` wrapping all state fields
- [x] Only the fields that changed — LangGraph merges the partial update into the existing state
- [ ] The next node name to route to
:::

:::key-takeaway
LangGraph turns your agent into a stateful directed graph where cycles, retries, and human-in-the-loop pauses are first-class features — but every cycle needs a hard iteration cap or you will pay for an infinite loop at 3 a.m.
:::
