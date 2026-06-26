---
id: orchestration-frameworks
track: 09-genai-rag
title: "Orchestration: LangChain vs LlamaIndex vs LangGraph"
badge: HOT
minutes: 9
prereqs: []
tags: [langchain, llamaindex, langgraph, orchestration, rag, agents, genai]
xp: 60
hot2026: true
---

You're three weeks into building a customer-support bot for a Pune fintech startup. It works — mostly. But the code is 14 nested `.pipe()` calls. A bug only shows up in production. And when your manager asks "can we add a retry if the LLM goes off-script?" you realise your chain has no way to loop back.

You've hit the orchestration wall.

Welcome to the part of GenAI engineering nobody talks about in tutorials: picking the right plumbing *before* the pipes get too tangled to fix.

## What "orchestration" actually means here

An LLM by itself is a box: text in, text out. A real GenAI app is a *workflow* — retrieve docs, call a tool, decide what to do next, maybe loop, finally synthesise an answer. Orchestration frameworks give you the glue to build that workflow without writing plumbing from scratch.

There are three dominant choices in 2025–26: **LangChain**, **LlamaIndex**, and **LangGraph**. They overlap, but each has a very different sweet spot.

:::why-prod
Wrong framework choice means a painful rewrite at scale. LangChain's abstraction layers slow debugging to a crawl. LlamaIndex's RAG primitives save you weeks on document pipelines. LangGraph's explicit state machine prevents silent agent loops that burn tokens — and money — in production.
:::

## The three players

**LangChain** is the Swiss Army knife. It exploded in 2023 and has integrations for almost every LLM, vector DB, and tool you'll encounter. Its model is *chains* — composable units that pipe the output of one step into the next. Great for prototyping. Gets messy fast when logic turns non-linear.

**LlamaIndex** is the RAG specialist. Its mental model is *data connectors → index → query engine*. If your job is ingesting PDFs, chunking, embedding, and answering questions over a document corpus, LlamaIndex gives you the right abstractions out of the box. Less opinionated about agents; laser-focused on retrieval quality.

**LangGraph** (from the LangChain team, but a separate library) models your workflow as a *directed graph with shared state*. Each node is a function. Edges are conditional. A state object flows through every node. This makes complex logic — retries, human-in-the-loop, multi-agent routing — explicit and debuggable instead of hidden inside callbacks you can't inspect.

:::table {title="Framework quick-comparison"}
| | LangChain | LlamaIndex | LangGraph |
|---|---|---|---|
| **Best for** | Rapid prototyping, broad integrations | Document-heavy RAG pipelines | Stateful, multi-step agents |
| **Mental model** | Chains / Runnables | Connectors → Index → QueryEngine | Nodes + edges + shared state |
| **Agent support** | Yes (AgentExecutor) | Yes (ReActAgent) | Native — built for this |
| **Debugging** | Hard (deep call stacks) | Moderate | Easy (visualise the graph) |
| **Learning curve** | Medium (huge docs) | Low (focused API) | Medium–High |
| **When to avoid** | Complex conditional logic | General agent workflows | Simple single-step chains |
:::

## A minimal LangGraph state machine

Here is the simplest possible LangGraph workflow: two nodes, one conditional edge. This pattern is behind 90 % of production agentic RAG systems.

```python {title="LangGraph minimal agent loop" run=false}
# pip install langgraph langchain-openai
# Free alternative: replace ChatOpenAI with ChatOllama("llama3") via Ollama

from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    attempts: int

def retrieve(state: AgentState) -> AgentState:
    # Swap in your real vector-DB retriever here
    state["context"] = f"[retrieved docs for: {state['question']}]"
    return state

def generate(state: AgentState) -> AgentState:
    # Swap in your actual LLM call here
    state["answer"] = f"Based on context: {state['context']}"
    state["attempts"] = state.get("attempts", 0) + 1
    return state

def should_retry(state: AgentState) -> str:
    # Conditional edge — loop back up to 2 times if answer looks thin
    if len(state["answer"]) < 20 and state["attempts"] < 2:
        return "retrieve"
    return END

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", should_retry)

app = graph.compile()
result = app.invoke({"question": "What is RAG?", "context": "", "answer": "", "attempts": 0})
print(result["answer"])
```

The `should_retry` function is the piece LangChain can't do cleanly. In LangGraph it's just an `if` statement that returns a node name. You can read it, test it, and reason about it.

:::gotcha
LangChain's `AgentExecutor` silently swallows tool errors and retries on its own schedule. You don't see it happening and you can't easily cap it. In production this burns tokens and hides bugs. LangGraph makes every retry loop *explicit code you own*. Migrate any stateful agent to LangGraph before it hits prod.
:::

:::war-story {title="The infinite loop that billed ₹80k overnight"}
A Bangalore startup's support agent ran fine in staging. In production, one specific query triggered a tool that always returned an ambiguous result. LangChain's `AgentExecutor` interpreted this as "not done yet" and kept calling the tool — 3,400 times across eight hours. No alerting, no loop cap, no circuit breaker. Next morning: ₹80k in API bills and a very unhappy CTO. Migrating to LangGraph took two days; they added a hard `attempts < 5` guard on every conditional edge. Zero repeats since.
:::

## How to pick in practice

- **New RAG feature over docs?** — Start with LlamaIndex. Its `QueryEngine` + `NodePostprocessor` chain handles ingestion, retrieval, and reranking in ~30 lines.
- **Quick prototype or integration glue?** — LangChain. Its ecosystem is unmatched for speed.
- **Multi-step agent with branching or retry logic?** — LangGraph, even on day one. The explicitness pays off the moment you need a loop cap or a human approval step.
- **Enterprise agent platform?** — LangGraph + LangSmith for tracing. This is where serious production teams are landing in 2026.

You can also mix them. Using LlamaIndex as the retrieval layer inside a LangGraph agent is common and sensible — each framework stays in its lane.

:::interview-line
"I default to LlamaIndex for pure RAG pipelines and LangGraph for anything with conditional logic or retries — LangChain's abstraction cost isn't worth it once the flow turns non-linear."
:::

:::qa {q="When would you choose LangGraph over LangChain's AgentExecutor?"}
LangGraph models your workflow as an explicit graph with typed shared state, so every transition and retry is visible and testable. LangChain's `AgentExecutor` hides control flow inside callbacks, making it hard to add conditional logic, loop caps, or human-in-the-loop gates. Anything beyond a simple single-pass agent belongs in LangGraph where you own the control flow entirely.
:::

:::qa {q="A candidate says 'I just use LangChain for everything.' What would you probe?"}
LangChain is a great prototyping tool, but its deep abstraction stacks make debugging painful and stateful logic nearly unreadable at scale. I'd ask how they handle retry caps, conditional routing, and loop detection in production. A senior engineer knows which layer each framework optimises for and switches deliberately — not out of familiarity alone.
:::

:::drill {type="mcq" q="Your team ingests 500 PDFs nightly and answers employee questions over them. No complex agent logic needed. Which framework fits best?"}
- [ ] LangGraph — it handles all GenAI workflows
- [x] LlamaIndex — it's built around data connectors, indexing, and query engines
- [ ] LangChain LCEL — it has the most third-party integrations
- [ ] Build a custom pipeline; frameworks add too much overhead
:::

:::drill {type="mcq" q="What is the core advantage of LangGraph's nodes + edges + shared state model over LangChain's AgentExecutor?"}
- [ ] It runs inference faster at the model level
- [ ] It requires less code for simple single-step chains
- [x] Control flow is explicit — retries, conditions, and loops are visible code you can read, test, and cap
- [ ] It automatically selects the best LLM for each node at runtime
:::

:::key-takeaway
Match the framework to the problem: LlamaIndex for document RAG, LangGraph for stateful agents, LangChain for quick glue. Familiarity is the worst reason to pick a framework — the wrong choice shows up as a ₹80k overnight bill or a three-day debugging session you could have avoided.
:::
