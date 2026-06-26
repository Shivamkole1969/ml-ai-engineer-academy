---
id: short-term-memory
track: t5-agentic
title: "Memory: short-term vs long-term vs vector"
badge: HOT
minutes: 9
prereqs: []
tags: [memory, agents, vector-db, embeddings, context-window, rag]
xp: 60
hot2026: true
---

Imagine you've built a customer support agent. Day one: a user tells it "I prefer concise replies — just bullet points." Day two: they come back with a follow-up question. The agent greets them like a total stranger and writes a three-paragraph essay. They're annoyed. You're confused. The logic looks fine. What went wrong?

Memory architecture. The agent had none.

## What "memory" actually means for an agent

Agents don't have memory in any biological sense. They have **context** — the list of messages, tool outputs, and documents currently loaded into the prompt. When the session ends, that context evaporates.

So "memory" in agent design is really the answer to one question: **where does the agent store and retrieve information, and for how long?**

There are three distinct layers:

1. **Short-term memory** — the live context window
2. **Long-term memory** — persistent storage that survives between sessions
3. **Vector memory** — semantic search over past interactions or large corpora

Each layer solves a different problem. A real production agent needs all three.

:::why-prod
Most agent failures in production aren't about reasoning quality — they're about forgetting. An agent that re-reads the same document 10 times, forgets a user preference mid-session, or hallucinates a fact it saw two turns ago is expensive, slow, and erodes trust fast. Getting memory layers right is the difference between a compelling demo and a product people actually use.
:::

## Short-term memory: the context window

Short-term memory is everything currently in the prompt — conversation history, system instructions, tool results, retrieved chunks, the whole thing.

It's zero-latency and requires no extra infrastructure. It's also **finite**.

Modern models have large windows: 128k tokens for GPT-4o, 200k for Claude, up to 1M for some Gemini variants. That sounds enormous until your agent is 40 turns deep, has called 12 tools, and pasted in a 30-page PDF. Context fills faster than you expect.

When the window gets full, your options are:

- **Truncate** old messages (you lose early context — often the most important part)
- **Summarize** old turns into a compact digest and swap that in
- **Evict** information to long-term or vector memory

None of these are automatic. You have to build the strategy. The sibling lesson "Long-term memory & summarization" goes deep on this — but know that short-term memory is always your first layer and your fastest one.

## Long-term memory: persisting across sessions

Long-term memory is anything that outlives the conversation — user preferences, learned facts, past decisions, task history. You implement it with real storage: a relational database, Redis, a key-value store, even a JSON file to start.

The agent **writes** to it deliberately ("the user prefers metric units, bullet-point style") and **reads** from it at the start of each new session, injecting the relevant facts into the system prompt.

The hard part is deciding what to write and when. Write everything and you get noise. Write nothing and you get the amnesiac from our opening scenario.

A reliable pattern: at the end of each session, call the LLM once to extract a compact set of "facts worth remembering" from the transcript, then store those in structured form.

```python {title="File-based long-term memory (swap for Redis/Postgres in prod)" run=false}
import json
import pathlib

MEMORY_FILE = pathlib.Path("user_memory.json")

def load_memory(user_id: str) -> dict:
    """Return stored facts for this user, or empty dict if none."""
    if not MEMORY_FILE.exists():
        return {}
    data = json.loads(MEMORY_FILE.read_text())
    return data.get(user_id, {})

def save_memory(user_id: str, new_facts: dict) -> None:
    """Merge new facts into existing memory for this user."""
    data = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}
    data[user_id] = {**data.get(user_id, {}), **new_facts}
    MEMORY_FILE.write_text(json.dumps(data, indent=2))

# After a session, extract facts with your LLM, then persist them:
save_memory("user_42", {
    "preferred_tone": "concise",
    "preferred_format": "bullet points",
    "timezone": "UTC+5:30",
})

# At the start of the next session, reload and inject into system prompt:
facts = load_memory("user_42")
system_prompt = f"""You are a helpful assistant.
User preferences: {json.dumps(facts)}
Always follow these preferences."""
```

## Vector memory: semantic retrieval at scale

Structured long-term memory works great for known keys — preferences, flags, user IDs. But what about **unstructured knowledge**? A user's 300 past support tickets. A 600-page product manual. 18 months of agent decision logs.

You can't fit all of that in the context window. You can't query it meaningfully with SQL. That's where vector memory comes in.

The idea: convert each piece of text into an **embedding** — a dense numerical vector that captures meaning. Store those vectors in a vector database. When the agent needs context, run a similarity search: "find the 5 chunks most semantically related to this query."

This is the same mechanism behind RAG (Retrieval-Augmented Generation). Applied to an agent's own past interactions, it's sometimes called **episodic memory**.

Popular vector stores: **Chroma** (local, free, great for dev), **pgvector** (PostgreSQL extension, keeps your stack simple), **Pinecone** (managed cloud), **Weaviate** (open-source, self-hostable).

:::table {title="Three memory layers compared"}
| Layer | Where it lives | Survives session? | Latency | Best for |
|---|---|---|---|---|
| Short-term | Context window | No | ~0 ms | Current task, active turns |
| Long-term | DB / KV store / file | Yes | Low | Preferences, learned facts, decisions |
| Vector | Vector database | Yes | Low–medium | Semantic search over large corpora |
:::

:::gotcha
The most common mistake: stuffing everything into the context window because it's easy, then wondering why the agent crashes or behaves strangely at turn 30. Context overflow doesn't always throw an error — sometimes the model silently ignores the oldest tokens, and you lose your system prompt. Design your memory layers before you start, not after you hit the wall.
:::

:::war-story {title="The helpdesk agent that billed $4,000 overnight"}
A team built an internal IT helpdesk agent. No vector memory — every new ticket triggered a fresh LLM call that loaded all previous tickets into context for "reference." During a high-traffic Monday, 800 tickets came in overnight. Each one triggered a ~50k-token call. By morning: $4,000 in API costs, the context window was saturated anyway, and half the responses were wrong because relevant tickets were drowning in noise. Switching to a Chroma index with top-5 retrieval cut costs to roughly $40, improved answer quality, and took one afternoon to implement.
:::

:::interview-line
"Short-term memory is the context window — fast but finite. Long-term memory persists across sessions in a database. Vector memory retrieves semantically relevant chunks at scale. A production agent needs a deliberate strategy for all three."
:::

:::qa {q="What happens when an agent's context window fills up?"}
The model can only process tokens that fit within its limit. Without active memory management — truncation, summarization, or eviction to external storage — the agent either errors out or silently drops the oldest tokens. That often means losing the system prompt or early instructions, which breaks behavior in subtle and hard-to-debug ways. The fix is building overflow handling into your agent loop from day one.
:::

:::qa {q="When would you use vector memory instead of a structured database?"}
When you need **semantic** retrieval over unstructured text, and you don't know exactly what you'll need in advance. SQL is great for exact matches on known fields — "find tickets from user 42." A vector search is great for fuzzy meaning — "find past tickets that are conceptually similar to this new one." Use structured storage for facts with clear keys; use vector storage for large text corpora where relevance matters more than exact lookup.
:::

:::drill {type="mcq" q="Your agent needs to remember a user's preferred output language across separate sessions. Which memory layer fits best?"}
- [ ] Short-term memory (keep it in the context window)
- [x] Long-term memory (persist to a database or key-value store)
- [ ] Vector memory (embed the preference and retrieve by similarity)
- [ ] No memory needed — just detect the language from each message
:::

:::drill {type="mcq" q="A research agent has indexed 10,000 past reports. When answering a new query, it needs to pull in the 3 most relevant past reports. What's the right approach?"}
- [ ] Load all 10,000 reports into the context window on every query
- [ ] Store reports in SQL and use full-text keyword search
- [x] Embed all reports in a vector store and run a similarity search at query time
- [ ] Fine-tune the model on all past reports so it "knows" them
:::

:::key-takeaway
Agent memory has three layers: short-term (context window — fast, finite, gone when the session ends), long-term (persisted storage for preferences and learned facts), and vector (semantic retrieval over large text corpora). Design all three intentionally — most production agent failures trace back to a missing or overloaded memory layer.
:::
