---
id: long-term-memory
track: t5-agentic
title: "Long-term memory & summarization"
badge: HOT
minutes: 9
prereqs: []
tags: [agents, memory, summarization, rag, langchain, production]
xp: 60
hot2026: true
---

Imagine your customer-support agent has been live for a month. Users love it. Then a power user writes in for the 40th time — and the agent greets them like a complete stranger. "Hi! How can I help you today?" No context. No history. No recollection that this person has escalated three billing issues in the past two weeks.

That is a long-term memory failure. And in production agentic systems, it is one of the most common reasons users churn.

## What is long-term memory in an agent?

Every agent has a context window — the text it can "see" right now. Think of it as working memory: fast, but tiny and ephemeral. When a session ends, it vanishes.

Long-term memory is the system that saves what matters *across* sessions and loads it back when the agent needs it later. It is external storage that the agent can read from and write to, just like a human jotting notes in a notebook.

The sibling lesson "Memory: short-term vs long-term vs vector" covers the taxonomy. This lesson focuses on the production mechanics: **how you store it, how you compress it, and how you decide what to retrieve**.

:::why-prod
Without long-term memory, every agentic session starts from zero. Users repeat themselves. Agents miss context that would change the answer. Personalization — the feature that makes AI assistants feel magical — is impossible at scale.
:::

## Three storage patterns you'll actually use

:::table {title="Long-term memory storage options"}
| Pattern | What it stores | Best for | Gotcha |
|---|---|---|---|
| Key-value store (Redis, DynamoDB) | Structured facts: name, preferences, last order | Fast lookup of known attributes | Schema gets messy fast |
| Vector database (Pinecone, Qdrant, pgvector) | Conversation chunks as embeddings | Semantic search over past chats | Stale embeddings if you forget to re-index |
| Relational DB + full text | Summaries + metadata | Auditable, filterable history | Slower; needs thoughtful indexing |
:::

Most production systems combine two: a key-value store for hard facts (user tier, language preference) and a vector DB for fuzzy recall ("did we discuss this topic before?").

## Summarization: the art of not running out of tokens

Here is the core tension. You want to remember everything. But you cannot fit everything into the context window. Even with 200 K token windows, loading six months of chat history is slow, expensive, and often harmful (the model buries the lede under old noise).

Summarization is the solution. You compress old conversation turns into a shorter representation before they leave the active window. There are three common strategies:

**Rolling summary** — every N turns, ask the LLM to rewrite the current summary with the new turns folded in. Simple. Drifts over time.

**Hierarchical summary** — summarize sessions into day summaries, day summaries into week summaries. Great for long-running agents. More infra.

**Selective extraction** — instead of summarizing everything, extract only the facts that matter: decisions made, commitments given, user preferences stated. Sparse but high-signal.

```python {title="Rolling summary with LangChain" run=false}
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

# pip install langchain-anthropic langchain-core
# set ANTHROPIC_API_KEY in your env

llm = ChatAnthropic(model="claude-3-5-haiku-20241022")  # cheap + fast for summarization

def roll_summary(existing_summary: str, new_turns: list[dict]) -> str:
    """
    Compress a growing conversation into a rolling summary.
    Call this every 10–20 turns before they leave the context window.
    """
    turns_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in new_turns
    )
    prompt = f"""You are a memory assistant. Given the existing summary and new conversation turns,
produce a concise updated summary (max 200 words). Preserve key facts, decisions, and user preferences.

EXISTING SUMMARY:
{existing_summary or "(none yet)"}

NEW TURNS:
{turns_text}

UPDATED SUMMARY:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# --- usage ---
summary = ""
batch = [
    {"role": "user", "content": "I want to cancel my Pro subscription."},
    {"role": "assistant", "content": "I can help. May I ask why?"},
    {"role": "user", "content": "Too expensive, but I might come back."},
]
summary = roll_summary(summary, batch)
print(summary)
# → "User (potential churn) wants to cancel Pro citing cost. Open to returning later."
```

## How retrieval actually works

Storing memories is the easy part. Knowing *which* memories to load at the start of a new session is where most systems break.

The standard pattern: at session start, embed the user's first message (or their profile ID), run a similarity search against your vector store, and prepend the top-k results as a `[MEMORY]` block in the system prompt. Keep the block short — 300–500 tokens max. Let the LLM use it as context, not as a data dump.

A better pattern for structured facts: always load the key-value profile unconditionally (name, tier, known preferences), then load semantic memory only if the first message suggests a historical reference ("like last time", "remember when", etc.).

:::gotcha
If you embed and store raw conversation turns, your vector DB fills up with noise: pleasantries, filler, repetition. Always summarize or extract facts *before* storing, not after. Searching over "sounds good!" and "thanks!" is actively harmful — it pollutes retrieval results.
:::

:::war-story {title="The 40-turn context bomb"}
A team built a coding assistant that stored every user message as a separate vector. After three months, each new session retrieved 20 stale code snippets from old projects. The model started hallucinating API names from libraries the user had stopped using. The fix was embarrassingly simple: they added a `last_used_at` timestamp and filtered retrievals to the past 30 days. Retrieval relevance jumped overnight. Lesson: memory without a TTL (time-to-live) strategy becomes a liability.
:::

## Connecting it all: a minimal memory loop

1. **Session start** — load key-value profile + top-3 semantic memories into system prompt.
2. **During session** — keep full turns in the active context window.
3. **After N turns** — run rolling summary; update the summary store.
4. **Session end** — extract structured facts (any new preferences/decisions stated); upsert into key-value store; embed the new summary and upsert into vector DB.
5. **Optionally** — apply a TTL policy: archive or delete memories older than X days.

This loop is framework-agnostic. LangGraph, CrewAI, or raw API calls — the loop is the same.

:::interview-line
"Long-term memory is just a retrieval problem: summarize aggressively to fit the signal, embed to find it later, and always apply a TTL so stale context doesn't poison future sessions."
:::

:::qa {q="Why not just use a very long context window instead of external memory?"}
Long context windows are expensive and slow, and models often under-attend to information buried in the middle. External memory lets you retrieve only what is relevant to *this* turn, keeping the context lean and the model focused. You also get persistence across sessions, which no context window can give you.
:::

:::qa {q="What is the difference between a rolling summary and selective extraction?"}
A rolling summary rewrites a prose summary each time, folding in new turns — great for narrative continuity but lossy. Selective extraction pulls out discrete facts (name, preference, decision) into structured slots — lower recall but higher precision. Production systems often combine both: a prose summary for fuzzy recall and a structured profile for hard facts.
:::

:::qa {q="How do you decide what NOT to store in long-term memory?"}
Anything ephemeral or low-signal: greetings, filler, one-off clarifications that won't matter next session. A good heuristic: if you wouldn't write it in a handoff note to a colleague, don't store it. Storing less, more precisely, consistently beats storing everything and letting retrieval sort it out.
:::

:::drill {type="mcq" q="A user's conversation history has grown to 500 turns over three months. What is the BEST approach for loading memory at session start?"}
- [ ] Load all 500 turns verbatim into the context window
- [x] Load a rolling summary plus top-k semantically relevant excerpts, capped at ~400 tokens
- [ ] Load only the most recent 10 turns and ignore older history
- [ ] Disable long-term memory for users with large histories
:::

:::drill {type="mcq" q="You store raw conversation turns as vectors and notice retrieval quality is poor. What is the most likely root cause?"}
- [ ] The embedding model is too small
- [ ] The vector database is misconfigured
- [x] Noise turns (greetings, filler) are polluting the index and burying signal
- [ ] Long-term memory cannot work with vector databases
:::

:::key-takeaway
Long-term memory = compress aggressively (summarize before storing), retrieve selectively (embed + semantic search + TTL), and load minimally (300–500 tokens of context, not the full archive). Get these three right and your agent stops feeling like it has amnesia.
:::
