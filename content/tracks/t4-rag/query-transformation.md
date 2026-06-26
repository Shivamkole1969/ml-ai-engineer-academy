---
id: query-transformation
track: t4-rag
title: "Query transformation: HyDE, decomposition"
badge: HOT
minutes: 9
prereqs: []
tags: [rag, hyde, query-decomposition, retrieval, llm, embeddings]
xp: 60
hot2026: true
---

Imagine your RAG chatbot has been live for two weeks and the feedback is rolling in: "It keeps saying it doesn't know things that are clearly in our docs." You dig in. The retrieval logs show the right chunks are *in* the database — they just never get retrieved. Why? Because users write questions like "why does my invoice keep failing?" and the docs say "payment processing errors occur when…". The query and the answer live in different parts of embedding space. Your retrieval never had a chance.

That gap between *how people ask* and *how content is written* is the core problem query transformation solves.

## What query transformation actually is

A vanilla RAG pipeline embeds the user's raw query and does a nearest-neighbour search. Simple. Brittle.

Query transformation means *rewriting or expanding the query before retrieval* so the embedding lands closer to useful chunks. There are two heavy hitters worth knowing cold.

**HyDE — Hypothetical Document Embeddings**

The idea is elegant: instead of embedding the question, ask an LLM to *imagine what the perfect answer document would look like* and embed *that*. A hypothetical answer lives in the same stylistic neighbourhood as real answers, not in the question neighbourhood.

Step by step:
1. User asks: "why does my invoice keep failing?"
2. You call an LLM with: "Write a short paragraph from a knowledge base that would answer this question."
3. LLM returns: "Payment processing errors typically occur when the billing address does not match…"
4. You embed *that* text and search. Suddenly you're searching in answer-space.

HyDE consistently lifts recall on knowledge-base and technical documentation use cases. It costs one extra LLM call per query — cheap for most production loads.

**Query decomposition (multi-query / step-back)**

Some questions are compound. "Compare our Pro and Enterprise plans for a team of 50 that needs SSO and audit logs" is really four questions at once. No single chunk answers all of it.

Decomposition asks an LLM to split the question into atomic sub-queries, retrieves for each, deduplicates, then feeds the merged context to the final LLM. This dramatically extends the reach of RAG for analytical and comparison tasks.

A variant called *step-back prompting* first asks the LLM to identify the *principle* behind the question (e.g. "What determines plan pricing?") and retrieves against that abstract form — useful when the user's phrasing is too specific for the index.

:::why-prod
In production, users do not write queries that resemble documentation. HyDE bridges the style gap without any reindexing. Decomposition prevents single-turn failures on complex questions — which are exactly the questions that land on support desks and kill NPS scores.
:::

## Comparing the two techniques

:::table {title="HyDE vs. Decomposition at a glance"}
| | HyDE | Query Decomposition |
|---|---|---|
| Best for | Style/vocabulary gap between query and docs | Complex, multi-part or comparison questions |
| Extra LLM calls | 1 (generate hypothesis) | 1 (decompose) + N retrievals |
| Latency hit | Low–medium | Medium–high |
| Risk | Hallucinated hypothesis drifts retrieval | Sub-queries may overlap, bloating context |
| When to skip | Real-time voice / ultra-low-latency | Simple factual lookups |
:::

```python {title="HyDE + decomposition with LangChain" run=false}
# pip install langchain langchain-openai chromadb
# For a local/free model swap OpenAI for Ollama:
#   from langchain_community.llms import Ollama
#   llm = Ollama(model="llama3")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# ── HyDE ──────────────────────────────────────────────────────────────────
hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short knowledge-base paragraph that directly answers: {question}"
)

def hyde_retrieve(question: str, retriever) -> list:
    """Generate a hypothetical answer, embed it, then retrieve."""
    hypothesis = (hyde_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    )
    # Retriever embeds the hypothesis — not the original question
    return retriever.invoke(hypothesis)


# ── Query Decomposition ───────────────────────────────────────────────────
decompose_prompt = ChatPromptTemplate.from_template(
    "Break this question into up to 3 simple, independent sub-questions. "
    "Return ONLY a Python list of strings.\n\nQuestion: {question}"
)

def decompose_retrieve(question: str, retriever) -> list:
    """Decompose, retrieve per sub-query, deduplicate by doc id."""
    import ast
    raw = (decompose_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    )
    sub_questions = ast.literal_eval(raw)  # e.g. ["What is SSO?", "What audit logs…"]

    seen, docs = set(), []
    for sq in sub_questions:
        for doc in retriever.invoke(sq):
            key = doc.metadata.get("source", doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                docs.append(doc)
    return docs
```

:::gotcha
HyDE can backfire when the LLM hallucinates a plausible-but-wrong hypothesis. If your domain is highly specialised (medical dosages, legal clauses), the hypothetical paragraph may land in the wrong part of the embedding space entirely and retrieve irrelevant chunks with high confidence. Add a fallback: always also retrieve against the raw query and union the results (reciprocal rank fusion works well here).
:::

:::war-story {title="The customer support bot that confidently retrieved nothing"}
A B2B SaaS team shipped a support bot over their 800-page help centre. Average retrieval precision looked fine in offline evals. In production, enterprise customers asked questions like "what's the right architecture for multi-tenant isolation?" — abstract, strategic, nothing like the step-by-step how-to articles in the index. The bot returned "I couldn't find relevant information" on 40% of premium-tier tickets. Root cause: the query embeddings were clustering in abstract-question space; the docs were clustered in tutorial/procedure space. They added HyDE for the support bot and step-back decomposition for the strategic questions. Retrieval recall on those tickets went from 60% to 91% in A/B testing. No reindexing, no new data — just smarter query handling.
:::

:::interview-line
"Query transformation fixes the vocabulary gap without touching the index — HyDE retrieves in answer-space, decomposition handles compound questions that no single chunk can satisfy."
:::

:::qa {q="What problem does HyDE solve that naive RAG cannot?"}
Naive RAG embeds the user's question and searches for similar text. Questions and answers are written differently, so they sit in different neighbourhoods of embedding space. HyDE generates a hypothetical answer first and embeds that, so retrieval happens in answer-space where the real documents live. It costs one extra LLM call but requires zero changes to the index or chunking strategy.
:::

:::qa {q="When would you choose query decomposition over HyDE?"}
Choose decomposition when the user's question is genuinely multi-part — comparisons, checklists, questions that span multiple topics. No single retrieved chunk can satisfy them, so splitting into atomic sub-queries and merging the results gives the LLM the full picture it needs to answer correctly. HyDE helps when the *style* is the problem; decomposition helps when the *scope* is the problem.
:::

:::drill {type="mcq" q="A user asks 'What are the differences between our Starter, Pro, and Enterprise plans regarding storage limits, SSO support, and API rate limits?' — which technique fits best?"}
- [ ] HyDE, because the vocabulary gap is large
- [ ] Step-back prompting to abstract to 'pricing principles'
- [x] Query decomposition into multiple focused sub-questions
- [ ] Reranking with BM25 fusion
:::

:::drill {type="mcq" q="HyDE is most likely to degrade retrieval quality when:"}
- [ ] The user's query is very short (under 5 words)
- [x] The LLM generates a confident but factually wrong hypothetical paragraph
- [ ] The vector database uses cosine similarity instead of dot product
- [ ] The index has more than 1 million documents
:::

:::key-takeaway
Query transformation is the cheapest retrieval upgrade that requires no reindexing: HyDE bridges the style gap between questions and answers; decomposition handles compound questions that a single chunk can never satisfy. Add them before you consider rebuilding your index.
:::
