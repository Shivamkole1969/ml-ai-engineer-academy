---
id: lay-of-land
track: 15-cloud-genai
title: "The lay of the land: managed APIs vs training vs RAG vs agents on cloud"
badge: CORE
minutes: 8
prereqs: []
tags: [cloud, aws, gcp, snowflake, bedrock, vertex-ai, rag, agents, genai]
xp: 45
hot2026: false
---

Your PM drops a Slack message on Monday morning: "Can we add an AI feature by Friday? Something like — customer asks a question, AI answers using our product docs." You have a modest AWS account, a folder of PDFs, and two engineers. No GPUs. No ML team. No ₹50L compute budget.

Four years ago, you would have said no. Today, you have four clean options — and picking the wrong one will cost you two sprints and a painful rewrite. This lesson is the map.

## The four patterns, in plain English

Every GenAI feature on any cloud boils down to one of these four moves:

**1. Managed API call** — You send text to a cloud endpoint, you get text back. The model lives on someone else's GPU. You pay per token. No infra, no ML code, no waiting. This is where 80% of production GenAI features actually live.

**2. Fine-tuning / training** — You take a base model and adapt it with your own labelled data. Useful when the model needs to mimic a style, output a very specific schema, or know jargon that prompting can't fix. Expensive to set up; most teams reach for it too early.

**3. RAG (Retrieval-Augmented Generation)** — You store your documents in a vector index, retrieve the relevant chunks at query time, and shove them into the prompt. The model doesn't "learn" your docs — it reads them fresh each time. Fast to build, great for doc Q&A, product catalogues, internal wikis.

**4. Agents** — You give the LLM a set of tools (search, calculator, database query, API call) and let it decide which to invoke, in what order, to complete a task. Think of it as "the model writes and runs a mini-script on your behalf." Powerful and non-trivial to make reliable.

:::why-prod
Every cloud bill disaster, every "why is this hallucinating?" ticket, and every "we need to rebuild this" sprint traces back to picking the wrong pattern. Getting this mental model right before you touch a console is the single highest-leverage decision in a GenAI project.
:::

## Where each cloud platform fits

The three platforms you'll see in Pune-based jobs and remote-India interviews are AWS, GCP, and Snowflake. Here's where each one plays:

:::table {title="Cloud platform × GenAI pattern"}
| Pattern | AWS | GCP | Snowflake |
|---|---|---|---|
| Managed API | Bedrock (Claude, Llama, Titan, Mistral) | Vertex AI Model Garden + Gemini API | Cortex LLM functions (COMPLETE, CLASSIFY…) |
| Fine-tuning / Training | SageMaker Training Jobs | Vertex AI Training | Cortex Fine-tuning (limited) |
| RAG | Bedrock Knowledge Bases | Vertex RAG Engine | Cortex Search |
| Agents | Bedrock Agents | Vertex Agent Builder | Cortex Analyst (text-to-SQL) |
:::

Read that table as a menu, not a checklist. You rarely need all four patterns in one project.

## Choosing the right pattern: a 60-second decision tree

Start with the cheapest, fastest option and only escalate when you have a real reason:

```python {title="Pattern decision tree (pseudocode)" run=false}
def pick_pattern(problem):
    # Step 1: Can a plain API call + a good system prompt solve it?
    # If yes — use managed API. Ship it. Measure it.
    if works_with_prompt_engineering(problem):
        return "Managed API"  # Bedrock / Vertex / Cortex

    # Step 2: Is the model hallucinating because it lacks your private data?
    # Documents, PDFs, internal wikis, product catalogues → RAG first.
    if knowledge_gap_is_the_issue(problem):
        return "RAG"  # Bedrock KB / Vertex RAG Engine / Cortex Search

    # Step 3: Does the feature need to take actions, call APIs, query a DB?
    # Multi-step reasoning over tools → agents.
    if requires_multi_step_tool_use(problem):
        return "Agents"  # Bedrock Agents / Vertex Agent Builder

    # Step 4: Nothing above worked, AND you have labelled data, AND
    # the problem is a specific style/schema/domain mismatch.
    # Only now does fine-tuning make sense.
    return "Fine-tuning"  # SageMaker / Vertex Training
```

The ordering matters. Most teams jump to fine-tuning at Step 1 and spend two weeks building a training pipeline for a problem that a two-line system prompt would have solved.

:::gotcha
RAG and fine-tuning are not substitutes for each other. RAG gives the model access to information it didn't see during training. Fine-tuning changes how the model behaves — its tone, format, reasoning style. Combining them is valid, but if you try to "fine-tune in your docs," you'll get a model that confidently invents outdated details the moment any doc changes. Docs change; your fine-tuned model does not.
:::

## Free-tier survival guide (preview)

All four patterns are explorable for ₹0 or near-₹0:

- AWS Bedrock: free trial tokens for Claude/Titan via on-demand pricing; Bedrock Knowledge Bases has a small free tier.
- GCP Vertex AI: $300 free credits for new accounts; Gemini API has a free tier with generous RPM.
- Snowflake: 30-day free trial with Cortex functions enabled by default.

The "Cost & free-tier survival guide" lesson in this track will walk through each one in detail. For now, just know: you can build something real without a card charge.

:::interview-line
"I map every GenAI problem to one of four patterns — managed API, RAG, agents, or fine-tuning — before touching the console, because picking the wrong one is the most expensive mistake a team can make."
:::

:::qa {q="When would you choose RAG over fine-tuning?"}
When the model's problem is a knowledge gap — it lacks your private documents, recent data, or domain-specific facts. RAG solves that dynamically at query time. Fine-tuning is for behaviour gaps — tone, output format, or reasoning style. If your docs update frequently, RAG is almost always the right call because you'd have to re-fine-tune every time the data changes.
:::

:::qa {q="A product manager asks for an AI agent that queries a database and emails a summary. Which cloud patterns and services are involved?"}
This is a classic agents pattern: the LLM uses a database query tool and an email-send tool, deciding when to call each. On AWS you'd model this with Bedrock Agents and Lambda action groups. On GCP, Vertex Agent Builder with Function Calling. The underlying model access in both cases is a managed API — so you're combining two patterns: managed API (the LLM call) and agents (the orchestration loop). No training required.
:::

:::drill {type="mcq" q="Your team has 500 internal PDF docs that change every quarter. Users ask questions answered in those docs. The model currently halluccinates. What's the first pattern to try?"}
- [ ] Fine-tune a base model on the 500 PDFs
- [x] Build a RAG pipeline with the PDFs in a vector index
- [ ] Write a very long system prompt listing all doc contents
- [ ] Train a custom model from scratch on Vertex AI
:::

:::drill {type="mcq" q="Which statement about managed API calls on cloud platforms is TRUE?"}
- [ ] You need a GPU instance to run managed API calls
- [ ] Managed APIs require you to upload and host the model weights
- [x] You pay per token and the cloud provider manages all compute
- [ ] Managed APIs only support open-source models like Llama
:::

:::key-takeaway
There are exactly four GenAI patterns on cloud — managed API, RAG, agents, fine-tuning — and they solve different problems. Always start with the simplest one that could work, because escalating prematurely is the most common (and costly) mistake on GenAI projects.
:::
