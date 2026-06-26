---
id: rag-evaluation
track: 09-genai-rag
title: "RAG evaluation (RAGAS: faithfulness, relevancy, context precision/recall)"
badge: HOT
minutes: 10
prereqs: []
tags: [rag, evaluation, ragas, faithfulness, answer-relevancy, context-precision, context-recall, llm-as-judge, genai]
xp: 60
hot2026: true
---

Your RAG chatbot shipped three weeks ago. The product manager says it "feels good." Your tech lead says "users seem happy." A junior engineer manually spot-checked ten responses and gave it a thumbs-up. Then a customer support ticket arrives: the bot confidently told a user that their subscription auto-renews — but that policy was removed from your docs six months ago. It was still in a stale chunk, the LLM believed it, and now you have a chargeback dispute.

"Feels good" is not a monitoring strategy. You need numbers.

This is exactly the gap RAGAS fills. It turns your RAG pipeline's quality into four measurable scores you can track, alert on, and improve systematically — without a human reading every conversation.

## Why evaluation is a production concern, not a research concern

Every change you make to your RAG pipeline — new chunking strategy, different embedding model, adjusted top-K, updated reranker threshold — changes the answer quality. Without metrics, you're flying blind. With metrics, you ship a change, run an evaluation, and either merge it or revert it. That's a real engineering workflow.

:::why-prod
RAG quality degrades silently. Your index goes stale, your chunk boundaries shift, or a new document contradicts an old one. None of these produce an error or an alert. RAGAS metrics, run on a test set after every deployment, are your canary in the coal mine.
:::

## The four RAGAS metrics

RAGAS gives you four numbers, each measuring a different part of the pipeline. Learn them as two pairs.

**Pair 1 — Did the answer stay honest?**

**Faithfulness** asks: does every claim in the generated answer actually appear in the retrieved context? If the model says "refunds take 5 days" but the retrieved chunks say "7 business days," faithfulness is low. This is your hallucination detector. Score is 0–1; aim for above 0.9 in production.

**Answer Relevancy** asks: does the answer actually address the question? A model can be perfectly faithful to its context (no hallucinations) and still give an answer that wanders off-topic. RAGAS catches this by reverse-engineering: it generates several questions from the answer and checks how similar they are to the original question. If the generated questions don't resemble what the user asked, the answer went sideways.

**Pair 2 — Did retrieval do its job?**

**Context Precision** asks: of all the chunks you retrieved, how many were actually useful? If you retrieve 10 chunks and 9 of them are irrelevant noise, context precision is 0.1. Noisy context confuses the LLM — it may blend irrelevant facts into the answer even when good chunks are present. This metric tells you your retriever (or reranker) is polluting the prompt.

**Context Recall** asks: did your retrieval step surface *all* the information needed to answer the question? This is the miss-rate metric. If the correct answer requires three facts and your retrieved context only contains two of them, recall is low. The model will either hallucinate the missing fact or give an incomplete answer. Context recall uses a reference (ground-truth) answer to check whether everything needed was retrieved.

:::table {title="RAGAS metric cheat-sheet"}
| Metric | What it measures | Low score means | Component blamed |
|---|---|---|---|
| Faithfulness | Answer stays within retrieved context | LLM hallucinating beyond context | Generator (LLM) |
| Answer Relevancy | Answer addresses the actual question | Answer wanders off-topic | Generator / prompt |
| Context Precision | Retrieved chunks are actually useful | Retriever returning noise | Retriever / reranker |
| Context Recall | All needed facts were retrieved | Important chunks were missed | Retriever / index |
:::

## How RAGAS computes these scores

RAGAS uses an LLM as judge — by default it calls GPT-4o, but you can swap in any OpenAI-compatible model. This means the evaluation itself costs tokens, so run it on a curated test set (50–200 representative questions), not every live query.

For **faithfulness**: RAGAS breaks the answer into atomic claims ("refunds take 7 days", "approval is required"), then asks the LLM to verify each claim against the retrieved context. The score is the fraction of claims that are supported.

For **answer relevancy**: RAGAS generates N synthetic questions from the answer and embeds them. The score is the average cosine similarity between those synthetic questions and the original question.

For **context precision** and **context recall**: RAGAS checks each retrieved chunk for relevance (precision) and checks whether the ground-truth answer's key statements are covered by the retrieved chunks (recall).

```python {title="Running RAGAS on a small test set" run=false}
# pip install ragas datasets openai
# Works with any OpenAI-compatible endpoint — swap base_url for Groq, Together, or local Ollama

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Your test set: questions + your RAG system's actual outputs + reference answers
test_data = {
    "question": [
        "What is the refund window for premium subscribers?",
        "Can I use the API on the free plan?",
    ],
    "answer": [
        # These come from your RAG pipeline's live output
        "Premium subscribers can request a refund within 14 days of purchase.",
        "The free plan does not include API access.",
    ],
    "contexts": [
        # The actual chunks your RAG system retrieved for each question
        ["Premium plan: 14-day refund window. Standard plan: 7-day refund window."],
        ["API access is available on Pro and Enterprise plans only."],
    ],
    "ground_truth": [
        # Reference answers (written by a human) — needed for context_recall
        "Premium subscribers have a 14-day refund window.",
        "The free plan does not include API access; it is available on paid plans.",
    ],
}

dataset = Dataset.from_dict(test_data)

# To use a different LLM judge, set OPENAI_API_KEY and optionally OPENAI_API_BASE
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)
# Sample output:
# {'faithfulness': 0.95, 'answer_relevancy': 0.88,
#  'context_precision': 0.75, 'context_recall': 0.90}

# Convert to a pandas DataFrame to spot which questions are failing
df = results.to_pandas()
print(df[["question", "faithfulness", "context_precision"]].sort_values("faithfulness"))
```

## Reading the scores together

The four scores form a diagnostic map. Low faithfulness with high context recall means the model has the right information but is still inventing things — check your system prompt and temperature. Low context precision with high recall means you're retrieving too many chunks and some are noisy — tighten your reranker threshold. Low context recall means you're simply not finding the right chunks — revisit chunking strategy or embedding model.

:::gotcha
RAGAS requires a **ground-truth answer** for context recall. If you don't have human-written reference answers, you can skip context recall — but then you're blind to the miss-rate of your retriever. A practical shortcut: use your LLM to generate ground-truth answers from the *full document* (not retrieved chunks), then have a human spot-check 20% of them. This bootstraps a test set without manual annotation of every example.
:::

:::war-story {title="The noisy retriever that nobody caught"}
A Pune-based HR-tech team built a RAG chatbot over their policy docs. Faithfulness was 0.93 — the LLM was staying faithful to what it retrieved. But users kept complaining the answers felt "confused." When they finally ran RAGAS with context precision, they saw 0.31. Seventy percent of the retrieved chunks were loosely related but wrong documents — a leave-policy question was surfacing performance-review chunks alongside the right chunk. The LLM tried to reconcile conflicting information and produced hedged, confusing answers. Raising the reranker score threshold from 0.3 to 0.55 pushed context precision to 0.84. User complaints dropped the following week.
:::

:::interview-line
"RAGAS gives you four independently interpretable scores — faithfulness catches hallucination relative to context, answer relevancy checks the answer stayed on-topic, context precision measures retrieval noise, and context recall measures retrieval miss-rate. Together they tell you *which stage* to fix, not just that something is broken."
:::

:::qa {q="What is the difference between faithfulness and answer relevancy in RAGAS?"}
Faithfulness measures whether the answer's claims are grounded in the retrieved context — it's your hallucination guard. Answer relevancy measures whether the answer actually addresses what the user asked — it's your topic drift guard. A model can score high on faithfulness (everything it says is in the context) but low on answer relevancy (it answered a different question than the one asked). You need both.
:::

:::qa {q="Why does context precision matter even if context recall is high?"}
High context recall means you retrieved all the necessary information. But if context precision is low, you also retrieved a lot of irrelevant noise. That noise goes into the LLM's prompt, and LLMs are sensitive to what's in context — they may blend facts from irrelevant chunks into the answer, reducing faithfulness. Good retrieval means surfacing the right chunks *and only* the right chunks. Precision and recall are both required for a clean prompt.
:::

:::drill {type="mcq" q="Your RAG system's faithfulness score is 0.4 but context recall is 0.95. What is the most likely root cause?"}
- [ ] The retriever is missing important chunks
- [ ] The retrieved chunks contain too much irrelevant noise
- [x] The LLM is generating claims not supported by the retrieved context
- [ ] The ground-truth answers in the test set are incorrect
:::

:::drill {type="mcq" q="Which RAGAS metric requires a human-written ground-truth answer to compute?"}
- [ ] Faithfulness
- [ ] Answer Relevancy
- [x] Context Recall
- [ ] Context Precision
:::

:::key-takeaway
RAGAS turns RAG quality into four actionable numbers: faithfulness and answer relevancy diagnose the generator, while context precision and context recall diagnose the retriever. Run them on a fixed test set after every pipeline change — they tell you which specific stage broke, not just that something got worse.
:::
