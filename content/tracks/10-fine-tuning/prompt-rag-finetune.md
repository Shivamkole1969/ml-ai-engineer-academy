---
id: prompt-rag-finetune
track: 10-fine-tuning
title: "When to prompt vs RAG vs fine-tune (decision framework)"
badge: HOT
minutes: 9
prereqs: []
tags: [prompting, rag, fine-tuning, llm, decision-framework, peft, genai]
xp: 60
hot2026: true
---

Your PM walks over on a Tuesday morning with a Slack screenshot and a big smile: "We should fine-tune GPT on our support tickets — make it smarter about our product!"

You nod. Inside your head, a quiet alarm goes off.

Because you've seen this movie. Three months of labeling data. A ₹2L GPU bill. A model that now writes in the same *style* as your support agents — but still confidently tells customers the wrong refund policy. The original problem? Still there.

Before you open a training notebook, you need a decision framework. There are three tools on the table. They are not interchangeable. Picking the wrong one wastes months.

## The three tools

**Prompting** is instruction. You tell the model what to do and how to behave — via a system prompt, few-shot examples, or a carefully structured user message. The model's knowledge stays exactly as it is. You're just steering it.

**RAG (Retrieval-Augmented Generation)** is memory. You fetch relevant facts from your own documents at query time and hand them to the model as context. The model reads them and answers. Its weights never change. New facts? Update the document store. Done.

**Fine-tuning** is surgery. You take a pre-trained model and keep training it on your data — adjusting the weights so the model itself changes. It learns new style, new domain behavior, or new task patterns. This is the heavy option.

:::why-prod
In production, the wrong choice multiplies cost and fragility. Prompting a tiny model for a complex medical task fails quietly — at 3 am when a patient asks a dosage question. Fine-tuning when RAG would have worked burns budget and creates a model you now own and must maintain forever.
:::

## The decision framework

Ask these questions in order. Stop at the first "yes."

**1. Does the base model already know the domain?**
→ Yes + you just need format/tone control → **Prompt engineering first.**

**2. Does the problem require access to specific, up-to-date, or proprietary documents?**
→ Yes → **RAG.** The model cannot memorize a document it never saw at training time. Fine-tuning won't help here either — it bakes in a snapshot, not a live index.

**3. Is prompting too unreliable, too expensive at scale, or behaviorally impossible to guide with instructions alone?**
→ Yes → **Fine-tune.**

That third bucket covers things like:
- Needing a 7B model to match GPT-4 on a narrow task (cost/latency savings)
- Consistent structured output that few-shot examples can't stabilize
- Proprietary communication style that's hard to describe in a prompt
- Removing unsafe behaviors across millions of calls without a long system prompt tax

:::table {title="Quick decision matrix"}
| Situation | Best tool | Why |
|---|---|---|
| Wrong tone, poor formatting | Prompting | Model knows the domain; just guide output |
| Needs your product docs / policies | RAG | Documents change; weights shouldn't |
| Needs real-time data (news, prices) | RAG | No fine-tune can track live info |
| Smaller model, narrow task, scale | Fine-tune | Bake task into weights; drop big model |
| Consistent style across millions of calls | Fine-tune | System prompt tokens add up fast |
| Safety/refusal behavior changes | Fine-tune + alignment | Can't reliably prompt-engineer safety |
| New factual knowledge (e.g., your internal wiki) | RAG, not fine-tune | Fine-tune memorizes poorly; RAG retrieves accurately |
:::

## Why fine-tune does NOT teach new facts

This is the most expensive misconception in the field.

Fine-tuning adjusts how a model *behaves* — its style, its task patterns, its reliability on a narrow output format. It does not reliably inject new factual knowledge into weights. Researchers have shown that models fine-tuned on new facts still hallucinate those facts at high rates. Knowledge needs a retrieval system, not a training loop.

If you want the model to know your company's Q4 pricing, put it in a RAG index. Do not fine-tune it in.

```python {title="Token cost: context-stuffing vs RAG vs fine-tune" run=false}
# Illustrates why token strategy matters at scale
# Run locally: pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def token_count(text: str) -> int:
    return len(enc.encode(text))

# Your company's HR policy document (~2 000 words)
policy_text = "... (full 2 000-word HR policy) ..."
user_query  = "Can I expense a standing desk?"

# Option A: stuff the full doc into every single prompt
stuffed_tokens = token_count(policy_text + user_query)
print(f"Stuffed-context tokens/call : ~{stuffed_tokens}")   # ≈ 2 730

# Option B: RAG — retrieve only the relevant paragraph
retrieved_chunk = (
    "Employees may expense ergonomic equipment up to ₹15 000 per financial year "
    "with manager approval. Receipts must be submitted within 30 days."
)
rag_tokens = token_count(retrieved_chunk + user_query)
print(f"RAG tokens/call             : ~{rag_tokens}")       # ≈ 55

# Option C: fine-tune the model on HR tasks
# → 0 extra context tokens; behavior baked into weights
# → upfront cost, but amortised over millions of calls
# → still won't reliably recall the exact ₹15 000 figure — use RAG for facts!
print("Fine-tune inference tokens  : ~15  (query only)")

# At 1 million calls/month:
cost_per_token_inr = 0.0006  # rough gpt-3.5-turbo equiv in ₹
saving = (stuffed_tokens - rag_tokens) * 1_000_000 * cost_per_token_inr
print(f"\nRAG saves ~₹{saving:,.0f}/month vs full context stuffing")
# → ₹1,60,500/month saved — before you even think about fine-tuning
```

:::gotcha
Fine-tuning on your documents to "teach it the facts" is the #1 mistake teams make. The model learns to *sound like* it knows your data — but will confidently hallucinate specifics. For factual recall, always use RAG. Use fine-tuning for behavior, style, and task adaptation.
:::

:::war-story {title="The ₹8L fine-tune that needed a vector DB"}
A fintech startup in Pune spent three months fine-tuning Llama 2 on their 500-page loan product manual. The goal: answer customer questions accurately. Launch day, the model answered 70% of queries correctly — worse than their old keyword search. The problem? Loan terms change every quarter. The fine-tuned weights were already stale. A two-week RAG implementation on the same docs hit 91% accuracy and cost a fraction of the price. The CTO's words in the postmortem: "We solved the wrong problem beautifully."
:::

## When fine-tuning genuinely wins

To be fair, fine-tuning has real superpowers:

- **Cost at scale.** A fine-tuned Mistral 7B can match GPT-4 on a narrow task (e.g., SQL generation, code review comments) at 50x lower cost per call.
- **Latency.** Smaller fine-tuned model = faster response = better UX.
- **Behavioral reliability.** Structured JSON output, specific refusal patterns, or domain-specific reasoning that prompting can't hold steady across millions of calls.
- **No context window tax.** A long system prompt on every call burns tokens and money. Fine-tuning bakes the behavior into the weights; the system prompt shrinks or disappears.

:::interview-line
"I always ask: is this a knowledge problem or a behavior problem? Knowledge goes in a retrieval index. Behavior goes into fine-tuning. Conflating the two is how teams burn three months and a GPU budget."
:::

:::qa {q="When would you choose RAG over fine-tuning?"}
RAG wins whenever the problem is about accessing specific, current, or proprietary information the base model doesn't have. Documents change — weights don't. RAG lets you update a vector index instead of retraining. It's also faster to ship: no data labeling, no training loop, no model maintenance. Fine-tuning is for shaping behavior, not injecting knowledge.
:::

:::qa {q="Can you fine-tune and use RAG together?"}
Absolutely — and it's often the right production architecture. Fine-tune the model for consistent output format, domain tone, and task reliability. Then wrap it with RAG so it can reference live documents at query time. The fine-tune handles *how* it answers; the retrieval handles *what facts* it uses. Many production LLM services do exactly this.
:::

:::qa {q="How do you convince a PM that fine-tuning isn't always the answer?"}
I frame it as risk and timeline. Fine-tuning takes weeks of data collection, labeling, training, and evaluation — and may not fix a knowledge gap. I show them a RAG prototype in a day or two and let the demo speak. If RAG solves 80% of the problem in 5% of the time, the argument is over. Fine-tuning earns its place when we've confirmed the problem is behavioral, not informational.
:::

:::drill {type="mcq" q="Your e-commerce client wants the LLM to answer 'Is product X in stock?' accurately in real time. Which approach fits best?"}
- [ ] Fine-tune the model on historical inventory data
- [x] RAG — retrieve live inventory records at query time
- [ ] Prompt-engineer a detailed system prompt about stock logic
- [ ] Increase the model's context window
:::

:::drill {type="mcq" q="A team fine-tunes a model specifically so it reliably returns JSON with a fixed schema. This is a valid use of fine-tuning because:"}
- [ ] Fine-tuning is the only way to make a model output JSON
- [ ] The model will learn new factual knowledge about the domain
- [x] Consistent output format is a behavioral pattern that fine-tuning stabilizes better than prompting at scale
- [ ] It avoids the need for RAG entirely
:::

:::drill {type="mcq" q="Which signal most strongly suggests fine-tuning OVER prompting?"}
- [ ] The model doesn't know your company's pricing
- [ ] You want the model to be polite and concise
- [ ] The model needs to cite your internal documents
- [x] A 70B model is too slow and expensive; you need a 7B model to match its performance on your specific task
:::

:::key-takeaway
Knowledge gaps belong in a RAG index; behavior gaps belong in fine-tuning. When in doubt, build the RAG prototype first — it ships in days, not months, and it's almost always cheaper to maintain.
:::
