---
id: lora-vs-rag
track: 07-llm-foundations
title: "LoRA vs fine-tune vs RAG — the decision you make constantly"
badge: HOT
minutes: 10
prereqs: []
tags: [rag, fine-tuning, lora, decision, prompting]
xp: 60
hot2026: true
---

This is the single most common question you'll face in GenAI work — from your manager, in
interviews, in your own head at 11pm: *"The model doesn't do what I want. Do I fine-tune it,
or do I RAG it, or do I just write a better prompt?"*

Pick wrong and you waste weeks. So let's make the decision boringly easy with one question.

## The question that decides everything

> **Is the problem KNOWLEDGE, or is it BEHAVIOR?**

- **Missing knowledge** ("it doesn't know our Q3 policy / this customer / today's prices")
  → that's a **RAG** problem. Give it the facts at runtime.
- **Wrong behavior** ("it won't follow our format / tone / weird output schema no matter how I
  ask") → that's a **fine-tuning** problem. Teach it a new habit.
- **Neither, really** ("it basically can, I just haven't asked clearly") → that's a **prompting**
  problem. Cheapest fix; always try first.

:::table {title="The cheat sheet"}
| Approach | Fixes | Cost / effort | Freshness | Best when |
|---|---|---|---|---|
| **Prompting** | Clarity, format nudges | Minutes | Live | Always your first attempt |
| **RAG** | Missing / changing facts | Medium | **Live** (just update the docs) | Knowledge it never had or that changes |
| **Fine-tune / LoRA** | Style, format, skill, tone | High (data + training) | **Frozen** at train time | Behavior you can't prompt your way to |
:::

:::why-prod
RAG and fine-tuning solve **different** problems — they're not competitors, they're teammates.
The strongest production systems often do **both**: fine-tune the *behavior* (always answer in
this JSON, always cite), and RAG the *knowledge* (here are today's relevant documents).
:::

## A concrete scenario

You're building a financial-research assistant. Two complaints land:

1. *"It doesn't know about the broker note we uploaded this morning."* → **RAG.** The note didn't
   exist at training time and new ones arrive daily. No amount of fine-tuning fixes "yesterday's
   data" — you'd be retraining forever.
2. *"It rambles. We need a strict 3-bullet summary with a citation on every claim."* → **Fine-tune
   (LoRA).** That's a consistent behavior. Show it 500 good examples and it learns the habit.

See how the *same app* needs both, for different reasons? That's the whole insight.

:::gotcha
The classic mistake: trying to **fine-tune in facts**. People fine-tune a model on their docs
hoping it "memorizes" them. It doesn't reliably — it learns *style* far better than *facts*, and
the facts go stale the moment anything changes. Knowledge → RAG. Every time.
:::

## Why LoRA, not full fine-tuning?

Full fine-tuning updates **all** the weights — expensive, and you get a whole new multi-GB model
per task. **LoRA** freezes the base model and trains tiny "adapter" matrices (often <1% of params).
You get most of the quality for a fraction of the cost, and you can hot-swap adapters per task on
the same base. **QLoRA** goes further — 4-bit frozen base — so you can adapt a 7B model on a free
Colab T4. (Deep dive in the Fine-Tuning track.)

:::war-story {title="Six weeks fine-tuning what a retriever fixed in a day"}
A team spent six weeks fine-tuning a model to "know" their product catalog. Every catalog update
meant another training run. They eventually ripped it out and dropped in a simple RAG retriever
over the catalog — instantly fresh, no retraining, and *more* accurate. The fine-tune had been
solving a knowledge problem with a behavior tool. Right tool, right problem.
:::

:::interview-line
"Ask one question: is it a knowledge gap or a behavior gap? Knowledge → RAG, because it stays
fresh. Behavior → fine-tune, usually LoRA. And try a better prompt before either."
:::

:::qa {q="Should you fine-tune a model to teach it your company's latest documents?"}
No — that's a knowledge problem, and fine-tuning teaches style far more reliably than facts, plus
the facts go stale immediately. Use RAG so the knowledge stays live and updatable; reserve
fine-tuning for behavior/format/tone.
:::

:::qa {q="Why reach for LoRA instead of full fine-tuning?"}
LoRA freezes the base and trains tiny adapter matrices (<1% of params): far cheaper, much smaller
artifacts, and you can swap adapters per task on one shared base. QLoRA adds a 4-bit base so it
fits on modest GPUs.
:::

:::drill {type="mcq" q="Your chatbot doesn't know about a policy doc uploaded this morning. Best fix?"}
- [x] RAG — retrieve the doc at runtime (stays fresh)
- [ ] Fine-tune the model on the doc
- [ ] Train a model from scratch
:::

:::drill {type="mcq" q="The model's answers are correct but always too long and unformatted. Best fix?"}
- [ ] RAG
- [x] Fine-tune (e.g. LoRA) on well-formatted examples — it's a behavior gap
- [ ] Buy a bigger model
:::

:::key-takeaway
One question: **knowledge or behavior?** Knowledge → **RAG** (fresh, updatable). Behavior → 
**fine-tune/LoRA** (frozen habit). Prompt first; the best systems combine RAG *and* a light LoRA.
:::
