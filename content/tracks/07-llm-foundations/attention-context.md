---
id: attention-context
track: 07-llm-foundations
title: "Attention & context limits — the O(n²) wall"
badge: HOT
minutes: 10
prereqs: []
tags: [attention, transformers, context-window, scaling, inference, llm]
xp: 60
hot2026: true
---

It's 11pm. You've just shipped a "chat with your documents" feature. Works beautifully on
your test PDF — a five-pager. Next morning, a beta user uploads a 120-page contract.
Latency jumps from 2s to 45s. Then another user uploads one more, and the API returns a
cryptic 400 error. Your manager pings Slack.

The root cause isn't the model, the network, or your infra. It's *attention* — and the
quadratic wall it hits the moment you start stacking tokens.

## What attention actually does

Attention is the core idea of the Transformer: every token in a sequence "looks at"
every other token to decide what's relevant to it. Token 300 wants to know whether
token 12 is related. Token 1 wants to know about token 800. Everyone looks at everyone.

That's the power. And it's also the cost.

For a sequence of `n` tokens, each of the `n` tokens computes a score against all `n`
tokens. That's `n × n` = **O(n²) operations** — for *every* layer. Double the context,
quadruple the attention cost. Triple it, nine times the cost.

The actual math is:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

`Q Kᵀ` is an `n × n` matrix. Storing it takes `n²` memory. Computing it takes `n²` FLOPs.
You can't escape it in standard full attention.

```python {title="Watch how fast O(n²) blows up" run=false}
# Run locally with: python attention_cost.py
# No GPU needed — this is pure arithmetic to illustrate scaling

def attention_ops(seq_len: int, d_model: int = 4096) -> dict:
    """Approximate attention FLOPs for a single layer (forward pass)."""
    qk_ops = seq_len * seq_len * d_model      # Q·Kᵀ dot products
    av_ops = seq_len * seq_len * d_model      # softmax(QKᵀ)·V
    total = qk_ops + av_ops
    return {"seq_len": seq_len, "attention_ops": f"{total / 1e9:.1f}B"}

for n in [512, 2_048, 8_192, 32_768, 128_000]:
    print(attention_ops(n))

# 512    ->   2.1B ops
# 2 048  ->  33.6B ops
# 8 192  -> 536.9B ops   # 16x more tokens = 256x more ops
# 32 768 ->   8.6T ops
# 128 000 -> ~131T ops   # one layer, one forward pass
```

That 256× jump for a 16× token increase is the wall. It's not slow infrastructure — it's
fundamental math.

:::why-prod
Context length is the single biggest lever on inference cost and latency for long-input workloads.
A request with 32K tokens doesn't cost twice a 16K request — it costs four times as much in
attention alone, plus memory. This shows up directly in cloud API pricing, which charges per
**input token**, not per request.
:::

## The context window: hard limit, not soft

Every model has a maximum sequence length it was trained on. Exceed it and you get a hard
error — not degraded output, a crash. The model has no mechanism to handle position
embeddings beyond its training horizon.

:::table {title="Context windows across popular models (mid-2025)"}
| Model | Context window | Rough text equivalent |
|---|---|---|
| Llama 3.1 (8B / 70B) | 128K tokens | ~200 pages |
| Mistral Large 2 | 128K tokens | ~200 pages |
| GPT-4o | 128K tokens | ~200 pages |
| Claude 3.5 Sonnet | 200K tokens | ~320 pages |
| Gemini 1.5 Pro | 1M tokens | ~1,600 pages |
| Gemini 2.5 Pro | 1M tokens | ~1,600 pages |
:::

Bigger looks better, but the O(n²) cost means a 1M-token window isn't "free." Providers
absorb it via custom attention kernels (FlashAttention, ring attention) that reduce
*memory* from O(n²) to O(n) — but the FLOPs don't disappear.

## The quality trap hiding inside the limit

Here's the subtler problem. Even when you're *within* the context window, quality isn't
uniform. Research shows models attend more reliably to content near the **beginning** and
the **end** of a long context. Material buried in the middle gets under-attended — models
can literally miss it.

This is called the **"lost in the middle"** problem. Put the most critical information
at the edges when you control the prompt structure. Don't assume 200K tokens means 200K
tokens of equal comprehension.

:::gotcha
Extending context window by fine-tuning (or rope-scaling tricks) doesn't fix "lost in
the middle." Your model may technically accept the length but silently ignore the middle
40%. Always validate on real tasks with content planted at varied positions — don't just
confirm it doesn't crash.
:::

:::war-story {title="The chatbot that forgot the instructions it was given"}
A team built an enterprise support bot with a huge system prompt — product docs, tone
guidelines, escalation rules — around 20K tokens. It worked fine in testing. In
production, users had long conversations and the cumulative context (system + history +
user messages) crept toward 60K tokens. The bot started ignoring the escalation rules
buried in the middle of the system prompt. No error. No warning. Just quietly wrong
behavior. The fix: move critical rules to the **very top** of the system prompt *and*
to a trailing reminder before the user turn. Content position in context is a
production variable, not an afterthought.
:::

## What you actually do about it

Three levers, used in combination:

1. **Truncate aggressively.** Don't pass what the model doesn't need. Sliding window
   over chat history. Top-k retrieved chunks, not the whole corpus.

2. **Position critical content deliberately.** Instructions and must-respect rules at
   the top. Retrieved facts just before the user question. Pad in the middle if you
   must pad.

3. **Use a model sized for your task.** A 32K context at full quality often outperforms
   a 128K context where you're banking on the middle. Match the context to actual
   content density.

The KV cache lesson covers what happens to *memory* as context grows — this lesson is
about the compute and quality side.

:::interview-line
"Attention is O(n²) in both compute and memory — doubling context quadruples cost — so
context length is an engineering decision, not just a capability checkbox."
:::

:::qa {q="Why is the Transformer's self-attention O(n²)?"}
Each of the n tokens computes a dot-product score against every other token to build
the attention weight matrix, which is n × n. Both storing this matrix and computing
it scale quadratically with sequence length — per layer, per forward pass.
:::

:::qa {q="A product manager asks you to 'just use the 1M-token model for everything.' What's your pushback?"}
Long context windows don't come free: attention cost scales quadratically, so a 1M
request is thousands of times more expensive than a 1K one. Beyond cost, the "lost in
the middle" phenomenon means quality degrades for content far from the endpoints. The
right approach is to retrieve only the relevant chunks and keep context tight.
:::

:::qa {q="What is the 'lost in the middle' problem and when does it bite you?"}
Models attend more reliably to tokens near the beginning and end of a long context;
material in the middle is under-attended and can be functionally ignored. It bites
when you embed critical instructions or retrieved facts deep in a long prompt and the
model silently misses them — no error, just wrong output.
:::

:::drill {type="mcq" q="You double the sequence length fed into a Transformer. How does the attention computation cost change, approximately?"}
- [ ] It stays the same (attention is O(n))
- [ ] It doubles
- [x] It quadruples (attention is O(n²))
- [ ] It increases by 1.5×
:::

:::drill {type="mcq" q="Where should you place the most critical instructions in a long prompt to maximise the chance the model follows them?"}
- [ ] Buried in the middle, so they're surrounded by context
- [ ] Evenly distributed throughout the prompt
- [x] Near the beginning (and optionally repeated near the end)
- [ ] It doesn't matter — all tokens receive equal attention
:::

:::key-takeaway
Self-attention is O(n²): doubling context quadruples cost. Context windows are hard
limits, and quality degrades for content in the middle even within those limits. Treat
context length as a deliberate engineering variable — truncate, retrieve, and position
critical content at the edges.
:::
