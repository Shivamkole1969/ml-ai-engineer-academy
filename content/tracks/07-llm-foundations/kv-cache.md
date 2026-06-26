---
id: kv-cache
track: 07-llm-foundations
title: "KV Cache — the real inference memory bill"
badge: HOT
minutes: 11
prereqs: [attention-context]
tags: [inference, memory, gqa, paged-attention, kv-cache]
xp: 60
hot2026: true
---

Picture this. You load a 7B model onto a shiny 80GB GPU. The weights take ~14GB. You think:
*"Sweet, 66GB free — I'll serve a hundred users at once."* You flip it on... and at 30 users,
**boom — out of memory.** The weights never changed. So what ate the other 50GB?

Meet the **KV cache** — the quiet memory hog that actually decides how many people your model
can talk to at the same time.

## The one-sentence intuition

When an LLM generates text, it produces tokens **one at a time**. To produce token #500, it
needs to "look back" at tokens #1–499 (that's attention). Recomputing all of that every step
would be insanely wasteful — so the model **caches** the per-token Key and Value vectors and
reuses them.

That cache is the KV cache. And here's the kicker:

:::why-prod
Weights are a **fixed** cost — load once, done. The KV cache is a **growing** cost: it scales
with sequence length **and** with the number of concurrent requests. On a busy server the cache,
not the weights, is what runs you out of memory first. It is the thing that gates throughput.
:::

## Let's put a number on it

Think of the cache like a parking garage. Every token that's been generated parks **two cars**
(one Key, one Value) on **every floor** (every layer), in **every spot** (every attention head).
Long conversations and many users fill the garage fast.

The formula:

```python {title="Estimate KV cache bytes" run=false}
def kv_bytes(layers, heads, head_dim, seq, batch, dtype_bytes=2):
    # 2 = one Key + one Value per token
    return 2 * layers * heads * head_dim * seq * batch * dtype_bytes

# A 7B model, 8k context, 16 users at once, fp16:
gb = kv_bytes(layers=32, heads=32, head_dim=128, seq=8192, batch=16) / 1e9
print(round(gb, 1), "GB just for the cache")   # ~17 GB — more than the weights!
```

Seventeen gigs. For the *cache*. That's where your 50GB went — pile on more users and longer
chats and it balloons past the weights. Play with it yourself:

:::widget {name="kvcache"}
:::

:::table {title="The three big KV optimizations"}
| Idea | What it does | The tradeoff |
|---|---|---|
| **MQA / GQA** | Share K/V across heads (or groups of heads) instead of one set per head | Huge cache cut (e.g. 8×), tiny quality cost — almost free |
| **PagedAttention** (vLLM) | Store the cache in small fixed "pages" like OS virtual memory | Near-zero wasted memory; enables way more concurrency |
| **KV quantization** | Store K/V in int8/int4 instead of fp16 | Half/quarter the cache; re-check quality, don't trust perplexity alone |
| **Prefix caching** | Reuse the cache for a shared prompt prefix (e.g. a long system prompt) | Big win when many requests share the same opening |
:::

:::gotcha
After KV quantization, **don't trust perplexity alone**. It can look fine while your real task
(say, structured extraction) quietly degrades. Re-run your *actual* eval set, not a proxy.
:::

:::war-story {title="The 2am pager that said OOM, not slow"}
A team launched a support chatbot. Load tests at short prompts looked perfect. Launch day, real
users pasted in giant email threads (long sequences) and the concurrency spiked. The server
started OOM-ing — not because the model was big, but because long-sequence × high-concurrency
made the KV cache explode. The fix wasn't a bigger model or more weights; it was **GQA +
PagedAttention + a max-sequence cap**. Same model, 5× the users.
:::

:::interview-line
"The KV cache, not the weights, gates concurrency — that's exactly what GQA, PagedAttention,
and KV-quant are all attacking."
:::

:::qa {q="Why does the KV cache, not the weights, often limit throughput?"}
Weights load once and stay fixed. The KV cache grows with both sequence length and the number of
concurrent requests, so under real traffic it's the term that blows up first and caps how many
requests you can hold in memory simultaneously.
:::

:::qa {q="What's the cheapest big win for shrinking the KV cache?"}
GQA (grouped-query attention) — sharing K/V across groups of heads cuts the cache several-fold
with almost no quality loss. PagedAttention then removes the wasted-space overhead on top.
:::

:::drill {type="mcq" q="You double the context length and keep everything else the same. What happens to the KV cache size?"}
- [ ] Stays the same
- [x] Roughly doubles (it's linear in sequence length)
- [ ] Quadruples
- [ ] Halves
:::

:::drill {type="mcq" q="Which technique reuses the cache for a shared system prompt across many requests?"}
- [ ] KV quantization
- [x] Prefix caching
- [ ] Gradient checkpointing
:::

:::key-takeaway
KV cache ≈ `2 · layers · heads · head_dim · seq · batch · dtype`. It grows with context **and**
concurrency, so it — not the weights — is the long-context, high-traffic bottleneck. Attack it
with GQA, PagedAttention, KV-quant, and prefix caching.
:::
