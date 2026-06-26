---
id: transformer-bridge
track: t2-dl-transformers
title: "The Transformer: self-attention, Q/K/V, multi-head"
badge: HOT
minutes: 9
prereqs: []
tags: [transformers, self-attention, qkv, multi-head, deep-learning, llm]
xp: 60
hot2026: true
---

Imagine you are reviewing a 40-page legal contract. On page 38, a clause says "subject to the exceptions defined in Section 2.1." A good lawyer flips back instantly and reads both sections together. Now imagine a reader who can only see the sentence they are currently reading — they would miss that connection entirely.

That is the exact problem that broke RNNs (recurrent neural networks) and LSTMs on long text. They read left-to-right, one token at a time, compressing everything into a fixed-size memory that quietly forgot things. The Transformer threw that approach away. Every token attends directly to every other token, no matter the distance. That single design decision is why you are hearing about GPT, Claude, Gemini, and every large language model today.

## Why RNNs Couldn't Scale

RNNs pass a "hidden state" from one word to the next — a small vector that tries to carry all previous context. By the time you reach word 500, the hidden state has been overwritten hundreds of times. Long-range dependencies (like that page-38 clause referring to page-2) simply vanish.

Transformers removed the sequential bottleneck. Every position in the sequence processes in parallel, and every pair of positions can interact directly. This is what made training on billions of tokens practical.

## Self-Attention: The Matching Game

Self-attention asks one question for every token: *"Which other tokens in this sequence are most relevant to me right now?"*

Picture a search engine inside the model. Each token broadcasts a query ("what am I looking for?"), examines keys from all other tokens ("what do they contain?"), and collects values ("what useful information do they carry?").

The attention score between token A and token B is just a dot product — a measure of similarity. High score means A should borrow a lot of meaning from B.

## Q, K, V: The Mechanics

Each token embedding is projected into three separate vectors:

- **Query (Q)** — What this token is searching for.
- **Key (K)** — What this token advertises about itself.
- **Value (V)** — The actual content this token contributes if selected.

The attention scores are computed as:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

`d_k` is the dimension of the key vectors. Dividing by `√d_k` prevents the dot products from growing so large that softmax saturates and gradients vanish (a subtle but fatal bug in early prototypes).

:::why-prod
Every major production LLM — GPT, Claude, Llama, Mistral — runs on this same formula. Understanding Q/K/V is how you debug attention maps, tune context windows, interpret KV cache costs, and make informed decisions about model selection. You will see these letters everywhere.
:::

```python {title="Minimal scaled dot-product attention" run=false}
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: tensors of shape (batch, heads, seq_len, d_k)
    Returns: attended values of shape (batch, heads, seq_len, d_k)
    
    Run locally: pip install torch  (CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)
    """
    d_k = Q.size(-1)                                # e.g. 64
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, seq, seq)
    scores = scores / (d_k ** 0.5)                 # scale to avoid softmax saturation
    weights = F.softmax(scores, dim=-1)             # attention weights sum to 1 per row
    return torch.matmul(weights, V)                 # weighted sum of values

# Quick smoke test — no GPU needed
batch, heads, seq_len, d_k = 2, 4, 16, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)
out = scaled_dot_product_attention(Q, K, V)
print(out.shape)  # torch.Size([2, 4, 16, 64])
```

## Multi-Head Attention: Many Conversations at Once

One attention head learns one type of relationship. But language is layered — "bank" relates to "river" geographically, to "money" semantically, and to "sat" syntactically, all at once.

Multi-head attention runs H independent attention heads in parallel, each with its own Q, K, V projection matrices. The outputs are concatenated and projected back to the model dimension.

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₙ) · Wₒ
```

In GPT-2 (small), there are 12 heads. In GPT-3, 96. Each head specializes spontaneously during training — some track subject-verb agreement, others track coreference ("she" → "the doctor"), others handle positional proximity.

:::table {title="Transformer dimensions across common models"}
| Model | d_model | Heads | d_k per head | Layers |
|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 64 | 12 |
| GPT-2 XL | 1600 | 25 | 64 | 48 |
| Llama 3 8B | 4096 | 32 | 128 | 32 |
| GPT-4 (est.) | ~12 288 | ~96 | ~128 | ~96 |
:::

## Where Self-Attention Lives in the Full Stack

A Transformer block is: Multi-Head Attention → Add & Norm → Feed-Forward Network → Add & Norm. Stack that block N times, add positional encoding at the input (so the model knows token order — attention itself is order-agnostic), and you have the encoder. Decoder-only models (GPT family) mask future tokens during training so the model cannot "cheat" by looking ahead.

The feed-forward sub-layer is surprisingly important — it stores factual associations. Attention decides *where* to look; the FFN decides *what* to retrieve.

:::gotcha
People assume more attention heads always means better performance. In practice, heads beyond a certain threshold become redundant or even noisy — "attention head pruning" research shows you can often delete 30–50% of heads post-training with minimal quality loss. Don't conflate head count with model quality. Also, the `1/√d_k` scaling factor is easy to forget when implementing from scratch — omit it and training will silently diverge because softmax outputs collapse to near-zero gradients.
:::

:::war-story {title="The context cliff at token 3 900"}
A team deployed a summarization service using a 4 096-token model. It worked beautifully in testing, where documents averaged 2 000 tokens. In production, legal filings routinely hit 3 800–4 000 tokens. At token 3 900, the model started producing hallucinated clause numbers — confidently wrong. The bug took two weeks to find. The attention scores near the sequence limit were near-uniform (all tokens got equal weight), meaning the model had effectively lost its ability to discriminate. The fix: chunked summarization with a sliding window. The lesson: know your model's context cliff, and test explicitly at 90%+ of the limit.
:::

:::interview-line
"Self-attention lets every token attend to every other token in O(n²) time — that's both its superpower and its scaling cost, which is why KV caching and attention approximations exist."
:::

:::qa {q="Why do we divide attention scores by √d_k?"}
Dot products grow with the vector dimension: if d_k = 64, a typical dot product is ~64× larger than when d_k = 1. This pushes softmax into regions where gradients are near zero (the function saturates). Dividing by √d_k keeps scores in a stable range and prevents training from stalling.
:::

:::qa {q="What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?"}
Encoder-only models (BERT) see the full sequence bidirectionally — great for classification and embeddings. Decoder-only models (GPT, Llama) mask future tokens and generate left-to-right — the standard for language generation. Encoder-decoder models (T5, BART) use an encoder to process input and a decoder to generate output, useful for translation and summarization where input and output are distinct sequences.
:::

:::qa {q="What does each attention head learn?"}
Heads specialize spontaneously during training — you cannot control what they learn. Empirically, different heads track syntactic roles (subject-verb), coreference links, positional proximity, and semantic similarity. This is why stacking multiple heads (and multiple layers) is so powerful: the model builds a rich, layered representation without explicit supervision.
:::

:::drill {type="mcq" q="Why does scaled dot-product attention divide by √d_k?"}
- [ ] To normalize the value vectors so they sum to one
- [ ] To reduce the number of parameters in the attention layer
- [x] To prevent large dot products from saturating the softmax and killing gradients
- [ ] To match the output dimension to the model's embedding size
:::

:::drill {type="mcq" q="In a decoder-only Transformer (like GPT), what prevents the model from attending to future tokens during training?"}
- [ ] A separate encoder that processes only past tokens
- [ ] Positional embeddings that zero out future positions
- [x] A causal mask that sets future attention scores to −∞ before softmax
- [ ] Gradient clipping applied only to forward-looking attention weights
:::

:::drill {type="mcq" q="A model has d_model=512 and 8 attention heads. What is d_k per head?"}
- [ ] 512
- [ ] 128
- [x] 64
- [ ] 8
:::

:::key-takeaway
Self-attention lets every token talk directly to every other token — Q finds what to look for, K advertises what is there, V carries the content. Multi-head runs this in parallel across H independent subspaces so the model learns many types of relationships at once. This is the core of every production LLM today.
:::
