---
id: batching
track: 08-inference-serving
title: "Batching: static → dynamic → continuous (in-flight)"
badge: HOT
minutes: 10
prereqs: []
tags: [inference, serving, throughput, batching, llm, production]
xp: 60
hot2026: true
---

It's 11 PM on a Tuesday. Your startup just got featured in a popular newsletter. Traffic spiked 10x. Your GPU is crying. Latency is spiking to 40 seconds. Your Slack lights up.

You open the serving logs. Your model is handling **one request at a time**. The GPU — a ₹12 lakh A100 — is sitting at 12% utilization while users rage-quit.

The fix is batching. And understanding *which kind* of batching is the difference between burning money and sailing through the spike.

## What even is batching?

Your GPU is a parallelism machine. It has thousands of CUDA cores that want to do math *simultaneously*. When you feed it a single request, you're using maybe 5–15% of its potential. Feed it 32 requests shaped the same way? Now you're cooking.

**Batching** = grouping multiple inference requests together and running them through the model in one forward pass.

The payoff: same GPU time, many more tokens served. Throughput goes up, cost-per-token goes down.

:::why-prod
A single A100 running unbatched LLM inference might serve 10 req/s. Well-tuned continuous batching on the same card can hit 200+ req/s. That's the difference between a ₹2L/month GPU bill and a ₹40K one.
:::

## Three generations of batching

### 1. Static batching — the blunt instrument

You wait until you have exactly N requests, then you run them together.

Simple. But broken for LLMs. Why? Because LLMs generate tokens **one at a time** until they hit a stop condition. In a batch of 32, one request might finish at 20 tokens; another runs to 800. You have to **pad the short ones** to the length of the longest — wasting memory and compute on zeros.

:::table {title="Static vs Dynamic vs Continuous"}
| Mode | How it groups | LLM padding waste | Works well for |
|---|---|---|---|
| Static batching | Fixed size N, wait for it | Extreme (short seqs padded to max) | Simple CV models, fixed-length tasks |
| Dynamic batching | Flexible size, timeout-triggered | Moderate | Classic NLP, moderate traffic |
| Continuous (in-flight) | Per-step iteration, swap in/out | Near-zero | LLM token generation at scale |
:::

### 2. Dynamic batching — smarter waiting

Instead of waiting for exactly N requests, you wait for **either** N requests or a timeout (say, 20ms), whichever comes first. Smaller batches on low traffic, larger on high traffic.

Frameworks like NVIDIA Triton ship this out of the box. It's a meaningful improvement over static, but it still has the padding problem for variable-length outputs.

### 3. Continuous batching — the LLM game-changer

This is what modern engines like vLLM, TGI, and SGLang actually do.

The key insight: instead of thinking batch-by-request, think **batch-by-iteration** (each forward pass = one new token per sequence).

After every single decoding step, the engine checks: did any sequence finish? If yes, kick it out immediately and slot in a waiting request. New requests join the running batch **mid-flight**, not at the start of the next batch.

No wasted padding. No one hogging the batch because they want 800 tokens. Sequences enter and exit continuously, like a conveyor belt.

```python {title="Toy illustration: continuous batch slot management" run=false}
from collections import deque

# Simulated continuous batching loop (conceptual, not production code)
# Real engines (vLLM, TGI) do this inside their C++/CUDA kernels.

MAX_BATCH = 8  # max concurrent sequences

running = {}   # seq_id -> tokens_so_far
waiting = deque()  # incoming requests

def mock_decode_step(running_seqs):
    """One forward pass: returns {seq_id: new_token} for each running seq."""
    results = {}
    for sid, tokens in running_seqs.items():
        # Pretend: sequence ends after 5 tokens for demo
        new_token = "TOKEN" if len(tokens) < 5 else "<EOS>"
        results[sid] = new_token
    return results

def run_loop(requests):
    for req in requests:
        waiting.append(req)

    step = 0
    while running or waiting:
        # Fill empty slots from the waiting queue
        while len(running) < MAX_BATCH and waiting:
            req = waiting.popleft()
            running[req["id"]] = []
            print(f"  [step {step}] Added seq {req['id']} to batch")

        # One forward pass over all running sequences
        outputs = mock_decode_step(running)

        # Process results; evict finished sequences
        finished = []
        for sid, tok in outputs.items():
            if tok == "<EOS>":
                finished.append(sid)
                print(f"  [step {step}] Seq {sid} done — slot freed")
            else:
                running[sid].append(tok)

        for sid in finished:
            del running[sid]

        step += 1
        if step > 20:  # safety break for demo
            break

# Run with 12 requests competing for 8 slots
run_loop([{"id": f"req-{i}"} for i in range(12)])
```

:::widget {name="throughput"}
:::

:::gotcha
**The padding trap with static batching.** If your batch has sequences of length [12, 14, 800], every sequence gets padded to 800 tokens. You just wasted 98% of the compute on the short ones. Always use continuous batching for autoregressive LLM inference. Static batching is fine for embedding models or image classifiers where output length is fixed.
:::

## How throughput and latency trade off

Batching is not free. Bigger batches mean each individual request waits longer to be scheduled — **latency goes up as throughput goes up**. This is the fundamental serving tension.

The practical way to tune this:
- **Low traffic / latency-sensitive** (e.g. chatbot): small max batch size, short timeout.
- **High traffic / throughput-sensitive** (e.g. batch reranking, embeddings pipeline): larger batch, longer timeout.

Continuous batching helps here because you can set a `max_batch_tokens` budget instead of a fixed request count. If a waiting request is small (20 tokens), it fits easily into an existing step.

:::war-story {title="The static-batch meltdown nobody caught in staging"}
A Pune-based ML team deployed a summarisation service on Triton using static batching with batch size 64. In staging, all test documents were about the same length (400 tokens). In prod, 5% of documents were legal contracts — 8,000 tokens. Those contracts forced the entire batch of 64 to pad to 8,000 tokens each. GPU memory OOM'd, the service crashed, and 63 users' summaries were dropped to serve one legal contract. Switching to dynamic batching with `max_batch_tokens=32768` (total budget, not per-request) fixed it in 20 minutes.
:::

:::interview-line
"We use continuous batching — requests slot in and out after every decoding step, so we get near-zero padding waste and can sustain high throughput without sacrificing p99 latency the way static batching would."
:::

:::qa {q="Why is static batching bad for LLM inference specifically?"}
LLMs generate variable-length outputs autoregressively. Static batching waits for a fixed number of requests, then runs them until *every* sequence finishes. Short sequences must be padded to the length of the longest one, wasting GPU compute and memory proportional to the length variance. Continuous batching avoids this by recycling slots after each decoding step.
:::

:::qa {q="What is the trade-off when you increase batch size in a serving system?"}
Larger batches improve throughput (more tokens out per GPU-second) and reduce cost per token. But each request waits longer before it enters the batch, raising average and tail latency. The right batch size depends on your SLA: latency-sensitive chat products use small batches; async pipelines like bulk embeddings or reranking can tolerate larger ones.
:::

:::qa {q="What does 'in-flight batching' mean and which production engines support it?"}
In-flight (continuous) batching means new requests are added to a running decode iteration as soon as a finished sequence frees a slot — no waiting for the current batch to fully complete. vLLM, HuggingFace TGI, and SGLang all implement this. It is now the baseline expectation for any production LLM serving engine.
:::

:::drill {type="mcq" q="A sequence of length 10 and a sequence of length 500 are in the same static batch. How many tokens does the short sequence actually consume in the forward pass?"}
- [ ] 10 — only the real tokens
- [ ] 255 — averaged with the long one
- [x] 500 — padded to the length of the longest sequence
- [ ] 0 — it is skipped until the long one finishes
:::

:::drill {type="mcq" q="Which batching strategy eliminates nearly all padding waste for autoregressive LLM decoding?"}
- [ ] Static batching with large N
- [ ] Dynamic batching with a short timeout
- [ ] No batching — one request at a time
- [x] Continuous (in-flight) batching — swap sequences per decoding step
:::

:::key-takeaway
For autoregressive LLM serving, continuous batching is non-negotiable: it recycles GPU slots after every decoding step, eliminating padding waste and letting you scale throughput without proportionally wrecking latency.
:::
