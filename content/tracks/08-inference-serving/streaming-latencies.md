---
id: streaming-latencies
track: 08-inference-serving
title: "Streaming & the two latencies: TTFT vs TPOT"
badge: HOT
minutes: 9
prereqs: []
tags: [inference, latency, streaming, ttft, tpot, llm-serving, production]
xp: 60
hot2026: true
---

It's 11 PM. A product manager pings you: "The chat feels slow. Users are complaining they stare at a blank screen for five seconds, then everything dumps out at once." You look at your p95 metrics and everything looks green. Throughput is fine. Average latency is fine.

But the PM is right. The product feels broken.

Here's why: you were measuring the wrong thing.

## Two very different latencies

When a user types a prompt and hits Enter, two distinct clocks start ticking — and they measure completely different feelings.

**TTFT — Time To First Token.** How long before the user sees *anything* at all. This is the "blank screen" time. It's purely a waiting experience: the user is staring at a cursor, wondering if their request even registered.

**TPOT — Time Per Output Token.** Once tokens start flowing, how long between each one. This controls how *fast* the text streams in front of them. A TPOT of 30 ms per token gives a smooth, typewriter-like experience. 300 ms per token feels like the model is hunting and pecking.

These two numbers come from completely different places in your system. Fixing one does not fix the other.

:::why-prod
Users tolerate high TPOT far better than high TTFT. A blank screen for four seconds followed by fast streaming feels broken. A half-second wait followed by even 50 ms-per-token streaming feels responsive. Your SLA should treat them as separate targets — TTFT < 1 s for chat, TPOT < 40 ms per token for a fluent reading pace.
:::

## Where each latency comes from

TTFT is dominated by **everything that happens before the first decode step**: HTTP routing, auth, tokenization, prompt processing through the full prefill pass. On a long prompt — say, a 4 k-token system prompt plus user message — the model has to attend over every single token before it can emit the first output token. GPU memory bandwidth and the KV cache fill time both matter here.

TPOT is dominated by **the autoregressive decode loop**: one forward pass per output token, one token at a time. Each step loads the full model weights plus the growing KV cache from GPU HBM. Memory bandwidth, not compute, is usually the bottleneck. Doubling your GPU's FLOPS does almost nothing for TPOT. Doubling memory bandwidth — or batching more users together per step — directly cuts it.

:::table {title="TTFT vs TPOT at a glance"}
| | TTFT | TPOT |
|---|---|---|
| **What it measures** | Blank-screen wait | Streaming speed |
| **Bottleneck** | Prefill compute + KV cache fill | HBM memory bandwidth |
| **Levers** | Shorter prompts, chunked prefill, faster routing | Continuous batching, quantization, bigger batch sizes |
| **User experience impact** | Trust & perceived responsiveness | Reading flow |
| **Typical SLA (chat)** | < 1 s | < 40 ms/token |
:::

## The prefill–decode tension

Here is the trap that surprises most engineers coming from traditional APIs: **prefill and decode fight over the same GPU**.

A long prefill request is compute-heavy and monopolises GPU SMs for several hundred milliseconds. While it runs, every *other* user waiting in the decode queue stalls — their TPOT goes up. Conversely, if you batch many small decode steps together to cut TPOT, arriving prefill requests queue behind them and TTFT rises.

Serving systems like vLLM (covered in a sibling lesson) use **chunked prefill** to break long prefill passes into smaller chunks interleaved with decode steps. This is one reason purpose-built inference servers exist at all — `model.generate()` in a script gives you neither streaming nor this kind of scheduling.

```python {title="Measuring TTFT and TPOT with the OpenAI-compatible streaming API" run=false}
import time
import openai  # pip install openai; works with any OpenAI-compatible server (vLLM, LiteLLM, etc.)

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # point at your local vLLM instance
    api_key="not-needed-locally",
)

prompt = "Explain the transformer attention mechanism in simple terms."

start = time.perf_counter()
first_token_time = None
token_times = []

stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

for chunk in stream:
    now = time.perf_counter()
    delta = chunk.choices[0].delta.content
    if delta:  # skip empty keep-alive chunks
        if first_token_time is None:
            first_token_time = now
            ttft = first_token_time - start
            print(f"TTFT: {ttft * 1000:.1f} ms")
        else:
            token_times.append(now)

if token_times:
    # average gap between consecutive tokens after the first
    intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
    avg_tpot = sum(intervals) / len(intervals)
    print(f"Avg TPOT: {avg_tpot * 1000:.1f} ms/token")
    print(f"Tokens after first: {len(token_times)}")
```

Run this against a local vLLM server (`vllm serve <model>`) — no cloud account needed. Compare TTFT on a 10-token prompt vs a 2 000-token prompt. You will see TTFT scale with prompt length. TPOT will stay relatively flat because it is not affected by prompt length, only by batch size on the server.

:::gotcha
It's tempting to log only end-to-end latency (request in → last token out). That number hides everything. You can have stellar TTFT and terrible TPOT, or vice versa, and a single average looks identical. Always log and alert on TTFT and TPOT *separately* from day one. Add them to your Prometheus/Grafana dashboard before launch, not after the first PM ping.
:::

:::war-story {title="The 'fast API, slow chat' incident"}
A team deployed a Llama-3 8B model serving fine-tuned customer-support responses. Average request latency was 1.1 s — acceptable. But user surveys showed people hated the chat interface. Investigation revealed the model was not streaming at all: the server was buffering the full response and flushing it in one shot. TTFT was therefore equal to total latency: 1.1 s of blank screen every message. Enabling streaming in the server config and forwarding chunked HTTP responses to the frontend dropped perceived latency to under 300 ms without changing a single model weight.
:::

:::interview-line
"TTFT and TPOT are separate latency budgets driven by different bottlenecks — TTFT by prefill compute, TPOT by memory bandwidth — and you have to monitor and tune them independently."
:::

:::qa {q="A hiring manager asks: how would you reduce TTFT without changing the model?"}
Route requests to the nearest serving node to cut network overhead. Use chunked prefill so long prompts don't block the first token behind a wall of prefill compute. Cache the KV state of static system prompts so the model skips re-processing them every request. Shorter, tighter system prompts also help directly.
:::

:::qa {q="What does 'memory bandwidth bound' mean for LLM decode, and why does it matter?"}
During decode, the GPU spends most of its time loading model weights and the KV cache from HBM (high-bandwidth memory) rather than doing arithmetic. Adding more CUDA cores doesn't help because the cores are already idle, waiting on memory reads. You improve TPOT by increasing batch size (amortising the weight load across more users per step), using quantisation to shrink the data that needs loading, or upgrading to a GPU with higher memory bandwidth — not raw FLOPS.
:::

:::drill {type="mcq" q="A user reports that the chat feels like it hangs for 3 seconds before streaming starts, even on short replies. Which metric is most directly broken?"}
- [ ] TPOT (time per output token)
- [x] TTFT (time to first token)
- [ ] Throughput (tokens per second across all users)
- [ ] KV cache hit rate
:::

:::drill {type="mcq" q="You want to reduce TPOT by 40%. Which of these levers is most likely to achieve it?"}
- [ ] Moving the model to a GPU with 2× the FLOPS but the same memory bandwidth
- [ ] Shortening the user's input prompt from 500 tokens to 100 tokens
- [x] Increasing the number of requests batched together during each decode step
- [ ] Using a faster tokenizer library
:::

:::key-takeaway
TTFT and TPOT measure fundamentally different user experiences and have different root causes. Always track them separately — TTFT < 1 s and TPOT < 40 ms/token are the numbers that make chat feel alive.
:::
