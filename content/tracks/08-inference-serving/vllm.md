---
id: vllm
track: 08-inference-serving
title: "vLLM — the reference serving engine"
badge: HOT
minutes: 9
prereqs: []
tags: [vllm, inference, paged-attention, kv-cache, llm-serving, gpu]
xp: 60
hot2026: true
---

It's Friday evening. A startup in Pune has just launched a coding-assistant feature backed by Llama 3.1 70B. Traffic is light, everything looks fine. Saturday morning — a tech blog covers them — and suddenly 300 users hit the API concurrently. The GPU is sweating. Latency spikes to 30 seconds. The team's first instinct is "we need more GPUs." But a teammate says: "Did you try vLLM?"

They swap the backend. Memory usage drops 40 %. Throughput triples. No extra hardware.

That's what vLLM does.

## What makes LLM serving hard

When a model generates a token, it needs a **KV cache** — a chunk of GPU memory that holds intermediate attention computations (keys and values) for every token it has seen so far in the sequence. The longer the conversation, the bigger the cache.

With a naive implementation, you have to allocate that memory *upfront* — for the maximum possible sequence length. Most requests are short, so you waste enormous amounts of VRAM. Worse, fixed allocations fragment memory like a 2003 Windows XP machine — gaps appear everywhere and you can't fit a new request even when free memory exists.

vLLM fixes this with one elegant idea.

## PagedAttention: the core trick

vLLM borrows a page from operating systems. In an OS, physical RAM is split into fixed-size *pages* and mapped to processes on demand. Nothing is wasted; a process only holds the pages it actually needs.

vLLM does the same for KV cache:

- GPU memory is carved into **fixed-size blocks** (think: 16 tokens per block by default).
- Each request's KV cache is stored in **non-contiguous blocks**, tracked by a lookup table.
- Blocks are allocated **as tokens are generated** — not in advance.
- When a request finishes, its blocks are freed immediately and reused.

This is **PagedAttention**. The result: near-zero KV cache memory waste, and you can pack far more concurrent requests onto the same GPU.

:::why-prod
Production LLM services live and die by GPU utilisation. Wasted VRAM = wasted money = slower responses for every user sharing that GPU. PagedAttention is the single biggest reason managed APIs (like OpenAI, Groq, and every Indian cloud AI offering) can serve thousands of users on a finite GPU cluster without scaling linearly.
:::

## How continuous batching plugs in

PagedAttention solves *memory*. Continuous batching solves *throughput*. Together they make vLLM fast.

With continuous batching, the engine never waits for the whole batch to finish before accepting a new request. The moment one sequence produces its final token, its GPU slot (and now its KV cache blocks) are handed to a waiting request. Utilisation is near-constant.

vLLM implements both. That's why it's the reference engine.

:::table {title="vLLM architecture at a glance"}
| Component | Job | Why it matters |
|---|---|---|
| PagedAttention kernel | Non-contiguous KV block management | Eliminates memory fragmentation |
| Scheduler | Continuous batching + priority queues | Maximises GPU utilisation |
| Block manager | Tracks free/used blocks per sequence | Enables preemption + swapping |
| OpenAI-compatible API | `/v1/completions` & `/v1/chat/completions` | Drop-in for existing clients |
:::

## Spinning it up locally

vLLM runs on any Linux box with a CUDA GPU. For dev/testing, a single A10 or even a T4 on Google Colab (free tier) is enough for a 7B model.

```python {title="Start vLLM server (Mistral 7B)" run=false}
# pip install vllm  (CUDA 12.1 wheel ships by default)
# Free option: run on Colab A100 or Vast.ai T4 (~₹1/hr)

from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    # Reduce KV cache block size if low on VRAM:
    # gpu_memory_utilization=0.85,  # default is 0.90
)

params = SamplingParams(temperature=0.7, max_tokens=256)

prompts = [
    "Explain PagedAttention in one paragraph.",
    "What is continuous batching?",
]

outputs = llm.generate(prompts, params)

for out in outputs:
    print(out.outputs[0].text)
```

```bash {title="OpenAI-compatible HTTP server" run=false}
# Exposes POST /v1/chat/completions — swap your OpenAI client base_url
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dtype bfloat16 \
  --max-model-len 8192
```

## Prefix caching — the bonus trick

vLLM 0.4+ adds **automatic prefix caching** (APC). If many requests share a long system prompt (e.g., a 2000-token RAG preamble), the KV blocks for that prefix are computed once and reused across requests. Your TTFT for subsequent requests collapses dramatically. Enable it with `--enable-prefix-caching`.

:::gotcha
vLLM's `gpu_memory_utilization` defaults to 0.90 — it grabs 90 % of VRAM for KV cache blocks. On a shared GPU machine this will OOM-kill your neighbour's process. Always set this explicitly (try 0.75 to start) and monitor `nvidia-smi` before going to production. Also: vLLM's offline batch API and online server have different timeout/error semantics — test both if your use case mixes async jobs with real-time chat.
:::

:::war-story {title="The Friday night OOM incident"}
A fintech team in Pune deployed vLLM on a shared A100 with `gpu_memory_utilization=0.90` (the default). A data-science colleague was simultaneously running a fine-tuning job on the same GPU. At peak load, vLLM's block allocator expanded, the fine-tuner ran out of memory, and the whole node crashed. The fix took three minutes: set `gpu_memory_utilization=0.75` and add a `MIG` partition. But the post-mortem took three hours. Defaults are not production settings.
:::

:::interview-line
"vLLM's PagedAttention treats KV cache like OS virtual memory — blocks are allocated on demand, not pre-reserved — which is why it can pack 3–5x more concurrent requests onto the same GPU compared to a naive implementation."
:::

:::qa {q="What problem does PagedAttention solve, and how?"}
Naive KV cache allocation wastes GPU memory because it reserves space for the maximum possible sequence length upfront. PagedAttention splits the KV cache into fixed-size blocks and allocates them incrementally as tokens are generated — just like OS paging. This eliminates fragmentation and wasted VRAM, letting many more requests share the same GPU.
:::

:::qa {q="How is vLLM different from just calling model.generate() in a loop?"}
A plain generate() loop processes one request at a time and holds the full KV cache in memory from the start. vLLM adds continuous batching (new requests join mid-batch the moment a slot frees), PagedAttention (non-contiguous KV blocks), and a scheduler that manages preemption and swapping — together raising throughput by 10–20x at the same latency target.
:::

:::qa {q="When would you NOT use vLLM?"}
vLLM requires Linux + CUDA. For Mac development, llama.cpp or Ollama are better. For fine-tuning jobs (not serving), vLLM adds no value. And for extremely latency-sensitive single-user CLI tools, its server overhead (FastAPI, async scheduling) can be overkill compared to direct Transformers inference.
:::

:::drill {type="mcq" q="A team sees that their vLLM server handles 50 req/s at low traffic but drops to 12 req/s as sequence lengths grow longer. What is the most likely bottleneck?"}
- [ ] The model weights are being reloaded from disk on each request
- [ ] Continuous batching is disabled in their config
- [x] KV cache blocks are exhausted; the scheduler must preempt or wait, starving the batch
- [ ] The OpenAI-compatible API layer is throttling requests
:::

:::drill {type="mcq" q="What does setting --enable-prefix-caching do in vLLM?"}
- [ ] Caches the model weights in CPU RAM to reduce GPU load
- [ ] Stores completed responses in a Redis cache for identical prompts
- [x] Reuses computed KV blocks for a shared prompt prefix across different requests
- [ ] Enables disk-based KV cache offloading to extend effective VRAM
:::

:::drill {type="mcq" q="You're deploying a vLLM server on a machine with 40 GB VRAM that you share with a colleague. Which gpu_memory_utilization value is the safest starting point?"}
- [ ] 0.95 — maximise throughput
- [ ] 1.0 — vLLM will auto-manage
- [x] 0.75 — leaves a reasonable buffer for the other workload
- [ ] 0.10 — GPU sharing requires almost no utilisation
:::

:::key-takeaway
vLLM's PagedAttention + continuous batching is why production LLM APIs can serve hundreds of concurrent users on a handful of GPUs. Master the memory model — `gpu_memory_utilization`, block size, prefix caching — and you can tune any deployment without throwing hardware at the problem.
:::
