---
id: gpu-vs-cpu
track: 08-inference-serving
title: "GPU vs CPU — choose by workload"
badge: FOUNDATION
minutes: 6
prereqs: []
tags: [inference, gpu, cpu, serving, cost, throughput, latency]
xp: 30
hot2026: false
---

Your startup just shipped an LLM-powered feature. You rent an A100 GPU on AWS — $3.20/hour. Traffic
trickles in: five requests a minute. The GPU sits 97% idle all night. At the end of the month the
bill lands and your manager asks the question you should have asked first: *"Did we actually need
a GPU for this?"*

Sometimes the answer is no. Knowing which hardware fits your workload is one of those things that
separates engineers who spend wisely from engineers who just copy the tutorial.

## What makes a GPU fast — and when it stops mattering

A GPU isn't universally faster. It's fast at one specific thing: doing *many simple operations at
the same time*. An A100 has 6,912 CUDA cores all executing in lockstep. A CPU has maybe 32 cores,
but each one is smarter and can jump around in memory freely.

For inference this creates a clear dividing line: **batch size**.

When a request arrives alone (batch size = 1), a transformer's matrix multiplications only fill a
tiny slice of the GPU. The rest of the silicon is waiting. You're paying for a sports car and
driving 20 km/h in third gear. The CPU, in contrast, has no such penalty — it's optimised for
exactly this single-stream, low-parallelism case.

Push batch size to 64 and the GPU roars to life. Every core is useful. Throughput climbs. Cost per
token drops. The GPU now has an overwhelming edge.

The real bottleneck in transformer inference is **memory bandwidth**, not raw floating-point
compute. Every token generation reads the entire model's weights from memory once. A100 HBM2e
delivers ~2 TB/s. A beefy CPU delivers ~100–200 GB/s. For a large model at low batch size,
bandwidth is everything — and the GPU still wins on that axis, but you need *enough* requests to
keep it busy to make the price worth it.

:::why-prod
At low QPS (< ~10 req/min) a quantised model on CPU can be cheaper than a GPU instance and still
meet your latency SLA. At high QPS the GPU's bandwidth advantage becomes decisive — CPU can't
keep up regardless of how many cores you throw at it.
:::

## The decision matrix

:::table {title="GPU vs CPU for inference — quick reference"}
| Factor | CPU wins | GPU wins |
|---|---|---|
| QPS | < ~5–20 req/min | > ~50 req/min |
| Model size | < 7B params (or aggressively quantised) | 7B–70B+ |
| Batch size | 1–4 | 8–256+ |
| Latency SLA | Relaxed (seconds OK) | Strict (< 500 ms) |
| Cost sensitivity | Very high | Throughput > cost |
| Where to run | Edge, on-prem, low-traffic API | Cloud, high-traffic API |
:::

A useful rule of thumb: if you can't sustain a batch size of at least 8 in production (averaged
over time), do the math on a CPU deployment before provisioning a GPU.

## llama.cpp: the CPU comeback

A few years ago "CPU inference" meant slow toy demos. That changed. Libraries like **llama.cpp**
run 4-bit quantised LLMs on a Mac M2 or a cheap 32-core VM at 20–40 tokens/second — plenty for
many internal tools, chatbots, or batch jobs.

```python {title="Benchmarking tokens/sec on CPU with llama-cpp-python" run=false}
# pip install llama-cpp-python  (free, local, no GPU needed)
from llama_cpp import Llama

# Load a 4-bit quantised GGUF model (download from HuggingFace)
llm = Llama(
    model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,       # tune to your CPU core count
    n_gpu_layers=0,    # force CPU-only; set > 0 to offload layers to GPU
    verbose=False,
)

import time

prompt = "Explain the difference between GPU and CPU inference in two sentences."
start = time.time()
output = llm(prompt, max_tokens=128)
elapsed = time.time() - start

tokens_generated = output["usage"]["completion_tokens"]
print(f"{tokens_generated / elapsed:.1f} tokens/sec")
# On a 32-core VM: expect 15–30 tok/s for a 7B Q4 model
# On Apple M2 Pro: expect 35–50 tok/s (uses Metal GPU automatically)
```

Set `n_gpu_layers` to a positive integer and llama.cpp will offload that many transformer layers to
a GPU — a handy middle ground when you have a smaller GPU (8 GB VRAM) and a larger model.

:::gotcha
Don't benchmark on your laptop and assume the production CPU VM will be similar. Cloud VMs share
memory bandwidth with neighbours. A `c7g.8xlarge` under neighbour contention can be 30–40% slower
than a quiet dedicated metal box with the same specs. Always load-test on the actual target
instance type.
:::

:::interview-line
"The GPU vs CPU choice is really a batch-size and QPS question — at low traffic a quantised model
on CPU is often cheaper and sufficient; the GPU earns its cost when you can keep it batch-full."
:::

:::qa {q="When would you choose CPU inference over a GPU in production?"}
When QPS is low enough that a GPU would sit mostly idle, the cost of the GPU instance outweighs its
speed benefit. Quantised models (INT4/INT8) on CPU via llama.cpp or ONNX Runtime can hit 20–50
tokens/second — enough for internal tools, low-traffic APIs, or edge deployments. The break-even
point depends on your SLA, model size, and cloud pricing, so always benchmark both paths.
:::

:::qa {q="Why does batch size matter so much for GPU utilisation?"}
A GPU achieves speed through massive parallelism — thousands of cores doing the same operation
simultaneously. At batch size 1, only a fraction of those cores are active during each matrix
multiply; the rest stall. Larger batches fill more cores and amortise the memory-bandwidth cost
of loading model weights across more tokens at once, dramatically improving tokens-per-second and
cost-per-token.
:::

:::drill {type="mcq" q="Your team runs a Slack bot that gets ~8 LLM requests per hour. It uses a 7B model. Which infra choice is most cost-effective?"}
- [ ] A100 GPU instance — best latency
- [x] CPU VM with a Q4 quantised model — low QPS means GPU is wasteful
- [ ] Multi-GPU setup for redundancy
- [ ] Spot GPU instance to save cost
:::

:::drill {type="mcq" q="A transformer model at batch size 1 is primarily bottlenecked by:"}
- [ ] Floating-point compute (FLOPS)
- [ ] CPU branch prediction
- [x] Memory bandwidth — weights must be read from VRAM on every token
- [ ] Network I/O between CPU and GPU
:::

:::key-takeaway
GPU wins on throughput when you can keep it busy; CPU wins on cost when you can't. Check your QPS
and batch size before provisioning — the right hardware is the one that stays busy.
:::
