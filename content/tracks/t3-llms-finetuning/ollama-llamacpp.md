---
id: ollama-llamacpp
track: t3-llms-finetuning
title: "Local LLMs: Ollama & llama.cpp (free)"
badge: HOT
minutes: 9
prereqs: []
tags: [ollama, llama.cpp, local-llm, inference, gguf, on-prem, privacy]
xp: 60
hot2026: true
---

You're mid-sprint when your enterprise client drops the news: "We've reviewed the architecture. Patient records cannot leave our network — not even to an API." Suddenly the cloud LLM you've been prototyping with is off the table. The meeting ends. You have two weeks to ship.

This is exactly the moment local LLMs were built for. Tools like **Ollama** and **llama.cpp** let you run a capable model on your own laptop or server — no internet, no per-token billing, no data leaving the building. Getting started takes about five minutes.

## What "Local LLM" Actually Means

A local LLM is a quantized model (compressed weights, smaller file) running entirely on your hardware — CPU, GPU, or Apple Silicon. You download the model file once, and inference happens on-device forever.

The most common reasons teams go local:

- **Privacy / compliance** — healthcare, finance, government, legal
- **Zero inference cost** — no per-token billing once downloaded
- **Predictable latency** — no shared API queues, rate limits, or upstream outages
- **Offline development** — client sites, conferences, spotty Wi-Fi
- **Fast experimentation** — swap models in seconds without burning budget

The trade-off is hardware. A 7B model in 4-bit quantization (covered in the *Quantization* lesson) needs roughly 6–8 GB of RAM. A 13B model needs ~10 GB. Modern M1/M2/M3 MacBooks handle this well because their unified memory serves both CPU and GPU — a 16 GB MacBook Air can run a capable 7B model smoothly.

## llama.cpp: The Engine Underneath

`llama.cpp` is a C++ inference library by Georgi Gerganov. It runs quantized models fast — on CPU, NVIDIA CUDA, AMD ROCm, and Apple Silicon Metal — and reads models in a format called **GGUF** (a single self-contained file: weights plus metadata, one download).

You rarely touch llama.cpp directly. But knowing it exists matters because every major local LLM tool — Ollama, LM Studio, Jan — runs llama.cpp under the hood. When something breaks, understanding the layer below lets you debug it.

## Ollama: The Part You Actually Use

Ollama wraps llama.cpp into a dead-simple CLI plus a local REST server. Install it, pull a model, and you have a running LLM API that is **100% compatible with the OpenAI client**. That compatibility is the killer feature — your existing code works without modification.

```bash
# Install from https://ollama.com  (macOS, Linux, Windows)
ollama pull llama3.2          # ~2 GB download  (3B params, 4-bit)
ollama pull mistral           # ~4 GB download  (7B params, 4-bit)
ollama run llama3.2           # opens an interactive chat session
```

Ollama starts a server at `http://localhost:11434`. Point your OpenAI SDK at it:

```python {title="Drop-in local inference with Ollama" run=false}
from openai import OpenAI

# One line changes. Everything else stays identical.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",   # required by the SDK, ignored by Ollama
)

response = client.chat.completions.create(
    model="llama3.2",   # must match what you pulled
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "Explain RAG in one sentence."},
    ],
    temperature=0.3,
)

print(response.choices[0].message.content)
# Works exactly like it would against gpt-4o-mini
```

The `base_url` swap is the whole trick. Write your code against Ollama in dev, point it at a cloud API in staging or production by changing one environment variable. No new SDK, no refactoring.

:::why-prod
Local LLMs unlock privacy-sensitive verticals that cloud APIs simply cannot serve. Ollama's OpenAI-compatible endpoint means your code is portable: build offline, deploy on cloud GPUs — same `chat.completions.create()` call, same behaviour.
:::

:::table {title="Ollama vs. llama.cpp direct: when to use each"}
| | Ollama | llama.cpp (direct) |
|---|---|---|
| **Setup** | One-liner install | Compile from source |
| **API** | OpenAI-compatible REST | CLI or embedded C++ |
| **Best for** | Dev, prototyping, small deploys | Custom serving, fine-grained control |
| **Models** | Ollama Hub (pull by name) | HuggingFace GGUF files |
| **GPU offloading** | Automatic | Manual flag (`--n-gpu-layers`) |
| **High concurrency** | Not designed for it | Use llama.cpp server or vLLM instead |
:::

:::gotcha
Never pull a model larger than your available RAM. It sounds obvious, but people grab `llama3:70b` on a 16 GB laptop, the OS starts swapping to disk, inference crawls to 0.5 tokens per second, and they conclude "local LLMs are too slow." They're not — the model choice was wrong. **Rule of thumb: GGUF file size in GB + 2 GB headroom = minimum RAM needed.** Check `free -h` (Linux) or Activity Monitor (macOS) for *available* memory, not *installed* memory.
:::

:::war-story {title="The 70B Model That Took Down a Compliance Server"}
A fintech team set up Ollama in an air-gapped on-prem server to summarise documents for compliance review. They pulled `llama3:70b` — technically fitting in the server's 64 GB RAM. But the same machine was running a monitoring stack, a database, and a queue. Under load, memory pressure spiked and the Linux OOM killer started terminating inference workers mid-request. Documents were partially processed, audit trails broke, and the compliance team had a very bad afternoon. The fix: switch to `mistral:7b`, add a task queue, and monitor memory before scaling up. Six months later — zero incidents.
:::

:::interview-line
"I use Ollama in dev because it gives me an OpenAI-compatible local endpoint — same client code, same method calls, just a different base_url when I need to go cloud."
:::

:::qa {q="When would you use llama.cpp directly instead of Ollama?"}
When you need fine-grained control — tuning GPU layer offloading, embedding llama.cpp inside a C++ application, or serving at higher concurrency than Ollama supports. Ollama is a single process optimised for convenience. For production-scale serving, you'd reach for the llama.cpp HTTP server or, more commonly, vLLM.
:::

:::qa {q="A client insists on zero data leaving their servers. How do you approach LLM selection?"}
Start with a quantized open-weight model (Llama 3, Mistral, Qwen) pulled via Ollama onto their hardware. Run a quick benchmark on their target tasks to pick the right size — usually the smallest model that hits acceptable quality. Then document the inference setup so it can be reproduced; local models have no external dependency to break.
:::

:::drill {type="mcq" q="You set base_url='http://localhost:11434/v1' in your OpenAI client. What needs to change in the rest of your chat.completions.create() call?"}
- [ ] Add an ollama=True flag to every request
- [ ] Switch from the openai library to the requests library
- [x] Nothing — Ollama exposes the same API shape as OpenAI
- [ ] Replace model= with local_model= in the call signature
:::

:::drill {type="mcq" q="A teammate says 'local LLMs are unusably slow on my laptop.' What's the most likely root cause?"}
- [ ] llama.cpp doesn't support GPU acceleration on consumer hardware
- [ ] Quantized models are fundamentally lower quality than cloud APIs
- [x] The chosen model is too large for available RAM, causing disk swapping
- [ ] Ollama doesn't expose streaming, making responses feel slow
:::

:::key-takeaway
Ollama is the fastest path to a local LLM: one install, one pull, one OpenAI-compatible REST endpoint. Match model size to your available RAM before anything else — that single decision determines whether local inference feels fast or broken.
:::
