---
id: pytorch-vs-jax
track: 01-core-ml
title: "PyTorch vs JAX — pick on purpose"
badge: CORE
minutes: 7
prereqs: []
tags: [pytorch, jax, frameworks, ml-stack, deep-learning]
xp: 45
hot2026: false
---

You just joined a Pune startup's ML team. End of your first week, someone asks: "For this new project — JAX or PyTorch?" The room looks at you. You pause. You say "PyTorch" — but you said it because that's what you know, not because you reasoned through it.

That's the trap this lesson helps you escape.

Both frameworks can train the same model. But they come from different philosophies, have different ergonomics, and make very different production trade-offs. Knowing *why* to pick one — and being able to say it out loud — is what separates an ML engineer from someone who just runs notebooks.

## Two Very Different Philosophies

**PyTorch** treats your model like an object. You subclass `nn.Module`, define `forward()`, and Python runs it line-by-line as you call it. The computation graph is built *on the fly* as tensors flow through. This is called **eager execution** — what you write is what runs, immediately.

**JAX** treats your model like a *pure function*. There are no classes, no `.backward()` — instead, you write regular Python functions and pass them to JAX's **function transforms**: `jax.grad()` to differentiate, `jax.jit()` to compile, `jax.vmap()` to vectorise over a batch dimension. The whole computation gets compiled down to XLA and dispatched to whatever hardware is available.

These aren't just syntax differences. They're two genuinely different mental models for what a neural network *is*.

:::why-prod
PyTorch's eager mode makes debugging trivial — set a breakpoint anywhere, print tensors mid-forward-pass, inspect shapes in real time. JAX's functional + JIT style enables aggressive compiler optimisations (op fusion, auto-parallelism) but eats your stack trace when something breaks. The right choice depends on your team's debugging culture, not raw benchmark numbers.
:::

## The Same Gradient, Two Mental Models

The simplest illustration is computing a gradient. Same math, completely different approach:

```python {title="Gradient computation: PyTorch vs JAX" run=false}
# ── PyTorch: stateful, autograd lives on the tensor ──────────────────
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # computation graph is built here, eagerly
y.backward()
print(x.grad)     # 6.0  →  d(x²)/dx evaluated at x=3

# ── JAX: functional, grad is a transform on a function ───────────────
import jax

def f(x):
    return x ** 2

grad_f = jax.grad(f)      # returns a NEW callable — not a value
print(grad_f(3.0))        # 6.0 — same answer, completely different worldview

# Run locally (free): pip install torch jax[cpu]
```

Notice what changed. In PyTorch, the tensor *carries* the gradient — the graph lives inside `x`. In JAX, `grad` *wraps the function* and gives you a new one. `grad_f` is now a callable that computes derivatives, not a number.

Once that shift clicks, JAX becomes intuitive. But it does take a weekend to internalize.

## When to Pick Which

:::table {title="PyTorch vs JAX at a glance"}
| Dimension | PyTorch | JAX |
|---|---|---|
| Execution model | Eager by default | Lazy + JIT compiled via XLA |
| Debugging | Easy — breakpoint anywhere | Hard — traces compile away |
| HuggingFace / OSS ecosystem | Massive, PyTorch-first | Growing but much smaller |
| Hardware sweet spot | CUDA GPUs (excellent), TPUs (ok) | TPUs (native), GPUs (strong) |
| Team fit | Most Indian startups, product ML | Google / DeepMind-style research |
| Parallelism primitive | `DataParallel`, `DistributedDataParallel` | `pmap` over devices natively |
| Interview relevance (2026) | Asked in almost every ML role | Asked at Google, research labs |
:::

The honest answer for most Pune-based roles: **PyTorch wins by default.** The OSS ecosystem — HuggingFace, diffusers, vLLM, trl, everything — is PyTorch-first. Most Indian product companies and AI startups run PyTorch in production. If a job description says "experience with transformers" and doesn't mention JAX, safely assume PyTorch.

Pick JAX when: you're targeting Google, DeepMind, or a pure research lab; the project runs on TPUs; you need `vmap` for per-sample gradient tricks (useful in differential privacy or meta-learning); or the team is already on JAX and migration cost is real.

:::gotcha
"JAX is faster so I should always use it." This is the most common mis-framing. JAX *can* be faster — but only after JIT warm-up, and only when XLA can fully fuse your ops. In eager/un-JIT'd mode, PyTorch and JAX are similar. More importantly: speed means nothing if your team can't debug a production failure at 2am. Don't optimise for benchmarks before you optimise for operability.
:::

:::interview-line
"I default to PyTorch because the OSS ecosystem is PyTorch-first, but I understand JAX's functional transforms well enough to switch if the team is already there and the hardware warrants it."
:::

:::qa {q="What is the key conceptual difference between PyTorch and JAX?"}
PyTorch uses eager execution — the graph builds as code runs, tensors carry gradient history, and `.backward()` traverses it. JAX is functional and JIT-first — you write pure functions, then apply transforms like `grad` and `vmap` that compile via XLA. PyTorch feels like Python with tensors; JAX feels like a differentiable math library that happens to run on GPUs.
:::

:::qa {q="When would you choose JAX over PyTorch for a production training job?"}
If the team runs on TPUs and needs XLA's compiler optimisations natively, or if the work needs composable function transforms — for example, per-sample gradients with `vmap` (used in differential privacy) or clean multi-device sharding with `pmap`. JAX also wins in research settings where custom gradient rules (`jax.custom_vjp`) are needed without monkey-patching. For most product ML work, PyTorch's ecosystem advantage is decisive.
:::

:::drill {type="mcq" q="A teammate says 'We should rewrite our training pipeline in JAX because benchmarks show it's faster.' What is the most complete response?"}
- [ ] Agree — JAX always outperforms PyTorch because XLA fuses ops
- [ ] Disagree — PyTorch is faster because eager execution avoids compilation overhead
- [x] It depends — JAX can win after JIT warm-up, but debugging cost, ecosystem lock-in, and team familiarity must all factor in
- [ ] Neither framework matters; use the one with the best TPU support regardless of team skills
:::

:::drill {type="mcq" q="In JAX, what does jax.grad(f) return?"}
- [ ] The scalar gradient value of f at its current inputs
- [ ] A compiled XLA binary that runs on the next call
- [x] A new callable that computes the gradient of f with respect to its first argument
- [ ] A PyTorch-compatible autograd node
:::

:::key-takeaway
PyTorch is the safe default for most ML engineering roles in India — the ecosystem, tooling, and job market all point there. Understand JAX's functional-transform model well enough to speak to it confidently in an interview, and switch only when the project or team genuinely calls for it.
:::
