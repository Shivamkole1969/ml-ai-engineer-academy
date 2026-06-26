---
id: quantization
track: 08-inference-serving
title: "Quantization: INT8/INT4/FP8, PTQ vs QAT, GPTQ/AWQ"
badge: HOT
minutes: 11
prereqs: []
tags: [quantization, inference, INT8, INT4, FP8, GPTQ, AWQ, PTQ, QAT, llm, serving]
xp: 60
hot2026: true
---

You joined a Pune startup four months ago. The team's 7B-parameter model is live, users love it — but the AWS bill just hit ₹1.1 lakh last month. Your CTO forwards the invoice with one word: "fix."

The model weights are sitting in FP16. Every parameter burns 2 bytes of GPU VRAM. A 7B model = 14 GB just for weights — before activations, KV-cache, or batching overhead. You're renting an A10G (24 GB) and it's 90% full before a single user token is decoded.

Quantization is how you claw that memory back. Shrink each weight from 16-bit floating point to 8-bit or 4-bit integers. Suddenly the same GPU fits twice the model or runs twice the batch. Cost halved. Done.

## What quantization actually does

Every weight in a neural network is a number. In FP16 it gets 16 bits — a 5-bit exponent and a 10-bit mantissa, which lets it represent numbers from ~6×10⁻⁵ to ~65504 with fine granularity.

INT8 gives you just 8 bits: integers from -128 to 127. To map a float weight into that range you need a **scale factor** (and sometimes a **zero-point**). The math is:

```
w_quantized = round(w_float / scale) + zero_point
```

On the way out (dequantization), you reverse it. The compute happens in low-bit integers (fast on modern hardware), then results are converted back to float for accumulation.

INT4 halves again: -8 to 7. FP8 keeps a floating-point format but compresses to 8 bits — two hardware variants exist: **E4M3** (4 exponent bits, 3 mantissa, better for weights) and **E5M2** (5 exponent bits, better for gradients). H100 and H200 GPUs have native FP8 tensor cores, making FP8 a first-class citizen for production in 2025+.

:::why-prod
A 70B model in FP16 needs ~140 GB VRAM — four A100-40GB cards minimum. INT4-quantized, it fits on two 40 GB cards, or even a single 80 GB A100. That's not a toy difference; it halves your hardware budget for the same model.
:::

:::table {title="Bit-width quick reference"}
| Format | Bits | 7B VRAM (weights) | Accuracy loss | Hardware support |
|---|---|---|---|---|
| FP16 / BF16 | 16 | ~14 GB | Baseline | All modern GPUs |
| FP8 (E4M3) | 8 | ~7 GB | Negligible | H100, H200 |
| INT8 | 8 | ~7 GB | Minimal | A10G, A100, T4 |
| INT4 (NF4/GPTQ) | 4 | ~3.5 GB | Small–moderate | Any CUDA GPU |
:::

## PTQ vs QAT — pick your trade-off

There are two points in the model's lifetime where you can apply quantization.

**Post-Training Quantization (PTQ)** happens after training is done. You feed a small calibration dataset (a few hundred representative prompts) through the model, collect statistics on weight and activation ranges, then snap the weights to low-bit integers. No gradient, no GPU cluster, done in minutes to hours. The trade-off: aggressive quantization (INT4) can slightly degrade quality, especially on tricky tasks.

**Quantization-Aware Training (QAT)** simulates quantization *during* training. The model sees fake-quantized weights in the forward pass and learns to route around the rounding error. The result is better accuracy at the same bit-width — but you need to run training again (or at least fine-tuning), which costs compute and time.

For most production LLM deployments: **start with PTQ**. It's 95% of what's used in the field. Reserve QAT for when PTQ quality is genuinely too poor and you have the GPU budget to re-fine-tune.

## GPTQ and AWQ — two smart PTQ algorithms

Plain PTQ naively rounds each weight independently. That causes rounding errors to accumulate badly in Transformers. GPTQ and AWQ are smarter.

**GPTQ** (Frantar et al., 2022) quantizes one layer at a time. For each row of a weight matrix, it uses approximate second-order information (the Hessian) to figure out which rounding errors are most damaging — then compensates for them in the *remaining* unquantized weights before moving on. The effect: the quantization error nearly cancels out across the row. Result: INT4 quality that's surprisingly close to FP16.

**AWQ** (Lin et al., 2023) takes a different angle. It notices that not all weights matter equally — weights that multiply large activations have an outsized effect on output. So before quantizing, AWQ *scales up* those salient channels (making them easier to represent in low-bit form) and scales down the corresponding activations to keep the product unchanged. Then it quantizes. AWQ typically beats GPTQ slightly in perplexity and is faster to apply.

In practice: **AWQ for latency-critical serving** (vLLM supports it natively), **GPTQ when you need the widest ecosystem compatibility**.

```python {title="Load a 4-bit AWQ model with vLLM (free to run locally)" run=false}
# pip install vllm  — free, runs on any CUDA GPU
# Download a pre-quantized AWQ model from HuggingFace, e.g.:
#   TheBloke/Mistral-7B-Instruct-v0.2-AWQ

from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",          # vLLM unpacks AWQ natively
    dtype="float16",             # activations stay in FP16
    max_model_len=4096,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain quantization in one sentence."], params)
print(outputs[0].outputs[0].text)

# Expected VRAM: ~4.5 GB vs ~14 GB for the FP16 original
# Throughput gain: typically 1.5–2x on the same GPU
```

:::gotcha
INT4 quantization of **embedding layers and LM head** hurts more than quantizing attention or MLP layers. Most good tooling (GPTQ, AWQ) keeps these in FP16 by default. If you're rolling your own pipeline and you quantize everything uniformly, you'll see a big quality drop and spend hours blaming the algorithm. Check which layers are actually quantized with `model.hf_quantizer.quantization_config`.
:::

:::war-story {title="The overnight perplexity cliff"}
A team quantized their fine-tuned 13B model with GPTQ at INT4, pushed it to prod on a Friday, and went home. By Saturday morning, support tickets were flooding in — the model kept hallucinating dates and numbers. Post-mortem: their fine-tuning data was heavy on structured numerical output, and the GPTQ calibration set was generic web text. The calibration dataset didn't represent the actual input distribution, so GPTQ protected the wrong weights. Fix: re-run GPTQ with calibration samples drawn from *your own task distribution*. Lesson learned the expensive way.
:::

:::interview-line
"We run AWQ INT4 in production — it cuts VRAM by 3.5× with less than 1 point perplexity regression, and vLLM handles the dequantization in fused CUDA kernels so we don't pay a latency penalty."
:::

:::qa {q="What is the difference between PTQ and QAT, and when would you use each?"}
PTQ quantizes a trained model using a calibration dataset — fast, no retraining, slight accuracy loss. QAT simulates quantization during training so the model adapts — better accuracy but requires a training run. Use PTQ first for LLMs; switch to QAT only if PTQ quality is unacceptable and you have budget to fine-tune.
:::

:::qa {q="Why does AWQ sometimes outperform GPTQ at the same bit-width?"}
AWQ identifies weights that multiply large activations — these have the most impact on model output. It pre-scales those salient channels to make them easier to represent at low bit-width before quantizing, rather than treating all weights equally. GPTQ compensates for rounding error after the fact using second-order information, but doesn't preferentially protect high-impact weights.
:::

:::qa {q="When would you choose FP8 over INT4 in production?"}
When you're running on H100 or H200 hardware that has native FP8 tensor cores, and your top priority is accuracy with memory savings. FP8 preserves floating-point semantics (handling outliers better than integers) so quality regression is nearly zero. INT4 saves more memory (half of FP8) but trades some accuracy — it wins when you need to fit a larger model on a fixed GPU budget.
:::

:::drill {type="mcq" q="A colleague reports that their GPTQ INT4 model is much worse than expected, even though the same algorithm worked great on a similar model. What is the MOST likely culprit?"}
- [ ] GPTQ doesn't work with models larger than 7B parameters
- [ ] INT4 always loses too much accuracy — they should use INT8 instead
- [x] The calibration dataset doesn't match the model's actual input distribution
- [ ] GPTQ requires QAT to be run first before it can quantize
:::

:::drill {type="mcq" q="You have a Llama-3 8B model in FP16 (16 GB VRAM). You apply AWQ INT4. Approximately how much VRAM do the weights now need?"}
- [ ] 12 GB — AWQ only compresses by ~25%
- [ ] 8 GB — INT4 is half of FP8, not half of FP16
- [x] 4 GB — INT4 is one-quarter the bit-width of FP16
- [ ] 2 GB — AWQ also compresses activations at runtime
:::

:::drill {type="mcq" q="Which quantization method is most appropriate when model quality after PTQ is unacceptable and you have GPU budget for retraining?"}
- [ ] GPTQ with a larger calibration set
- [ ] AWQ with salient-channel scaling
- [ ] FP8 on an H100
- [x] Quantization-Aware Training (QAT)
:::

:::key-takeaway
Quantization is the single highest-leverage inference optimization: INT4 fits a 4× larger model on the same GPU with minimal quality loss. Start with AWQ PTQ — it's one command in vLLM. Graduate to QAT only if accuracy matters more than the retraining cost.
:::
