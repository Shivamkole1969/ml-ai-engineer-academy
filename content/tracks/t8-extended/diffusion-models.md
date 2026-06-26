---
id: diffusion-models
track: t8-extended
title: "Diffusion models & image generation"
badge: CORE
minutes: 9
prereqs: []
tags: [diffusion, generative-ai, stable-diffusion, image-generation, deep-learning]
xp: 45
hot2026: false
---

Imagine your product team comes to you on a Monday morning: "We want to add AI image generation to the app by Friday." You nod calmly, but inside you're wondering — is this a GAN? A VAE? Something else entirely? You open the Stable Diffusion docs and see words like *forward process*, *score matching*, and *classifier-free guidance*. This lesson gives you the mental model to read those words without flinching — and to deploy the result without melting your GPU budget.

## What Even Is a Diffusion Model?

Here is the core trick: instead of teaching a model to draw an image in one shot (hard), you teach it to *clean up a slightly noisy image* (much easier). Do that thousands of times in a row, starting from pure random noise, and you get a crisp image.

There are two processes:

**Forward process** — You take a real image and *add Gaussian noise* in T small steps until it looks like TV static. This is pure math; no learning happens here. Think of it as slowly stirring sand into a glass of water until you can no longer see the water.

**Reverse process** — A neural network (usually a U-Net) learns to *predict and remove the noise* one step at a time, going from static back to a coherent image. This is what gets trained. The model never sees the final image directly; it just keeps answering: "what noise was added here?"

At inference time you skip the forward process entirely. Start with random noise and run the reverse process — the network sculpts it into an image.

:::why-prod
Production teams care about diffusion models because they currently produce the highest-quality image outputs at scale: product photo generation, creative asset pipelines, game texture synthesis, and medical imaging augmentation all rely on them. Understanding the mechanics lets you tune quality vs. speed trade-offs and debug artifacts rather than just hoping the defaults work.
:::

## Latent Diffusion: Why Stable Diffusion Is Fast

Running diffusion on raw 512×512 pixels is slow — each denoising step touches millions of values. Stable Diffusion's insight: compress the image into a tiny *latent space* first (using a VAE encoder), run all the diffusion steps there, then decode back to pixels at the very end.

The latent space is ~8× smaller in each spatial dimension. This cuts compute dramatically while barely touching visual quality. That is why you can run Stable Diffusion on a consumer GPU when older approaches needed data-center hardware.

**Text conditioning** happens via CLIP text embeddings injected into each U-Net layer through cross-attention. The text prompt literally steers the denoising at every step — which is why prompt engineering has such a big impact.

:::table {title="Diffusion model family tree"}
| Model | Key idea | Typical use |
|---|---|---|
| DDPM (Ho et al. 2020) | Original formulation, ~1000 steps | Research baseline |
| DDIM | Deterministic sampling, ~50 steps | Faster inference, same weights |
| Stable Diffusion 1.x / 2.x | Latent diffusion + CLIP text | Open-source image gen |
| SDXL | Larger U-Net, 1024px native | High-res product images |
| Stable Diffusion 3 / Flux | Transformer-based (DiT) backbone | State-of-art quality 2024+ |
| ControlNet | Adds spatial conditioning (pose, depth) | Controlled generation |
:::

## Classifier-Free Guidance (CFG)

This is the knob that most people misuse. CFG runs the denoiser *twice* per step — once with your text prompt and once without (unconditional). It then pushes the output *away* from the unconditional direction and *toward* the prompt direction by a factor called the guidance scale.

- **CFG = 1** — pure unconditional; prompt is ignored
- **CFG = 7–9** — sweet spot; prompt followed, good diversity
- **CFG > 15** — prompt over-followed; images get oversaturated and lose realism

:::gotcha
Setting guidance scale too high (>12) is one of the most common mistakes in production image pipelines. Colors blow out, textures become waxy, and faces distort. If your generated images look "AI-ish" in a bad way, lower CFG before anything else. Also watch out for NSFW or copyright-infringing outputs at high CFG — they tend to memorize training data more aggressively.
:::

## Inference in Code

```python {title="Minimal Stable Diffusion inference (free via diffusers)" run=false}
# pip install diffusers transformers accelerate
# Free to run locally on CPU (slow) or any CUDA GPU
from diffusers import StableDiffusionPipeline
import torch

# Load model — downloads ~4 GB on first run, cached after
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,        # halve VRAM usage
)
pipe = pipe.to("cuda")  # swap to "cpu" if no GPU (very slow)

# Generate — num_inference_steps controls quality vs. speed
image = pipe(
    prompt="a golden retriever in a sunlit library, photorealistic",
    negative_prompt="blurry, low quality, cartoon",  # what to avoid
    num_inference_steps=30,   # 20–50 is practical; 1000 is DDPM default
    guidance_scale=7.5,       # CFG sweet spot
    height=512,
    width=512,
).images[0]

image.save("output.png")
# Tip: use DPM-Solver++ scheduler for same quality in ~20 steps
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

## Faster Sampling: DDIM and DPM-Solver

DDPM needs ~1000 denoising steps. That would take minutes per image. DDIM rewrites the math as a *deterministic ODE* instead of a stochastic SDE, letting you skip most steps — 50 steps gets 80% of the way there. DPM-Solver++ can get acceptable quality in as few as 20 steps by using higher-order integration. In production you almost never use the default DDPM scheduler.

:::interview-line
"Diffusion models learn to reverse a noise-adding process; in production you skip the forward pass entirely — start from random noise and denoise with a fast ODE solver in 20–50 steps."
:::

:::qa {q="What is the role of the U-Net in a diffusion model?"}
The U-Net is the learned denoiser. At each reverse step it takes the noisy latent and a noise-level embedding as input, and predicts the noise that was added. Subtracting the predicted noise gives a slightly cleaner latent. It uses skip connections to preserve spatial detail and cross-attention layers to condition on text embeddings.
:::

:::qa {q="Why does Stable Diffusion operate in latent space rather than pixel space?"}
Pixel space for a 512×512 image has ~786k values per step — very expensive to denoise hundreds of times. A VAE compresses the image into a ~64×64 latent (8× smaller each side), so diffusion runs ~64× faster with negligible quality loss. The VAE decoder reconstructs the final image only once, at the very end.
:::

:::drill {type="mcq" q="Classifier-free guidance (CFG) scale of 20 will most likely cause which of the following?"}
- [ ] Faster inference because fewer diffusion steps are needed
- [ ] Smoother gradients and better image diversity
- [x] Over-saturated colors and loss of photorealism
- [ ] The text prompt to be ignored entirely
:::

:::drill {type="mcq" q="Which component in Stable Diffusion handles the text-to-image conditioning?"}
- [ ] The VAE encoder compresses text into latents
- [ ] The DDIM scheduler injects prompt tokens at each step
- [ ] The UNet's batch-norm layers align image and text distributions
- [x] Cross-attention layers in the U-Net attend to CLIP text embeddings
:::

:::key-takeaway
Diffusion models work by learning to reverse a noise-adding process — start from static, denoise step by step. In production: use latent diffusion for speed, keep CFG between 7 and 9, and switch to DDIM or DPM-Solver for fast sampling without retraining.
:::
