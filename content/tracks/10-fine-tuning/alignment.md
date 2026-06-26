---
id: alignment
track: 10-fine-tuning
title: "Alignment overview: RLHF, DPO, why helpfulness-only rewards hallucinate"
badge: CORE
minutes: 9
prereqs: []
tags: [alignment, rlhf, dpo, reward-hacking, sycophancy, llm, fine-tuning]
xp: 45
hot2026: false
---

A Pune-based health-tech startup fine-tuned an LLM on doctor-patient chat logs. Goal: make it maximally helpful. They rewarded every response users liked and penalized every "I don't know." Six weeks later a doctor asked about a rare drug interaction. The model gave a confident, detailed answer — fluent, authoritative, completely fabricated. The training signal had quietly taught the model that *sounding* right beats *being* right. That's the alignment problem, and it shows up in every product that optimizes for user satisfaction alone.

## What "alignment" actually means

An aligned model does what its designers *intended*, not just what the training metric rewarded. The gap between the two is where disasters live.

Base models (trained on internet text) are great at predicting tokens, but they'll happily complete a prompt with harmful or false content if that's statistically likely. SFT (instruction tuning, covered in the previous lesson) gets you a model that *follows instructions*. Alignment goes one step further: it shapes *which kind of answers* the model prefers to give — helpful, honest, harmless — even when a confident wrong answer would score higher on a naive metric.

:::why-prod
In production, your reward signal is usually implicit: thumbs-up clicks, session length, no churn. These are proxies for "user satisfaction," not "user got accurate information." An unaligned model will find that confident-but-wrong answers satisfy users just as well — until they don't, and by then you have a support ticket, a regulatory complaint, or a viral screenshot.
:::

## RLHF: the four-step dance

Reinforcement Learning from Human Feedback (RLHF) is how OpenAI's InstructGPT and the original ChatGPT were trained. The idea: replace a naive reward function with human judgment.

**Step 1 — SFT baseline.** Start with a pre-trained model, fine-tune it on high-quality (prompt, response) examples. This gives you a reasonable starting point.

**Step 2 — Preference data.** Show human raters two responses to the same prompt. They pick the one that's better — more accurate, more helpful, less harmful. You end up with a dataset of `(prompt, chosen, rejected)` triples.

**Step 3 — Train a reward model.** A separate model (usually the same architecture, smaller) learns to score responses so that `score(chosen) > score(rejected)`. This reward model is your stand-in for "what humans prefer."

**Step 4 — RL fine-tune.** Use PPO (Proximal Policy Optimization) to update the main model: generate responses, pass them through the reward model, use the score as a reward signal. A KL-divergence penalty keeps the model from drifting too far from the SFT baseline — without it, the model will "reward-hack" by producing gibberish that fools the reward model.

The whole pipeline works, but it's fragile. PPO is notoriously finicky to train. You're also introducing a *second* model (the reward model) that can itself be fooled.

## DPO: the shortcut

Direct Preference Optimization (DPO), proposed in 2023, skips the reward model entirely. It reformulates the RL objective so you can fine-tune the language model *directly* on preference pairs — no PPO, no separate reward model, no reward-score instability.

The math shows that the optimal policy under the RLHF objective has a closed form that depends only on the ratio of probabilities between the fine-tuned model and the reference model. DPO exploits that to turn alignment into a simple classification loss over `(chosen, rejected)` pairs.

In practice: DPO trains faster, is far more stable, and needs less GPU memory than PPO-based RLHF. Most open-source fine-tuners (TRL, LLaMA-Factory, Axolotl) have first-class DPO support. For most teams — especially without a dedicated RL researcher — DPO is the default choice today.

```python {title="Minimal DPO training with TRL" run=false}
# pip install trl transformers datasets
# Free to run on Colab with a small model (e.g. Qwen-0.5B)

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct"  # swap for any SFT-tuned base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Each row: one prompt, one "good" response, one "bad" response
dataset = Dataset.from_dict({
    "prompt": [
        "What is the boiling point of water at 5000m altitude?",
    ],
    "chosen": [
        "At ~5000m, atmospheric pressure is roughly 0.54 atm, "
        "so water boils at about 83°C — not 100°C.",
    ],
    "rejected": [
        "Water always boils at 100°C regardless of altitude.",
    ],
})

config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    beta=0.1,  # KL penalty weight — lower = closer to reference model
    learning_rate=5e-5,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # TRL will create a frozen copy automatically
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

:::table {title="RLHF vs DPO at a glance"}
| Dimension | RLHF (PPO) | DPO |
|---|---|---|
| Requires reward model | Yes — separate model | No |
| Training stability | Low — PPO is finicky | High |
| GPU memory overhead | High (3 models in memory) | Lower (2 models) |
| Data needed | Preference pairs + RL rollouts | Preference pairs only |
| Common libraries | trl PPOTrainer | trl DPOTrainer |
| Best for | Very large scale, custom rewards | Most fine-tuning teams |
:::

## Why "helpfulness only" teaches models to hallucinate

Here's the core trap: human raters — and real users — often *cannot tell if an answer is wrong*. A fluent, confident wrong answer feels more helpful than a hedged correct one. So a reward signal trained on human approval ends up rewarding *confidence*, not *accuracy*.

The model learns a few bad habits:

- **Sycophancy** — If the user's prompt implies a wrong belief ("Vaccines contain microchips, right?"), agreeing with them scores higher than correcting them. The model learns to agree.
- **Confabulation** — When the model doesn't know something, expressing uncertainty scores lower than making something up fluently. So it makes things up.
- **Verbose fluff** — Long, structured answers with bullet points and headers feel more authoritative. The model pads responses to seem thorough.

The fix is not just DPO — it's *better preference data*. If your human raters are told to prefer accurate-and-hedged over confident-and-wrong, the alignment signal improves. Constitutional AI (Anthropic's approach) takes this further: it uses an AI critique-and-revision loop to generate preference data from a written "constitution" of principles, reducing dependence on expensive human labelling at scale.

:::gotcha
The KL-divergence penalty in both RLHF and DPO is not optional decoration. Without it, the model drifts into degenerate outputs — repetitive tokens, extreme verbosity, or gibberish — that score high on the reward model but are useless to users. If your DPO-trained model produces odd repetitive outputs, check that `beta` is not set too close to zero. A value of 0.1–0.3 is a reasonable starting range.
:::

:::interview-line
"Helpfulness-only rewards teach models that confident wrong answers beat uncertain right ones — alignment is about closing that gap between 'what the metric rewards' and 'what you actually want.'"
:::

:::qa {q="What problem does DPO solve compared to RLHF with PPO?"}
DPO eliminates the need for a separate reward model and the brittle PPO training loop. It directly optimises the language model on (chosen, rejected) preference pairs by leveraging the mathematical equivalence between the RL objective and a classification loss. In practice this means faster, more stable training with lower memory overhead — making alignment accessible to teams without RL specialists.
:::

:::qa {q="Why do helpfulness-only reward signals lead to hallucinations?"}
Because users often cannot distinguish confident-wrong from hedged-right, a reward model trained on human approval ends up rewarding fluency and confidence rather than accuracy. The language model then learns that expressing certainty (even without knowledge) scores higher than saying "I'm not sure." The structural fix is preference data that explicitly rewards epistemic honesty — for example, raters who are domain experts, or synthetic preference data generated from a principled constitution.
:::

:::qa {q="What is the KL-divergence penalty in RLHF/DPO and why is it needed?"}
The KL penalty measures how far the current model's output distribution has shifted from the reference (SFT baseline) model. It acts as a regulariser: without it, the model will "reward-hack" by producing outputs that fool the reward function while being useless or nonsensical to real users. In DPO, the `beta` hyperparameter controls the strength of this penalty.
:::

:::drill {type="mcq" q="A model fine-tuned with RLHF starts producing very long, repetitive, slightly incoherent responses that nonetheless get high reward scores. What is the most likely cause?"}
- [ ] The preference dataset was too small
- [ ] DPO beta was set too high
- [x] The KL-divergence penalty is too weak, allowing reward hacking
- [ ] The reward model architecture is too large
:::

:::drill {type="mcq" q="Your team has preference data (prompt, chosen, rejected) and one GPU node with 40 GB VRAM. You want to do alignment fine-tuning with minimum engineering complexity. Which approach is most appropriate?"}
- [ ] RLHF with PPO using three model replicas
- [x] DPO with TRL's DPOTrainer, possibly combined with LoRA/QLoRA
- [ ] Supervised fine-tuning only, ignoring the preference data
- [ ] Train a reward model first, then run PPO
:::

:::key-takeaway
Alignment is what turns a model that can follow instructions into one that won't lie to seem helpful. DPO gives you most of the benefit of RLHF with far less complexity — but the real lever is preference data that rewards honesty, not just user satisfaction.
:::
