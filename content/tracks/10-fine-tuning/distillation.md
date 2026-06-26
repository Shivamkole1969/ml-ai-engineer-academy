---
id: distillation
track: 10-fine-tuning
title: "Distillation as adaptation (student/teacher)"
badge: CORE
minutes: 8
prereqs: []
tags: [distillation, knowledge-distillation, student-teacher, soft-labels, model-compression, kl-divergence]
xp: 45
hot2026: false
---

Your team spent three weeks fine-tuning a 70B Llama model. It's brilliant — it answers customer queries with exactly the right tone, never hallucinates your product names, and the PM loves it. Then infrastructure sends a Slack: "₹2.50 per 1K tokens, P95 latency 6 seconds. Ship it to the mobile app." The dream dies a little.

Distillation is how you resurrect it. You take that genius 70B (the **teacher**) and train a scrappy 7B (the **student**) to *think like it* — not just mimic its final answers, but absorb its entire probability worldview. You get a model that's 10× cheaper to serve and half a second to respond, while keeping most of the teacher's hard-won knowledge.

## The core idea: more than copying answers

The naive approach: run the teacher on all your data, collect its text outputs, SFT the student on those outputs. This is called **sequence-level distillation** and it works. It's essentially SFT with teacher-generated labels instead of human-written ones.

But there's a more powerful idea hiding in the teacher's raw logits.

When your teacher says "Paris", it doesn't just think Paris=1 and everything else=0. It assigns probabilities across the entire vocabulary. Maybe: Paris=0.72, France=0.11, Lyon=0.06, Brussels=0.04 …

That long tail of near-zero probabilities is **dark knowledge** — the teacher's implicit sense of what's related, what's plausible, what it's uncertain about. A hard label (just "Paris") throws all that away. Soft labels preserve it. And it turns out the student learns *much* faster from soft labels than from hard ones, because each token carries a richer training signal.

:::why-prod
In production you're always trading model quality for serving cost. Distillation lets you make that trade surgically — you control exactly how much teacher signal you inject — instead of just grabbing a smaller pretrained model and hoping for the best.
:::

## Temperature and soft labels

To make soft labels even more informative, you scale the logits by a **temperature T** before taking the softmax. Higher T flattens the distribution, making the differences between candidate tokens more visible.

At T=1 the teacher says: Paris=0.72, France=0.11 (the rest are noise).
At T=4 the teacher says: Paris=0.41, France=0.22, Lyon=0.14 (the relationships are clearer).

You apply the *same* temperature to the student during distillation training. After training is done, you discard T and serve the student at T=1 like normal.

:::table {title="Three flavours of distillation"}
| Approach | What the student learns from | When to use |
|---|---|---|
| Sequence-level KD | Teacher's generated text (SFT on teacher outputs) | Easiest to implement; good first baseline |
| Token-level KD | Per-token probability distribution (KL loss) | Better quality; needs access to teacher logits |
| Feature / attention KD | Intermediate hidden states or attention maps | Squeezes the most signal; architectures must be compatible |
:::

## The distillation loss

For token-level KD, the training loss has two terms mixed by a coefficient α:

```python {title="Token-level distillation loss (minimal, runnable)" run=false}
import torch
import torch.nn.functional as F

# Run locally: pip install torch transformers
# Teacher logits come from a frozen model; student logits are computed fresh each step.

def distillation_loss(
    student_logits: torch.Tensor,   # (batch, seq_len, vocab)
    teacher_logits: torch.Tensor,   # (batch, seq_len, vocab) — frozen, no grad
    labels: torch.Tensor,           # (batch, seq_len) — ground-truth token ids
    temperature: float = 4.0,
    alpha: float = 0.7,             # weight on the soft/teacher loss
) -> torch.Tensor:
    # --- Soft loss: student learns from teacher's distribution ---
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    # KL divergence: how far is student from teacher?
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    # Scale back up: temperature² restores gradient magnitudes (Hinton et al.)
    soft_loss = kl * (temperature ** 2)

    # --- Hard loss: student still trains on ground-truth labels ---
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

The `temperature**2` scaling is easy to forget but critical — it keeps the gradient magnitudes comparable regardless of the temperature you pick.

## How distillation fits alongside fine-tuning

These two techniques compose, they don't compete.

A common production pattern: fine-tune the big teacher on your task first (SFT, maybe DPO), then distill that tuned teacher into a smaller student. The student learns both the general "think like the teacher" signal and the task-specific behavior that was baked in during fine-tuning.

You can also distil a proprietary API model — collect outputs from GPT-4o or Claude, use them as labels. Many of the capable open-source 7B models in the wild were trained partly this way. (Check the model's licence; some teachers explicitly prohibit distillation.)

:::gotcha
Distillation is not free compression. If your student architecture is simply too small to hold the information, KL loss will converge to a floor and stay there — you'll get a model that's learned to look confident without actually understanding. Watch your validation loss curve: if soft loss plateaus while hard loss keeps dropping, the student can't absorb what the teacher is sending. Try a larger student or feature-matching on intermediate layers.
:::

## Practical checklist before you distil

1. **Decide your budget.** What latency and cost target must the student hit? That sets the size ceiling.
2. **Pick your flavour.** Sequence-level if you can't access teacher logits (API-only teacher). Token-level if you run the teacher yourself.
3. **Tune α and T on a small slice.** α=0.5–0.8, T=2–6 are common starting points. Grid-search cheaply.
4. **Evaluate on your real eval set**, not generic benchmarks. Distillation often preserves task-specific gains better than random pruning, but you must verify.
5. **Check the licence.** OpenAI and Anthropic terms restrict using their outputs to train competing models.

:::interview-line
"Distillation lets the student learn from the teacher's full probability distribution, not just its final answer — that soft-label signal trains faster and generalises better than hard labels alone."
:::

:::qa {q="What is the difference between sequence-level and token-level knowledge distillation?"}
Sequence-level KD runs the teacher to generate text, then trains the student on that text with standard cross-entropy — it's basically SFT with teacher-written data. Token-level KD matches the student's per-token probability distribution to the teacher's at every position using KL divergence, which gives a denser training signal and typically better results when you have access to the teacher's raw logits.
:::

:::qa {q="Why do you multiply the KL loss by T² in Hinton's distillation formula?"}
Temperature scaling divides the logits by T before softmax, which shrinks the gradients by a factor of 1/T². Multiplying the loss by T² compensates for that shrinkage so the soft-loss and hard-loss terms contribute gradients of comparable magnitude, making the α hyperparameter easier to tune and stable across different temperature choices.
:::

:::qa {q="Can you distil a closed-source model like GPT-4 into your own model?"}
Technically you can collect outputs and train on them (sequence-level KD). In practice, OpenAI's terms of service prohibit using their outputs to train a competing model. Anthropic's terms have similar language. Always read the licence. Many teams distil from their own fine-tuned large open-weight models (Llama 3 70B → 8B) where there is no such restriction.
:::

:::drill {type="mcq" q="What is the primary advantage of training a student on soft labels rather than hard one-hot labels?"}
- [ ] Soft labels reduce GPU memory usage during training
- [x] Soft labels encode the teacher's full probability distribution, providing richer inter-class relationship signals per token
- [ ] Soft labels remove the need for a ground-truth dataset entirely
- [ ] Soft labels allow the student to skip the tokenization step
:::

:::drill {type="mcq" q="You run distillation with T=5 and after convergence the student still makes different token choices than the teacher on 30% of positions. What is the most likely explanation?"}
- [ ] The temperature is too low; raise it above 10
- [ ] α is set too high, overpowering the hard-label loss
- [x] The student model is too small to capture the teacher's distribution — you've hit the capacity ceiling
- [ ] The T² scaling factor was applied twice, causing gradient explosion
:::

:::drill {type="mcq" q="Which distillation variant requires direct access to the teacher model's logits rather than just its generated text?"}
- [ ] Sequence-level knowledge distillation
- [ ] Synthetic data SFT
- [x] Token-level KL-divergence distillation
- [ ] Hard-label transfer learning
:::

:::key-takeaway
Distillation transfers a large model's probabilistic reasoning — not just its answers — into a smaller, deployable student by training on soft probability distributions instead of hard labels. The result is a model that punches above its weight class in production.
:::
