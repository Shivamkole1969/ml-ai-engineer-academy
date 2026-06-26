---
id: eval-finetune
track: 10-fine-tuning
title: "Evaluating a Fine-Tune: Forgetting, Merging, and Swappable LoRAs"
badge: HOT
minutes: 10
prereqs: []
tags: [fine-tuning, lora, evaluation, catastrophic-forgetting, model-merging, peft]
xp: 60
hot2026: true
---

You spend three days fine-tuning Mistral-7B on your company's internal SQL dialect. Loss curve looks clean. You eval on the held-out SQL set — 91% accuracy. QA signs off. It ships on Friday.

Monday morning, Slack lights up. Users are asking the assistant to help draft emails. It's refusing. "I'm only able to answer SQL questions," it says, confidently hallucinating a policy that was never in your training data. Your model forgot it was a language model. Welcome to catastrophic forgetting.

## The Evaluation Trap

Most teams make the same mistake: they only eval on the task they trained for.

That's necessary — but not sufficient. A solid eval suite has two layers.

**Layer 1 — Task eval.** Does it do *your* task well? Use a held-out set in the same format as your training data. This is the obvious part.

**Layer 2 — General capability check.** Did it stay sane? Run a small benchmark sweep. You don't need full MMLU. A 200-question sample across diverse categories takes four minutes and catches regressions before they reach users.

:::why-prod
In production, users ask the model things you never expected. A fine-tune that nails your task but breaks on "what year is it?" will destroy trust fast — and it's invisible unless you specifically look for it. Layered evals are cheap insurance.
:::

:::table {title="Eval dimensions for every fine-tune"}
| Dimension | What to check | Cheap tool |
|---|---|---|
| Task performance | Accuracy / F1 / ROUGE on held-out set | Custom eval script |
| Instruction following | Does it respect format prompts? | LLM-as-judge (GPT-4 / Claude) |
| General knowledge | No regression on open-domain Q&A | lm-eval MMLU 200-q sample |
| Safety / refusals | No new jailbreaks introduced | MT-Bench safety subset |
| Latency | Adapter overhead is acceptable | time your inference loop |
:::

## Catastrophic Forgetting

Fine-tuning shifts the model's weights toward your task. If the learning rate is high or the dataset is narrow, it over-shifts — crushing old capabilities to carve out space for new ones.

LoRA is a partial defence. Because you're only training low-rank adapter matrices (not the full weights), the base model's general knowledge stays largely intact. But it's not a free pass. A high rank, high learning rate, or tiny dataset can still cause forgetting even with LoRA.

**How to detect it:** Run a 100–200 question general eval on the base model before training, then again after. A drop of more than 2–3 accuracy points is a red flag. The fix: lower your learning rate or your LoRA rank, then retrain.

:::gotcha
Don't assume LoRA = no forgetting. If you set `r=128` and train for five epochs on 500 examples, you'll still overfit and forget. Keep rank low (8–32 for most tasks), use a cosine LR schedule, and always measure general capability before and after. The measurement takes four minutes. The consequences of skipping it take three days to untangle.
:::

## Merging: Two Models, One Inference Cost

Your SQL adapter is great. Your colleague's code-explanation adapter is also great. Can you get both capabilities without running two models? Yes — via model merging.

The popular approach is **TIES** (Task Interference via Sign Election) or **DARE**, both available in the open-source [`mergekit`](https://github.com/arcee-ai/mergekit) library. You merge the adapter weight *deltas* using task vectors, resolving conflicts where weights point in opposite directions across the two fine-tunes.

The result: one model with capabilities from both fine-tunes, zero extra GPU memory at inference.

```python {title="Merge two LoRA adapters with mergekit (CLI config)" run=false}
# pip install mergekit
# Step 1: merge each LoRA adapter into its base to get full model checkpoints
#   from peft import PeftModel
#   peft_model = PeftModel.from_pretrained(base, "adapters/sql-adapter")
#   merged = peft_model.merge_and_unload()
#   merged.save_pretrained("merged_sql")
#   (repeat for merged_codeexplain)

# Step 2: create merge_config.yaml
# ----------------------------------------
# merge_method: ties
# base_model: mistralai/Mistral-7B-v0.1
# dtype: bfloat16
# models:
#   - model: merged_sql
#     parameters:
#       weight: 0.5
#       density: 0.5
#   - model: merged_codeexplain
#     parameters:
#       weight: 0.5
#       density: 0.5
# ----------------------------------------

# Step 3: run the merge (needs a GPU or ~32GB RAM)
# mergekit-yaml merge_config.yaml ./merged_output --cuda

# Load the result like any HuggingFace checkpoint — no extra inference overhead
```

## Swappable LoRAs: One GPU, Many Personalities

Here's the production trick most teams overlook. You can run one base model on one GPU and hot-swap different LoRA adapters per request. This is how you deploy five "fine-tuned models" for the cost of one.

Each adapter is just a few MB of weights. The base model — gigabytes — never moves. You get near-zero switching latency.

```python {title="Hot-swap LoRA adapters at inference time (PEFT)" run=false}
# pip install peft transformers accelerate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model ONCE — stays in GPU memory the whole time
base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Attach the first adapter and register it with a name
model = PeftModel.from_pretrained(base, "adapters/sql-adapter", adapter_name="sql")

# Load a second adapter — base weights do NOT reload, just the tiny adapter
model.load_adapter("adapters/email-adapter", adapter_name="email")

# At request time: swap in under 1ms
model.set_adapter("sql")
inputs = tokenizer("SELECT all orders from...", return_tensors="pt").to("cuda")
print(model.generate(**inputs, max_new_tokens=128))

model.set_adapter("email")
inputs = tokenizer("Draft a follow-up email to...", return_tensors="pt").to("cuda")
print(model.generate(**inputs, max_new_tokens=128))
```

:::war-story {title="The 3AM Rollback That Didn't Happen"}
A fintech team in Bangalore fine-tuned a 13B model for loan eligibility Q&A. They shipped it, and three days later found it giving confidently wrong answers to basic arithmetic questions users dropped mid-conversation. Root cause: they'd only evaluated on loan Q&A data. General capability had regressed eight points on MMLU. They rolled back at 3AM, retrained with a lower learning rate, and added a 200-question general eval to their CI pipeline. The check now runs in four minutes and has caught two regressions since. No more 3AM Slack calls.
:::

:::interview-line
"We always run a 200-question MMLU sample before and after fine-tuning — it's a cheap regression test that has caught every forgetting issue we've seen in production."
:::

:::qa {q="What is catastrophic forgetting and how does LoRA help?"}
Fine-tuning shifts model weights toward a new task, potentially overwriting old knowledge. LoRA mitigates this by training only small low-rank adapter matrices while keeping base weights frozen — so general capabilities stay largely intact. You still need to measure it; high rank or high learning rate can cause forgetting even with LoRA.
:::

:::qa {q="How would you serve five different fine-tunes without five GPUs?"}
Load one base model and attach multiple named LoRA adapters using PEFT's `load_adapter` and `set_adapter` APIs. Each adapter is a few MB; the base model stays resident on one GPU. You swap adapters per request with negligible latency. This is standard practice for cost-efficient multi-task serving in production.
:::

:::qa {q="When would you merge two LoRA fine-tunes instead of running them as swappable adapters?"}
When both capabilities need to be active simultaneously in the same response, or when you want to simplify the serving stack to a single checkpoint. Merging with TIES or DARE combines the weight deltas into one model — both fine-tunes baked in, no extra inference cost — but you lose the ability to turn one off independently.
:::

:::drill {type="mcq" q="Your fine-tune scores 94% on your task eval but drops 9 points on MMLU vs. the base model. What is the most likely cause?"}
- [ ] The base model was undertrained
- [x] Catastrophic forgetting caused by over-shifting weights during fine-tuning
- [ ] MMLU is incompatible with fine-tuned models
- [ ] The tokenizer changed between training runs
:::

:::drill {type="mcq" q="You want to deploy a SQL assistant and an email assistant sharing one A100. Which approach is cheapest at inference time?"}
- [ ] Run two separate full model replicas and load-balance between them
- [ ] Fine-tune one combined model on both tasks simultaneously
- [x] Use a single base model with two named LoRA adapters swapped via PEFT's set_adapter
- [ ] Use prompt engineering only — no fine-tuning or adapters needed
:::

:::drill {type="mcq" q="What does mergekit's TIES merge method primarily resolve?"}
- [ ] Tokenizer conflicts between two different base models
- [ ] Learning rate differences between two training runs
- [x] Sign conflicts between task vectors from two fine-tunes
- [ ] GPU memory fragmentation during inference
:::

:::key-takeaway
Always evaluate your fine-tune on both your specific task *and* a general capability sample — catastrophic forgetting is silent and only shows up when users ask unexpected questions. Use swappable LoRA adapters to serve multiple fine-tunes on one GPU, and reach for model merging when you need capabilities combined in a single inference pass.
:::
