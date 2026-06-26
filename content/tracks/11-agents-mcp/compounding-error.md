---
id: compounding-error
track: 11-agents-mcp
title: "The compounding-error math (0.95¹⁰ ≈ 60%)"
badge: HOT
minutes: 8
prereqs: []
tags: [agents, reliability, probability, production, math, llm]
xp: 60
hot2026: true
---

Your teammate just shipped a ten-step agent: fetch Jira tickets → summarise each one → pick sub-tasks → call an internal tool → validate output → reformat → write to Notion → post on Slack — eight-ish steps, more or less. Every step passes its unit test. You both demo it on Friday, it's flawless.

Three weeks later ops is livid. The agent "just works" barely 60% of the time. Nobody touched the code. Every individual step is still 95% reliable. **What went wrong?**

Nothing went wrong. Probability happened.

## The one number worth tattooing on your brain

When n independent steps each succeed with probability p, the whole pipeline succeeds with:

> **P(pipeline) = p ^ n**

Substitute the feel-good numbers: `0.95 ^ 10 = 0.5987`. Round it — **~60% end-to-end success** when every step is "almost perfect."

The model is reliable. The tools are reliable. The *composition* is the trap. You are multiplying probabilities, not averaging them.

:::why-prod
Real production agents rarely stop at three steps. A realistic workflow — fetch context, plan sub-tasks, call tool A, validate, call tool B, reformat, write DB, notify — hits 10–20 steps easily. A 60% success rate means your agent silently fails every second job. At 90% per step over 20 steps, you're down to 12%. That is not a product. That is an on-call nightmare.
:::

## The math in your hands

```python {title="compounding_error.py" run=false}
# No dependencies — pure stdlib. Run: python compounding_error.py

def pipeline_success(step_accuracy: float, n_steps: int) -> float:
    """
    Probability ALL n steps succeed, assuming independence.
    Each step either succeeds with p or fails with (1 - p).
    """
    return step_accuracy ** n_steps


configs = [
    (0.99, 5),
    (0.99, 20),
    (0.95, 5),
    (0.95, 10),   # <-- the "that's fine" scenario that isn't
    (0.95, 20),
    (0.90, 10),
    (0.90, 20),
]

print(f"{'accuracy':>10}  {'steps':>6}  {'pipeline %':>12}")
print("-" * 34)
for acc, steps in configs:
    ps = pipeline_success(acc, steps)
    print(f"{acc:>10.0%}  {steps:>6}  {ps:>11.1%}")

# Key insight:
# Halving the number of steps has a bigger impact than improving accuracy per step.
# 0.95^5  ≈ 77%   vs   0.95^10 ≈ 60%   (same accuracy, fewer steps → big win)
# 0.99^10 ≈ 90%   vs   0.95^10 ≈ 60%   (better accuracy helps, but costs effort)
```

:::table {title="End-to-end success rate by step count and per-step accuracy"}
| Per-step accuracy | 3 steps | 5 steps | 10 steps | 20 steps |
|---|---|---|---|---|
| 99% | 97% | 95% | 90% | 82% |
| 95% | 86% | 77% | **60%** | 36% |
| 90% | 73% | 59% | 35% | 12% |
| 80% | 51% | 33% | 11% | 1% |
:::

Read the 90% / 20-step cell again. **One percent.** At that point the agent is theater, not engineering.

## What you can actually do

**Minimize steps first.** This is the highest-leverage move because you are changing the exponent. Every step you eliminate roughly doubles reliability. Ask: does the LLM need to validate *and then* reformat, or can one well-structured prompt do both?

**Raise per-step accuracy second.** Better prompts, typed tool schemas, structured output (JSON mode, Pydantic). A jump from 90% → 95% per step at n=10 takes pipeline success from 35% → 60% — nearly double the yield for the same chain length.

**Add idempotent retries with a budget.** A step that can be safely retried at 95% base accuracy needs to fail twice in a row to count as a failure: 0.05 × 0.05 = 0.0025. That is 99.75% effective accuracy for that step. Retries are powerful — but cap them hard. Uncapped retries compound your latency and API spend instead of your failures.

**Add checkpoints at milestone boundaries.** Validate the state of the world after expensive or irreversible steps. Fail fast and loud rather than silently propagating bad state five steps forward.

:::gotcha
Engineers almost always optimise the *last* step because that is where the visible error surfaces. But the last step is often just a messenger. The real corruption happened two or three steps upstream — a silently truncated API response, a mis-parsed date, a hallucinated field name. Always trace errors to the root tool call, not the output formatter. Your checkpoints should catch it early.
:::

:::war-story {title="The Slack bot that silently wrecked three sprint reviews"}
A startup's nightly agent read Jira, drafted a sprint summary, updated Notion, and posted to Slack — eight steps, all tested, all demo-perfect. In production after four weeks, only 44% of runs completed. Investigation revealed step 3 (Jira pagination) silently returned partial data 25% of the time. Downstream steps never threw an exception — they just processed incomplete input. The Slack messages looked fine. The sprint summaries were quietly wrong. Nobody noticed for three weeks. The fix was a two-line assertion after step 3. The lesson: **silence is not success.** Checkpoints are not optional.
:::

:::widget {name="agenterror"}
:::

:::interview-line
"A ten-step agent where each step is 95% accurate only succeeds 60% of the time — so I design agents to minimise steps first, add idempotent retries second, and checkpoint every milestone."
:::

:::qa {q="Why does a multi-step agent fail even when every individual step looks reliable?"}
Reliability compounds multiplicatively. If each of n steps has probability p of success, the whole pipeline succeeds with probability p^n. Ten steps at 95% each yields about 60% end-to-end success — not 95%. The pipeline is always weaker than its weakest link repeated n times.
:::

:::qa {q="What is the single most effective way to improve agent pipeline reliability?"}
Reduce the number of steps — that changes the exponent directly, which has the largest impact. After that, raise per-step accuracy through better prompts and constrained schemas, and add idempotent retries with a hard budget. Checkpoints that fail fast prevent bad state from propagating silently.
:::

:::qa {q="How does one safe retry change the effective failure rate of a step at 95% accuracy?"}
Without a retry the failure rate is 5% (0.05). With one retry both attempts must fail: 0.05 × 0.05 = 0.0025, a 0.25% failure rate. Effective accuracy jumps from 95% to 99.75%. The key constraint is that the step must be idempotent — retrying a non-idempotent operation (like writing to a DB without a transaction) can cause data corruption.
:::

:::drill {type="mcq" q="An agent has 5 independent steps, each with a 90% success rate. What is the approximate end-to-end success rate?"}
- [ ] 90%
- [ ] 72%
- [x] 59%
- [ ] 45%
:::

:::drill {type="mcq" q="You have a 10-step agent pipeline running at 95% per step (~60% end-to-end). Which change gives the biggest reliability gain?"}
- [ ] Tune only the final output formatter from 95% to 99%
- [x] Redesign the workflow to require only 5 steps at 95% each
- [ ] Add verbose logging to every step
- [ ] Increase LLM temperature for more creative outputs
:::

:::drill {type="mcq" q="A step in your agent has a 20% failure rate. You add one safe idempotent retry. What is the new approximate effective failure rate?"}
- [ ] 10%
- [ ] 5%
- [x] 4%
- [ ] 0.4%
:::

:::key-takeaway
Every step you add to an agent pipeline multiplies your failure surface. 0.95¹⁰ ≈ 60% — keep chains short, retries budgeted, and checkpoints loud.
:::
