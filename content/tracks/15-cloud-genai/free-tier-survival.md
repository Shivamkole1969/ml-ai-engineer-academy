---
id: free-tier-survival
track: 15-cloud-genai
title: "Cost & free-tier survival guide (practice each platform for ₹0)"
badge: CORE
minutes: 7
prereqs: []
tags: [aws, gcp, snowflake, cost, free-tier, bedrock, vertex-ai, sagemaker, bigquery]
xp: 45
hot2026: false
---

It's 11 PM. You've been hacking on a Vertex AI notebook for two hours, the experiment finally works, and you close your laptop. Thirty-six hours later you check GCP Billing — and there's a $47 charge for a notebook VM that kept running while you slept, ate, commuted, and slept again.

That cloud tax hits every engineer exactly once. After that, they learn the rules. This lesson gives you those rules upfront so your practice across five platforms costs you ₹0.

## What Each Platform Actually Gives You Free

Every platform tells a different free-tier story. Know it before you type your first API key.

:::why-prod
Unchecked cloud spend is a real career event. Teams in Pune get billed in USD — a forgotten ml.g4dn.xlarge left running for a weekend is ₹30,000+ before your Monday standup. Learning cost hygiene on your own dime, before you ever touch a company account, is a genuine professional advantage.
:::

:::table {title="Free-tier cheatsheet: Cloud GenAI platforms"}
| Platform | What's Free | The Catch |
|---|---|---|
| **Gemini API (AI Studio)** | 15 req/min, 1M tokens/min on Flash — genuinely free | Rate-limited; no SLA; not for prod |
| **AWS Bedrock** | No free tier — pay-per-token from call one | Playground is cheap; Haiku ~$0.00025/1K input tokens |
| **AWS SageMaker** | 2-month intro: 250 h Studio (t3.medium) + 125 h hosting | Studio charges even when your kernel sits idle |
| **GCP Vertex AI** | $300 free credit, 90 days, new accounts only | Expires; set a budget alert on day one |
| **BigQuery ML** | 10 GB storage + 1 TB queries/month forever | BQML PREDICT counts against your query bytes |
| **Snowflake** | 30-day trial, $400 Snowflake credits | Warehouse auto-resumes on any query — burns credits overnight |
:::

## Six Survival Rules

**Rule 1 — Set a billing alert before you write a single line of code.**

On AWS: Billing → Budgets → Create budget → $5 threshold → email yourself. On GCP: Billing → Budgets & alerts → $10. Do this on day zero. Not day two.

**Rule 2 — Shut down compute from the console, not from your browser tab.**

Closing a tab does nothing to the running VM behind it. SageMaker Studio: stop the kernel AND stop the instance. Vertex AI Workbench: go to the console and click Stop. This habit alone prevents 90 % of surprise bills.

**Rule 3 — Practice in the right order to maximise free runway.**

Start with **Gemini API via AI Studio** — free, instant, no billing setup. Then **BigQuery ML** — SQL you already know, 1 TB free queries per month. Then **Snowflake trial** — $400 credits last weeks on an XS warehouse. Save AWS for last; it requires a credit card and has no Bedrock free tier.

**Rule 4 — Estimate before you execute.**

Ten thousand rows through Claude 3 Haiku costs roughly $0.90. The same job through Sonnet is $13.50. That's a 15× difference. Estimate first:

```python {title="Quick Bedrock cost estimator — run locally before touching the API" run=false}
# python estimate_bedrock.py  — no cloud account needed for this step
# Prices as of mid-2025; always verify at https://aws.amazon.com/bedrock/pricing/

PRICES = {
    "claude-3-haiku":  {"input": 0.00025, "output": 0.00125},  # per 1 K tokens
    "claude-3-sonnet": {"input": 0.003,   "output": 0.015},
    "claude-3-opus":   {"input": 0.015,   "output": 0.075},
}

def estimate_usd(
    model: str,
    num_requests: int,
    avg_input_tokens: int = 300,
    avg_output_tokens: int = 150,
) -> float:
    p = PRICES[model]
    return (
        num_requests * avg_input_tokens  / 1000 * p["input"]
        + num_requests * avg_output_tokens / 1000 * p["output"]
    )

for model in PRICES:
    cost = estimate_usd(model, num_requests=10_000)
    print(f"{model:<22}  ${cost:>6.2f}")

# claude-3-haiku          $  0.94
# claude-3-sonnet         $ 13.50
# claude-3-opus           $ 67.50
```

**Rule 5 — Use the smallest compute tier for development.**

Snowflake XS warehouse = 1 credit/hour. XL = 16 credits/hour. Your dev SQL does not need XL. Same on SageMaker: ml.t3.medium for notebooks; only upgrade when you're actually training.

**Rule 6 — Auto-suspend Snowflake warehouses.**

```sql {title="Auto-suspend any dev warehouse — paste this right after CREATE WAREHOUSE" run=false}
-- Run in Snowflake Worksheets (free trial, no setup needed)
ALTER WAREHOUSE my_dev_wh
  SET AUTO_SUSPEND = 60    -- suspend after 60 seconds idle
      AUTO_RESUME  = TRUE; -- resume automatically on next query
```

This one line saves the trial credits of most engineers who forget to click "Suspend".

:::gotcha
GCP Vertex AI Workbench does NOT auto-stop instances — not after an hour, not after a day. If you use Workbench for experiments, set a phone reminder to stop it when you're done. Alternatively, use Colab Enterprise for short experiments; it manages runtimes automatically and counts against your $300 credit only while actively running.
:::

:::interview-line
"Before any cloud experiment I set a budget alert and run a cost estimate — I pick the model tier after seeing the number, not before."
:::

:::qa {q="How would you keep GenAI experimentation costs under control for a junior team?"}
Set account-level budget alerts on day zero — $5–$10 threshold, email notification. Establish a rule: estimate cost before running any batch job (a one-page script like the one above takes two minutes). Use Gemini API free tier and BigQuery ML's free quota for early prototypes. Require the estimated cost to appear in the PR description so reviewers can flag outliers before they run.
:::

:::qa {q="Which AWS service has a genuine free tier for GenAI inference?"}
Neither Bedrock nor SageMaker real-time endpoints offer a permanent free inference tier. SageMaker has a 2-month introductory free tier covering Studio notebook hours and some hosting hours, but Bedrock charges per token from the very first API call. For zero-cost inference experimentation, use Gemini API via Google AI Studio or BigQuery ML's free monthly query quota.
:::

:::drill {type="mcq" q="You leave a SageMaker Studio notebook open overnight with the kernel idle. What happens to your bill?"}
- [ ] Nothing — SageMaker only charges when the kernel is actively running code
- [x] The underlying notebook instance keeps running and you are charged for compute time
- [ ] SageMaker auto-stops notebook instances after 1 hour of inactivity by default
- [ ] You are only charged for storage, not compute, when the kernel is idle
:::

:::drill {type="mcq" q="Which platform sequencing minimises spend while building a cloud GenAI portfolio?"}
- [ ] Start with Bedrock (pay-per-token is cheap enough) → Vertex AI → Snowflake
- [ ] Start with Snowflake trial → SageMaker → BigQuery ML → Gemini
- [x] Start with Gemini API free tier → BigQuery ML free quota → Snowflake trial → AWS last
- [ ] All platforms have equivalent free tiers — order does not matter
:::

:::key-takeaway
Set a billing alert before writing any cloud code. Use Gemini API, BigQuery ML, and Snowflake trial for free practice. Shut compute down from the console — not from your browser tab — and always estimate cost before running a batch job.
:::
