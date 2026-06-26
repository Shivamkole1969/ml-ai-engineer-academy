---
id: ab-traps
track: 05-evaluation
title: "A/B testing traps (peeking, novelty, interference, SRM)"
badge: CORE
minutes: 10
prereqs: []
tags: [evaluation, ab-testing, statistics, experimentation, mlops]
xp: 45
hot2026: false
---

It's Friday afternoon. Your new recommendation model is live in a 50/50 A/B split. By 6 PM you open the dashboard — treatment is up 14% on click-through rate. You Slack the team. Everyone celebrates. You call the winner, kill the control, and go home happy.

Monday morning: engagement is back to baseline. Your manager is asking questions you can't answer.

You didn't have a bad model. You had a bad experiment. Welcome to the four traps that silently invalidate A/B tests — and end careers if you don't spot them.

## Trap 1: Peeking

You ran your test for two days and stopped early because the numbers looked great. That's peeking.

The problem is statistical. Every time you check a live experiment and make a decision based on what you see, you inflate your false-positive rate. Run enough checks, and a noisy random walk will briefly cross your significance threshold by pure chance. You declare a winner that isn't real.

The fix is simple but requires discipline: pre-commit to a sample size, compute it before you start (use a power calculator — target 80% power, α = 0.05), and don't peek until you hit it. If you absolutely must monitor early, use sequential testing methods (like mSPRT or always-valid p-values) that are designed for it.

:::why-prod
A/B tests that stop early on positive signals systematically overestimate effect sizes. In production this means you'll ship features that look like +14% wins but deliver +2% or nothing — wasting engineering cycles and eroding trust in your evaluation process.
:::

## Trap 2: Novelty Effect

Users click on anything new. Always. Then they don't.

If your treatment group sees a shiny new UI, recommendation carousel, or model output format, the initial spike in engagement isn't your model's quality — it's human curiosity. Run the experiment long enough for novelty to wear off (usually 2–4 weeks for user-facing features) and the effect often shrinks dramatically or disappears.

The inverse is also true: a new model that replaces something familiar might see a short-term *dip* before users adapt. Novelty cuts both ways.

## Trap 3: Interference (SUTVA Violation)

A/B testing assumes the control and treatment groups don't affect each other. In many real systems, they do.

Classic example: you A/B test a new fraud-detection model. The treatment arm starts catching more fraud. Fraudsters adapt and shift their attacks to the control arm. Control starts looking worse — not because it got worse, but because the treatment changed the environment it's operating in. You declare a winner that's partially an artifact.

This happens constantly in social networks (your friend in treatment recommends something to you in control), marketplaces (pricing changes in treatment affect supply available to control), and any shared-resource system. The formal name is a SUTVA violation — Stable Unit Treatment Value Assumption.

Solutions: cluster randomization (randomize by household/city/cohort rather than individual user), or holdout at the network edge.

## Trap 4: Sample Ratio Mismatch (SRM)

You asked for a 50/50 split. You got 52/48. Small rounding error, right?

Not necessarily. SRM — when your actual user ratio doesn't match your intended ratio — is usually a sign something broke in your randomization or logging pipeline. Maybe one arm is dropping events. Maybe the redirect has different latency and some users bail before being logged. Maybe a bot filter is asymmetric.

SRM invalidates your experiment entirely. Any metric difference you observe might just be from the different populations, not your treatment.

Check for SRM before you look at any other metric. Run a chi-squared test on your observed vs expected counts. If p < 0.01, stop. Debug the plumbing first.

```python {title="SRM detector — run locally with scipy" run=false}
from scipy.stats import chi2_contingency
import numpy as np

# Replace with your observed counts
observed_control = 48312
observed_treatment = 51688
expected_ratio = 0.5  # intended 50/50 split

total = observed_control + observed_treatment
expected_control = total * expected_ratio
expected_treatment = total * (1 - expected_ratio)

# chi-squared test for goodness of fit
obs = np.array([observed_control, observed_treatment])
exp = np.array([expected_control, expected_treatment])

chi2 = np.sum((obs - exp) ** 2 / exp)
from scipy.stats import chi2 as chi2_dist
p_value = chi2_dist.sf(chi2, df=1)

print(f"Chi2: {chi2:.2f}, p-value: {p_value:.4f}")
if p_value < 0.01:
    print("SRM DETECTED — do not trust this experiment's results.")
else:
    print("No SRM detected. Split looks healthy.")

# pip install scipy  — free, standard library
```

:::table {title="The four A/B traps at a glance"}
| Trap | What it looks like | Root cause | Fix |
|---|---|---|---|
| Peeking | "We saw significance at Day 2!" | Multiple comparisons over time | Pre-commit sample size; use sequential testing |
| Novelty effect | Huge win that fades after Week 2 | Users click new things | Run for full novelty decay period (2–4 weeks) |
| Interference | Control degrades mid-experiment | Treatment changes the shared environment | Cluster randomization; holdout at edge |
| SRM | Ratio is 52/48 not 50/50 | Logging bug / asymmetric filtering | Chi-squared check before reading any metric |
:::

:::gotcha
The deadliest combination: you peek (Trap 1) and see a big win, which is actually novelty (Trap 2), and you never noticed the SRM (Trap 4) that made both arms non-comparable anyway. Each trap alone is survivable. Together they produce a completely fictional "result" with full statistical confidence. Always run your SRM check first, before you look at anything else.
:::

:::interview-line
"Before I read any metric from an A/B test, I run an SRM check — if the split isn't what I asked for, the whole experiment is suspect and I debug the logging pipeline first."
:::

:::qa {q="Why does peeking inflate false positives even if you use α=0.05?"}
Each additional peek is an additional hypothesis test. If you peek 20 times at 5% alpha, your true false-positive rate is closer to 64%, not 5%. The p-value threshold was calibrated for a single decision point, not a sequential decision process. To test sequentially you need methods like mSPRT or group sequential tests that account for multiple looks.
:::

:::qa {q="How do you detect and handle interference in a marketplace A/B test?"}
Interference shows up when one arm's behavior changes what's available to the other — e.g., treatment users winning more auctions leaves fewer impressions for control. Detection: compare hold-out (no-treatment) groups across geographies. Mitigation: randomize at the level that doesn't share resources — by city, device type, or time-based switchback rather than user-level. Geo-based experiments are the standard for supply-side tests.
:::

:::drill {type="mcq" q="You launch an A/B test targeting a 50/50 split. After 3 days you have 41,200 in control and 58,900 in treatment. What should you do first?"}
- [ ] Declare treatment the winner — bigger sample means more power
- [ ] Rebalance by randomly dropping treatment users until 50/50
- [x] Stop reading metrics and investigate why the split is wrong (SRM)
- [ ] Continue the test — small deviations are statistically normal
:::

:::drill {type="mcq" q="A social feed ranking model shows +18% engagement in week 1 of A/B test, then +3% in week 3. The most likely explanation is:"}
- [ ] Sample ratio mismatch shifting week over week
- [ ] The model is learning and getting worse over time
- [x] Novelty effect — users explored the new feed initially, then settled into normal behaviour
- [ ] Peeking — you should have stopped at week 1
:::

:::key-takeaway
Run the SRM check before you look at any other number. If the split is off, every metric is tainted. Then stay disciplined about pre-committed sample sizes — the hardest A/B skill isn't statistics, it's resisting the urge to peek.
:::
