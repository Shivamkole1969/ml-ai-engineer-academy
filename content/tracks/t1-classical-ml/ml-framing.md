---
id: ml-framing
track: t1-classical-ml
title: "Supervised / unsupervised / RL — with real examples"
badge: CORE
minutes: 9
prereqs: []
tags: [supervised, unsupervised, reinforcement-learning, ml-fundamentals, problem-framing]
xp: 45
hot2026: false
---

Imagine a product manager walks up to you and says: "We want to use AI for our churn problem." You nod, open your laptop — and then freeze. *What kind of ML does this even need?* You've seen supervised learning, unsupervised learning, and reinforcement learning mentioned in every course. But nobody told you the decision logic for choosing one over the other in a real meeting. That moment of paralysis? This lesson fixes it.

## The three flavors of machine learning

All of ML boils down to one question: **what signal do you have to learn from?**

### Supervised learning — learning from labeled examples

You have a dataset where each row has both an input (features) and the correct output (a label). You train a model to predict that output for new, unseen inputs.

Classic examples:
- **Email spam detection** — each email is labeled "spam" or "not spam."
- **House price estimation** — each listing has a sale price attached.
- **Churn prediction** — each customer has a historical label: did they cancel, yes or no?

The word *supervised* is a useful metaphor. The labels are the "teacher" — the model learns by comparing its prediction against the known answer and adjusting until it gets good at it.

Two sub-types show up constantly:
- **Classification** — the output is a category (spam/not-spam, fraud/legit).
- **Regression** — the output is a number (price, temperature, revenue).

### Unsupervised learning — finding structure without labels

No labels here. You hand the model raw data and ask: "find patterns, groups, or anomalies."

Classic examples:
- **Customer segmentation** — "group our users into natural clusters based on behavior." Nobody labeled users as Segment A or B beforehand; the model discovers the structure.
- **Topic modeling on support tickets** — automatically surface themes without manually tagging thousands of tickets.
- **Anomaly detection on server metrics** — identify outliers without having a list of labeled past outages.

This is underused in production partly because it's harder to evaluate. How do you know if your clusters are *correct*? You need domain expertise to validate them.

### Reinforcement learning — learning from rewards

Here the model (called an *agent*) takes actions in an environment, receives reward signals, and learns a *policy*: which actions lead to the highest cumulative reward.

Classic examples:
- **Game-playing AI** (chess, Go, video games) — the reward is winning.
- **Ad bidding systems** — the agent decides how much to bid for each ad slot; the reward is profit.
- **LLM fine-tuning via RLHF** — the "reward" is a human preference rating; this is exactly how ChatGPT was tuned.

RL is powerful but expensive and tricky to get right. Use it when the right answer isn't known upfront and can only be measured as an outcome over time.

:::why-prod
Getting the framing wrong is the most expensive mistake in ML. Teams have built clustering pipelines when they should have used classification (they had labels all along), and built RL systems when a simpler supervised model would have solved the same problem in a week. Framing is a 10-minute decision that saves months.
:::

## How to pick the right flavor

The decision is almost always driven by three questions.

:::table {title="ML framing decision guide"}
| Question | Answer | Lean toward |
|---|---|---|
| Do you have labeled outputs to learn from? | Yes | Supervised |
| Do you want to discover hidden structure or groups? | Yes, no labels | Unsupervised |
| Is the "correct answer" only measurable as a future outcome/reward? | Yes | Reinforcement Learning |
| Do you have some labels but many unlabeled rows? | Yes | Semi-supervised (blend) |
:::

A useful rule of thumb: **start with supervised if you can get labels**. It's the most battle-tested, easiest to evaluate, and easiest to explain to stakeholders. Reach for unsupervised when labels don't exist or would be too expensive to create. Reserve RL for sequential decision problems where the environment changes based on your actions.

:::gotcha
The most common framing mistake is building an unsupervised model when you already have labels buried in your data. "We want to cluster customers" sounds exciting — but if you already track who churned, you *have* a churn label. A supervised classifier will almost always outperform clustering for that goal, and it's directly optimizing the thing you care about.
:::

## A worked example: same data, three framings

Say you work on a music streaming platform and have years of listening history.

- **Supervised framing**: Predict whether a user will cancel next month. Label = churned (yes/no) from historical data. Ship a classifier.
- **Unsupervised framing**: Discover natural listener archetypes (the "Gym Warrior", the "Late-Night Jazz fan") without predefined categories. Use clustering.
- **RL framing**: Optimize the recommendation queue so the agent learns to maximize total listening time over a session. Reward = minutes listened. Use a recommendation policy trained via RL.

All three are valid ML problems on the same data. The right one depends on what decision you're trying to make.

:::interview-line
"The first question I ask is whether we have labels — that single answer routes me to supervised versus unsupervised, and then I only reach for RL if the reward can't be observed until later in time."
:::

:::qa {q="What's the key difference between supervised and unsupervised learning?"}
In supervised learning you have labeled examples — the model learns by comparing its predictions to known correct answers. In unsupervised learning there are no labels; the model finds structure, patterns, or groups on its own. The practical trigger is simple: if someone has already labeled the data (even historically), use supervised.
:::

:::qa {q="When would you choose reinforcement learning over supervised learning?"}
When the correct answer can't be known upfront and is only observable as a downstream reward signal — like ad revenue, game score, or user satisfaction after a recommendation session. RL is also the right tool when the model's actions change the environment and you need to optimize a sequence of decisions, not a single prediction.
:::

:::qa {q="What is semi-supervised learning and when is it useful?"}
Semi-supervised learning uses a small set of labeled examples alongside a large pool of unlabeled data. It's useful when labels are expensive to collect — think medical imaging annotations by specialist doctors — but raw data is abundant. The model uses the unlabeled data to improve its internal representations before fitting the labeled examples.
:::

:::drill {type="mcq" q="A logistics company wants to predict whether a delivery will be late, using historical shipments that are already flagged as 'on-time' or 'delayed'. Which type of ML is the best fit?"}
- [ ] Unsupervised learning, because the patterns are unknown
- [x] Supervised classification, because labeled outcomes already exist
- [ ] Reinforcement learning, because the delivery agent makes sequential decisions
- [ ] Semi-supervised learning, because some shipments may not have labels
:::

:::drill {type="mcq" q="An e-commerce team has no predefined product categories but wants to automatically group their 50,000 SKUs into natural families for merchandising. Which approach fits best?"}
- [ ] Supervised regression on product prices
- [ ] Reinforcement learning with purchase reward signals
- [x] Unsupervised clustering on product features and sales patterns
- [ ] Supervised multi-class classification with manually labeled categories
:::

:::key-takeaway
Choose supervised when you have labels, unsupervised when you want to discover hidden structure without labels, and reinforcement learning only when the reward is a delayed outcome from sequential decisions. Framing correctly before writing a single line of code is the highest-leverage skill in applied ML.
:::
