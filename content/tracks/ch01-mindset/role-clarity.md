---
id: role-clarity
track: ch01-mindset
title: "Role clarity: where the AI/ML engineer sits"
badge: FOUNDATION
minutes: 9
prereqs: []
tags: [career, roles, ml-engineer, data-scientist, mlops, team-structure]
xp: 30
hot2026: false
---

Picture your first week at a new company. You have been hired as an "AI/ML Engineer." On Monday, the data science team asks you to tune a neural network. On Tuesday, the platform team wants you to set up a feature pipeline. On Wednesday, the product manager asks why the model's outputs look weird in production. By Thursday you wonder: *what exactly is my job?*

That confusion is not unique to you — it is baked into how this industry names roles. Understanding the landscape before you land saves you weeks of figuring it out on the fly.

## The three clusters that matter

Most teams, regardless of what they print on job postings, operate across three overlapping clusters.

**Data Scientists** are the researchers. Their primary job is finding signal: which features matter, which model architecture fits, what accuracy is achievable. They live in notebooks and care deeply about metrics.

**ML Engineers** (sometimes "Applied Scientists") are the bridge. They take a model that works in a notebook and make it work reliably at scale — serving, monitoring, retraining, version control, latency budgets. This is the role most companies are desperately short on.

**MLOps / Platform Engineers** build the infrastructure everyone else uses: training clusters, feature stores, CI/CD for models, experiment tracking. They think in pipelines and SLAs, not model weights.

In a startup with five people, one person does all three. In a large org, you might specialise in one. Knowing where you sit — and where you are *expected* to sit — changes what skills to sharpen first.

:::why-prod
In production, the gap between "model works in a notebook" and "model delivers value reliably" is enormous. Most production failures are not model failures — they are integration, monitoring, or handoff failures. An engineer who understands all three clusters can catch problems before they reach users.
:::

:::table {title="Role clusters at a glance"}
| Cluster | Primary question | Key tools | Success metric |
|---|---|---|---|
| Data Scientist | "Can we solve this at all?" | Jupyter, scikit-learn, PyTorch | Offline metric (AUC, RMSE, BLEU) |
| ML Engineer | "Can we run this reliably?" | FastAPI, Docker, cloud APIs, MLflow | Latency, throughput, uptime |
| MLOps / Platform | "Can everyone build fast?" | Kubeflow, Airflow, Terraform, feature stores | Developer velocity, infra cost |
:::

## Where the AI/ML Engineer role actually lives

The job title "Machine Learning Engineer" (MLE) has quietly become the most common production AI hire. Why? Because companies learned the hard way that data scientists without engineering support ship models that die in staging.

An MLE in most product companies is responsible for:

- Turning a research prototype into a deployable artifact
- Writing inference code that handles edge cases and bad inputs
- Setting up monitoring so the team knows when the model drifts
- Coordinating with data engineers on feature pipelines
- Sometimes — when the team is small — doing the modeling too

The GenAI wave added a fourth flavour: **GenAI / LLM Engineers** who work primarily with foundation models, prompts, RAG pipelines, and agents rather than training from scratch. The engineering discipline is the same; the model artifact is rented from a provider instead of trained in-house.

:::gotcha
Many candidates prepare only for the "what algorithm would you use?" interview but get tripped up by system design questions like "how would you serve this model to 10,000 concurrent users?" Know your cluster and prepare accordingly. If the JD says "ML Engineer," study serving, monitoring, and pipelines — not just model selection.
:::

## How to read a job description

Job descriptions are written by committees and often mix all three clusters into one role. A practical heuristic:

Look at the **required tools**, not the title. If you see Kubernetes, Airflow, and SLAs — that is MLOps-flavoured. If you see Jupyter, experimentation, and A/B testing — that is DS-flavoured. If you see model serving, REST APIs, and latency — that is MLE-flavoured.

Then look at who you report to. Reporting to a VP of Engineering signals that engineering craft matters as much as model quality. Reporting to a VP of Data Science signals the opposite.

Neither is wrong. They are just different jobs.

:::interview-line
"I think of my role as the connective tissue between research and production — I make sure models that work in experiments actually work for users."
:::

:::qa {q="How is an ML Engineer different from a Data Scientist?"}
A Data Scientist focuses on finding what works — exploration, experimentation, and offline metrics. An ML Engineer focuses on making it work in production — serving, reliability, monitoring, and integration. In many teams these roles overlap, but when they split, the MLE owns the path from trained model to live system.
:::

:::qa {q="What is MLOps and where does it fit?"}
MLOps is the practice of applying DevOps principles to machine learning: version-controlled models, automated retraining pipelines, reproducible experiments, and infrastructure-as-code for training and serving. MLOps engineers build the platform that data scientists and ML engineers rely on. At smaller companies, the ML Engineer often covers MLOps duties too.
:::

:::qa {q="What does an LLM/GenAI Engineer do differently from a traditional ML Engineer?"}
A GenAI engineer works primarily with pre-trained foundation models via APIs or local deployment, rather than training models from scratch. Their unique skills are prompt engineering, RAG architecture, fine-tuning strategies, and cost/latency management for large models. The production engineering concerns — serving, monitoring, safety — are the same; the training infrastructure is largely outsourced to a model provider.
:::

:::drill {type="mcq" q="A job posting requires: FastAPI, Docker, model latency SLAs, and on-call rotation. Which cluster best describes this role?"}
- [ ] Data Scientist
- [ ] Research Scientist
- [x] ML Engineer
- [ ] Data Analyst
:::

:::drill {type="mcq" q="Your team's data scientist hands you a trained model checkpoint. You are asked to 'productionise it.' Which task is NOT typically your responsibility as the ML Engineer?"}
- [ ] Writing a prediction endpoint with input validation
- [ ] Setting up latency monitoring and alerting
- [x] Choosing the model architecture for the next experiment
- [ ] Containerising the model for deployment
:::

:::key-takeaway
"ML Engineer" is a production role. Its core job is bridging research and reliability — know which cluster a role expects you to fill, and tailor both your preparation and your first-week questions accordingly.
:::
