---
id: docs-operable
track: 14-production-safety
title: "Documentation that makes a system operable: model cards, ADRs, runbooks"
badge: CORE
minutes: 8
prereqs: []
tags: [documentation, model-cards, adr, runbooks, mlops, production]
xp: 45
hot2026: false
---

Six months after you shipped that loan-scoring model, the risk team pings you on Slack at 11 PM. "Approval rates dropped 30% — is the model broken or is this intentional?" You pull up the repo. The notebook is named `final_v3_USE_THIS.ipynb`. The git log says "fixes" three times. There is no record of *why* the approval threshold is 0.62 instead of 0.5, *which* dataset version was used, or *what* an on-call engineer is supposed to do when confidence collapses. Multiply this by five models and a new team member who joined last month. That is a documentation problem wearing an operations costume. The fix is three artefacts: a **model card**, an **ADR**, and a **runbook**.

## Three artefacts — one passport, one diary, one playbook

Think of them that way.

- **Model card** — the passport: who this model is, what it can do, where it must *not* go.
- **ADR (Architecture Decision Record)** — the diary: why we made that technical call at that moment.
- **Runbook** — the playbook: exactly what an engineer does when the dashboard goes red at 11 PM.

None of them are lengthy. All of them are survival gear.

:::why-prod
On a production team, models are built by one person and operated by another — often months apart, across team changes and incidents. Without these three artefacts, every incident is archaeology. With them, your system can be owned, handed off, and debugged by someone who did not build it.
:::

## Model cards

Google published the model card paper in 2019. The idea is simple: a one-page "nutrition label" for your model. Not how to retrain it — that is a README job. A model card answers: what did this model learn, on what data, under what conditions, and where is it safe to deploy?

A minimal internal model card covers:

- **Intended use** — loan default risk for retail applicants. NOT for employment screening or insurance pricing.
- **Training data** — Q3 2024 bureau data, 40k samples, balanced 50/50 default/non-default.
- **Performance** — AUC 0.84, precision 0.81, recall 0.79 on a held-out 20% split.
- **Known limitations** — degrades for applicants with < 6 months credit history (only 4% of training data).
- **Threshold policy** — score < 0.62 → reject; 0.62–0.75 → manual review; > 0.75 → auto-approve.
- **Owner and refresh trigger** — @risk-ml-team, retrain quarterly or if Population Stability Index (PSI) exceeds 0.2.

That last bullet is the one most teams skip. A model card without an owner and a refresh trigger is just archaeology-in-waiting.

## ADRs

An ADR answers one question: *why did we build it this way?*

The format is deliberately short: title, date, status (proposed / accepted / deprecated), context, decision, consequences. One page. Version-controlled with the code. Done.

Example from a real-feeling system:

> **ADR-007: Use XGBoost over a deep model for loan scoring**
>
> *Context:* Team has < 1k GPU-hours/month. Data is tabular with high-cardinality categoricals. Regulation requires a written explanation for every rejection.
>
> *Decision:* XGBoost with SHAP feature attributions.
>
> *Consequences:* Inference is sub-millisecond and explainability is native. Risk accepted: tree models may underfit if we later add unstructured text features. Re-evaluate in 12 months.

Now when someone asks "why not a neural net?" the answer is one link, not a 20-minute retrospective.

Store ADRs in `docs/adr/` inside the repo — not on a wiki. Wikis decouple from the code and drift. ADRs in the repo appear in the same git history, get reviewed in the same PRs, and get archived when the code is archived.

:::table {title="Three artefacts at a glance"}
| Artefact | What it captures | Who reads it | When to write it |
|---|---|---|---|
| Model card | What the model is, what it can do, and what it must not do | Product, legal, ops, new engineers | At release; regenerate each retrain |
| ADR | Why a specific technical decision was made, and what trade-offs were accepted | Future engineers inheriting the system | At decision time, before or during implementation |
| Runbook | Step-by-step actions for when something breaks in production | On-call engineer, possibly at 2 AM | Before go-live; review quarterly |
:::

## Runbooks

A runbook is the doc you *wish* existed during the last incident. It does not describe the system architecture — the model card and ADRs handle that. It answers: what are the symptoms, what do I check first, what actions can I take, and when do I escalate?

A useful structure:

1. **Trigger** — when to open this runbook. Example: "Approval rate drops > 20% versus the 24-hour rolling average."
2. **Checklist** — concrete steps in the order to perform them: check feature pipeline freshness → check model server latency → open drift dashboard.
3. **Remediation** — for each failure mode, the specific action: "If feature pipeline job `bureau-etl` has not run in 4h, restart it via Airflow UI → DAGs → bureau-etl → trigger."
4. **Escalation** — if unresolved in 30 minutes, page `@risk-ml-oncall` via PagerDuty.

A rough runbook covering 80% of incidents is infinitely more valuable than a perfect document that does not exist yet.

:::gotcha
Documentation rots. A model card written once and forgotten gives false confidence. The fix: attach model card regeneration to your retrain pipeline so it updates automatically. Gate ADR changes through PR review so they stay in sync with code changes. Put a "last tested on" date in every runbook and schedule a quarterly review. If it is not on a calendar, it does not happen.
:::

:::interview-line
"We treat model cards, ADRs, and runbooks as first-class deliverables, not optional docs we write after the first incident — they are what makes a system operable by someone other than the person who built it."
:::

:::qa {q="What does a model card capture that a README does not?"}
A README explains how to run the code. A model card captures the ML-specific trust context: intended use and explicit out-of-scope uses, training data provenance, performance metrics (ideally sliced by subgroup), known failure modes and edge cases, the threshold policy, and the owner plus refresh cadence. It is written for product managers, legal reviewers, and on-call engineers — not just ML engineers.
:::

:::qa {q="Why store ADRs inside the code repo rather than on a team wiki?"}
ADRs and code age together. In the repo, an ADR appears in git blame, travels with the branch, gets reviewed in PRs, and is automatically archived when the repo is archived. A wiki page decouples from the code — it will drift, lose context, and become impossible to trust. The rule of thumb: if a decision affects the code, the record of that decision belongs next to the code.
:::

:::drill {type="mcq" q="Which field on a model card is most critical for preventing misuse of an ML model?"}
- [ ] The hyperparameters used during training
- [ ] The cloud region where training ran
- [x] The explicit list of intended uses and out-of-scope uses
- [ ] The number of training epochs
:::

:::drill {type="mcq" q="An ADR is marked 'Deprecated'. What does that signal?"}
- [ ] The document was written but never formally reviewed
- [ ] The decision it records turned out to be a mistake
- [x] The decision was valid at the time but has since been superseded by a newer ADR
- [ ] The system that used this decision has been decommissioned
:::

:::key-takeaway
Model cards, ADRs, and runbooks are the difference between a system only its creator can operate and one a whole team can own, hand off, and fix at 11 PM. Write them before go-live — not after the first incident.
:::
