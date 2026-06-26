---
id: schema-evolution
track: 12-pipelines
title: "Schema evolution (backward/forward/breaking; the silent unit bug)"
badge: CORE
minutes: 8
prereqs: []
tags: [schema, data-contracts, pipelines, feature-engineering, mlops]
xp: 45
hot2026: false
---

It's a Tuesday morning. Your recommendation model's click-through rate has dropped 30% overnight. No pipeline errors. No schema validation failures. Data flows in cleanly, features are computed, predictions are served. Everything *looks* fine.

Four hours later you find it. The upstream team changed `session_duration` from **seconds to milliseconds** — same column name, same data type (`int64`), just a different unit. Your feature store ingested it happily. Your model silently learned that "a long session" is now 1,000x larger than before. Every prediction since the upstream deploy has been subtly wrong.

This is the schema evolution problem. And the unit bug is its most dangerous form.

## What schema evolution actually means

A schema is the contract for your data: column names, types, and — critically — *semantics* (what a column means, including its unit).

Schema evolution is what happens when that contract changes over time. Upstream systems add columns, rename things, change types, or quietly change units as business logic shifts. The schema is rarely frozen; your pipeline's job is to survive the changes it knows about and *detect* the ones it doesn't.

There are three kinds of change, and they have very different consequences.

:::why-prod
ML models are uniquely fragile to schema drift. A web API can throw a 400 error when a field goes missing. Your feature pipeline will often just fill in a default or NaN and keep going — silently feeding garbage into a model that cannot complain.
:::

:::table {title="The four schema change categories"}
| Type | Definition | Example | Pipeline outcome |
|---|---|---|---|
| **Backward compatible** | New writer, old reader still works | Add a nullable column | Usually safe |
| **Forward compatible** | Old writer, new reader still works | Reader ignores unknown columns | Usually safe |
| **Breaking** | Old and new cannot interoperate | Rename a column; int → string | Loud crash |
| **Silent / semantic** | Type stays the same, meaning changes | `session_duration` unit: sec → ms | Worst of all |
:::

## Breaking changes are loud — silent changes are deadly

A breaking change is actually the *easy* problem. Your pipeline crashes, you get paged, you fix it within the hour. The error is loud and points directly at the cause.

The silent unit bug is different. It passes every type check. It passes schema validation. It passes null checks. The only thing that breaks is your model's accuracy — and that might not surface until a business review two weeks later.

Why does it happen so often? Because engineers think of schema as "name + type" and forget **semantics**. When the mobile team switches their event timer from seconds to milliseconds for sub-second precision, that's a local win for them. They updated their internal docs. Nobody told the ML team whose feature pipeline reads `session_duration` and was trained assuming seconds.

```python {title="Pandera schema check — catch drift before it reaches your model" run=false}
import pandera as pa
import pandas as pd

# Declare what your features should look like at training time.
# Pin this alongside your model artifact.
schema = pa.DataFrameSchema({
    "session_duration_sec": pa.Column(
        float,
        checks=[
            pa.Check.ge(0),       # can't be negative
            pa.Check.le(7200),    # cap at 2 hours; 1000x spike = unit bug
        ],
        nullable=False,
    ),
    "page_views": pa.Column(int, checks=pa.Check.ge(0)),
})

def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Call this right after reading from your feature store or upstream table."""
    # Raises pandera.errors.SchemaError loudly — loud is always better than silent.
    return schema.validate(df)

# Local setup: pip install pandera  (no infra needed)
```

## Backward vs forward: which direction matters more in ML?

In web APIs, you care about both directions equally. In ML systems, **backward compatibility is the higher-stakes direction**.

Here is why. Your trained model is a frozen reader — it was fitted on a schema at a point in time. If the serving-time schema drifts from the training-time schema, you have *training-serving skew*: the model sees inputs it was never trained on, and degrades silently. That drift is almost always driven by a backward-compatibility break or a silent semantic change.

Forward compatibility matters too — if you roll back a pipeline version, can the old code still read the data your new pipeline wrote? Think about this before you deploy anything that writes to a shared feature table.

## Three layers of defence

**1. Schema registry with version pinning.** Tools like Apache Avro + Confluent Schema Registry, or even a simple Pandera schema saved alongside your model, let your pipeline declare "I was trained on schema v4." At serving time, the pipeline refuses to ingest a v5 payload until you explicitly accept the migration.

**2. Statistical validation on numeric features.** After ingestion, check that features stay within expected ranges. If `session_duration_sec` suddenly has a mean 1,000x higher than your training baseline, alert. This catches the unit bug even if your schema registry sees no type change.

**3. Semantic contracts in column names.** `session_duration_sec` is unambiguous. `session_duration` is a landmine. This costs nothing and saves hours.

:::gotcha
Adding a new nullable column feels safe — and structurally it is backward compatible. But it can quietly hurt ML pipelines later. If that column becomes a feature, historical records where it is NULL represent *structural missingness* (the column did not exist yet), not random missingness. Your imputation strategy will be wrong, and your model's behaviour on old data will diverge from new data. Always document when a column was first backfilled and whether pre-backfill NULLs are structural.
:::

:::interview-line
"Type-check-breaking schema changes are loud and fast to fix. The dangerous ones are semantic — same type, different meaning — and the only defence is statistical range validation plus encoding units directly in column names."
:::

:::qa {q="What is the difference between backward and forward schema compatibility?"}
Backward compatibility means new writers, old readers — existing consumers keep working after an upstream change. Forward compatibility means old writers, new readers — you can roll back a consumer and still read data produced by the newer version. In ML, backward compatibility carries more weight because a trained model is a frozen reader that must keep correctly interpreting data long after deployment.
:::

:::qa {q="A numeric feature's mean suddenly spikes 1000x with no code change on the ML side and no pipeline error. What is your first hypothesis and next step?"}
First hypothesis: a silent semantic change upstream — most likely a unit change on a column with the same name and type. The next step is to check the upstream table's changelog or ping the owning team directly, then compare the raw value distribution before and after the spike date. If the mean shifted exactly 1000x, it is almost certainly seconds-to-milliseconds.
:::

:::qa {q="How do you prevent training-serving skew caused by schema drift?"}
Pin the schema your model was trained on — using a schema registry, a Pandera schema file, or even a JSON spec stored alongside the model artifact. At serving time, validate incoming features against that pinned schema: structural checks (types, nullability) and statistical checks (range bounds, mean/std guard rails). Alert or reject rather than silently passing drifted data to the model.
:::

:::drill {type="mcq" q="An upstream team renames a column from `user_age` to `user_age_years` with no other changes. What kind of schema change is this?"}
- [ ] Silent/semantic — the meaning changed
- [ ] Backward compatible — old readers still work because the data is the same
- [x] Breaking — any downstream code referencing `user_age` by name will fail
- [ ] Forward compatible — the new name carries more information
:::

:::drill {type="mcq" q="Your model RMSE degrades two weeks after a data pipeline deploy. No schema errors, no null spikes, no code changes on the ML side. Which is MOST likely?"}
- [ ] A bug introduced in the model loss function during training
- [ ] A newly added nullable column caused imputation bias
- [x] A silent semantic change — a unit or definition shift in a feature column
- [ ] The schema registry was queried with the wrong version tag
:::

:::key-takeaway
Breaking schema changes crash loudly and get fixed fast. Silent semantic changes — same name, same type, different meaning — degrade your model quietly for weeks. Encode units directly in column names and add statistical range checks after ingestion; type validation alone will never catch this class of bug.
:::
