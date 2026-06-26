---
id: numpy-pandas
track: t0-foundations
title: "NumPy & Pandas: the daily drivers"
badge: CORE
minutes: 9
prereqs: []
tags: [numpy, pandas, data-wrangling, vectorization, python]
xp: 45
hot2026: false
---

Your first feature pipeline is humming along on a 10 000-row CSV. Then the data team drops a 5 million-row file on you. Suddenly your code — full of neat Python `for` loops — takes 45 minutes to run. Your team lead walks over. You squint at the terminal. You realize you've been doing math one number at a time when the computer was begging to do it a million numbers at once.

That's the NumPy & Pandas moment. Once you see it, you can't unsee it.

## NumPy: math at machine speed

NumPy gives you the **ndarray** — a grid of numbers stored in a contiguous block of memory. Unlike a Python list, every element is the same type and lives right next to the others. That lets the CPU crunch whole arrays in a single instruction (SIMD), rather than looping through Python objects one by one.

The golden rule: **replace loops with array operations.**

```python {title="Vectorized vs loop — spot the difference" run=false}
import numpy as np

prices = np.array([10.5, 20.0, 15.75, 8.9])

# Bad: Python loop, slow at scale
discounted = []
for p in prices:
    discounted.append(p * 0.9)

# Good: vectorized — NumPy does the loop in C, not Python
discounted = prices * 0.9          # [9.45, 18.0, 14.175, 8.01]

# Useful ops you'll use every day
print(prices.mean())               # 13.7875
print(prices.std())                # 4.19
print(np.clip(prices, 9.0, 18.0)) # clamp outliers
print(prices[prices > 12.0])      # boolean mask — only rows you want

# Run locally:  pip install numpy  then  python this_file.py
```

**Shape and axis** are the two concepts that trip people up most. An array's `.shape` is `(rows, cols)` for 2-D data. When you call `arr.sum(axis=0)` you collapse *down* the rows (giving column totals); `axis=1` collapses across columns (row totals). Sketch it once on paper and it clicks forever.

## Pandas: NumPy with labels

Pandas wraps NumPy arrays into a **DataFrame** — a table where each column has a name and each row can have an index. Think of it as a spreadsheet that you control with code.

The two structures you use constantly:

- **Series** — a single labelled column (basically a NumPy array with an index).
- **DataFrame** — multiple Series sharing the same index; the thing your ML model will consume.

```python {title="Pandas in 30 lines — the moves you'll use every shift" run=false}
import pandas as pd
import numpy as np

# Load data (CSV, Parquet, database — all look like this once loaded)
df = pd.read_csv("orders.csv")   # swap for pd.read_parquet() for prod files

# First look — always do this before anything else
print(df.shape)          # (rows, cols)
print(df.dtypes)         # spot text columns hiding as objects
print(df.isnull().sum()) # null counts per column

# Select columns
prices = df["unit_price"]            # Series
subset = df[["user_id", "amount"]]   # DataFrame

# Filter rows (boolean mask — same idea as NumPy)
high_value = df[df["amount"] > 500]

# Create / transform a column
df["discount_price"] = df["unit_price"] * 0.9
df["log_amount"] = np.log1p(df["amount"])   # log1p = log(1+x), safe for 0s

# Groupby — the aggregation workhorse
summary = df.groupby("category")["amount"].agg(["mean", "sum", "count"])

# Merge two tables (like SQL JOIN)
merged = df.merge(users_df, on="user_id", how="left")

# Save to Parquet for the next step in the pipeline
df.to_parquet("orders_clean.parquet", index=False)

# Run locally: pip install pandas pyarrow
```

:::why-prod
In production pipelines, how fast you process data directly affects infrastructure cost and SLA latency. Vectorized NumPy/Pandas operations can be 50–500x faster than equivalent Python loops. Knowing which Pandas operations are copy-safe vs. mutating in place also prevents subtle data-corruption bugs that only surface in parallel workers.
:::

## The operations you'll reach for every day

:::table {title="Pandas one-liners you'll type from memory"}
| Task | Code |
|---|---|
| Drop duplicates | `df.drop_duplicates(subset=["user_id"])` |
| Fill nulls | `df["col"].fillna(0)` or `df["col"].ffill()` |
| Rename columns | `df.rename(columns={"old": "new"})` |
| Change dtype | `df["col"].astype("float32")` |
| Clip outliers | `df["col"].clip(lower=0, upper=99_000)` |
| Sample rows | `df.sample(n=1000, random_state=42)` |
| Sort | `df.sort_values("amount", ascending=False)` |
| Reset index | `df.reset_index(drop=True)` |
:::

## Memory matters at scale

A default Pandas DataFrame stores numbers as `float64` — 8 bytes per value. Switch to `float32` and you halve your RAM instantly. On a 50-column, 10 million-row dataset that's the difference between fitting in memory and crashing.

```python {title="Shrink DataFrame memory without losing precision" run=false}
# Downcast numeric columns after loading
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")

# Convert low-cardinality string columns to Category
df["status"] = df["status"].astype("category")

print(df.memory_usage(deep=True).sum() / 1e6, "MB")
# Run locally: pip install pandas
```

:::gotcha
`df["col"] = df["col"].apply(lambda x: x * 2)` looks harmless but `apply` with a Python lambda is almost as slow as a raw loop. Use `df["col"] * 2` instead — it's vectorized. The same trap appears with `iterrows()`: iterating row-by-row in Pandas is almost always the wrong move. If you find yourself writing `for _, row in df.iterrows()`, pause and ask whether a vectorized expression or `groupby` can replace it.
:::

:::interview-line
"I default to vectorized NumPy/Pandas operations and only reach for apply() or loops when the logic genuinely can't be expressed as an array operation — then I profile it first."
:::

:::qa {q="What is the difference between a Pandas Series and a DataFrame?"}
A Series is a single one-dimensional array with a labelled index — essentially one column. A DataFrame is a collection of Series sharing the same index, forming a two-dimensional table. In practice, selecting one column from a DataFrame returns a Series; selecting multiple columns returns a DataFrame.
:::

:::qa {q="When would you use np.clip() in a feature engineering step?"}
`np.clip()` caps values below a floor or above a ceiling, which is useful for taming extreme outliers before feeding data to a model. For example, clipping a revenue column at the 99th percentile prevents a handful of anomalous values from dominating normalisation and skewing learned weights.
:::

:::qa {q="Why is .astype('float32') useful when processing large datasets?"}
float32 uses 4 bytes per value versus float64's 8 bytes, halving memory consumption with negligible precision loss for most ML workloads. On multi-million-row datasets this can be the difference between a pipeline that fits in RAM and one that spills to disk or crashes, directly affecting throughput and cost.
:::

:::drill {type="mcq" q="You have a DataFrame with 8 million rows and a float64 column. Which change gives the biggest immediate memory reduction?"}
- [ ] Calling df.copy() to defragment memory
- [x] Casting the column to float32 with .astype('float32')
- [ ] Resetting the index with reset_index(drop=True)
- [ ] Sorting the column before processing
:::

:::drill {type="mcq" q="What does df.groupby('country')['revenue'].mean() return?"}
- [ ] A single float — the global mean of the revenue column
- [ ] A DataFrame with one row per unique country and all original columns
- [x] A Series indexed by country with the mean revenue for each country
- [ ] A DataFrame with one row per country and a 'mean' and 'count' column
:::

:::drill {type="mcq" q="Which of the following is the FASTEST way to double every value in a Pandas column?"}
- [ ] df['col'].apply(lambda x: x * 2)
- [ ] for i, row in df.iterrows(): df.at[i, 'col'] = row['col'] * 2
- [x] df['col'] * 2
- [ ] df['col'].map(lambda x: x * 2)
:::

:::key-takeaway
NumPy and Pandas are not just "data tools" — they are the performance layer between your ideas and production-grade pipelines. Replace Python loops with vectorized operations, watch your dtypes, and your pipelines will handle 100x more data with the same code.
:::
