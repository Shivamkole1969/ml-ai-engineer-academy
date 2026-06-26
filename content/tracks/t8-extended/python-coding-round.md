---
id: python-coding-round
track: t8-extended
title: "The Python coding round"
badge: HOT
minutes: 9
prereqs: []
tags: [python, interviews, coding-round, numpy, pandas, data-structures, problem-solving]
xp: 60
hot2026: true
---

You have spent three rounds impressing the panel. You nailed the ML design question, held your own on system design, and now it is the final screen — a live coding session. The interviewer says: "Write a function that finds the top-K most frequent tokens in a list of strings." You know Python. You use it every day. And yet, for a few terrible seconds, your mind goes completely blank.

That is the Python coding round. This lesson is your unfair advantage.

## What the Interviewer Is Actually Testing

The Python coding round for ML/AI roles is not a competitive programming contest. Nobody expects you to invert a binary tree in ninety seconds. What interviewers want to see is three things:

1. **Pythonic fluency** — Can you write clean, idiomatic Python, or do you write C++ with Python syntax?
2. **Data-wrangling instincts** — Can you reshape, filter, and aggregate data without reaching for a loop every time?
3. **Awareness of complexity** — Do you know when your elegant one-liner is secretly O(n²)?

Most questions fall into one of a handful of categories. The companies change; the patterns barely do.

:::why-prod
In production, bad Python is usually not wrong — it is slow and unreadable. A teammate's list-of-loops that processes ten million rows in forty minutes gets replaced with a two-line vectorized NumPy call that takes four seconds. The coding round is a proxy for that instinct.
:::

## The Patterns That Show Up

**Token / frequency counting** is the most common ML-adjacent pattern. Build a frequency map, then select the top-K entries. The naive approach works; the Pythonic approach uses `collections.Counter` and its `.most_common(k)` method — done in two lines.

**Sliding-window over sequences** appears whenever the question involves a "window" of recent predictions, sensor readings, or tokens. You keep two pointers and update a running state instead of re-scanning from scratch. This takes O(n) instead of O(n²).

**Vectorized NumPy / Pandas operations** replace explicit Python loops over arrays. The rule is simple: if you are looping over a NumPy array element by element, there is almost certainly a better way involving broadcasting or a built-in ufunc.

**Generator pipelines** for large data. If your interviewer adds "assume the file is too big to fit in memory," the answer almost always involves `yield` — you process one record at a time instead of loading everything.

**Dictionary / set lookups for O(1) membership.** Whenever you catch yourself checking `if x in some_list` inside a loop, convert `some_list` to a `set` first. The list version is O(n) per check; the set version is O(1).

```python {title="Top-K frequent tokens — the clean way" run=false}
from collections import Counter

def top_k_tokens(documents: list[str], k: int) -> list[str]:
    """
    Return the k most frequent tokens across all documents.
    Each document is a whitespace-separated string.

    Run locally:  python solution.py
    """
    # Flatten all tokens in one pass using a generator — no intermediate list
    all_tokens = (token for doc in documents for token in doc.lower().split())

    # Counter handles the frequency map; most_common is O(n log k)
    counter = Counter(all_tokens)
    return [token for token, _ in counter.most_common(k)]


# Quick sanity check
docs = ["the cat sat", "the cat ate", "the dog sat"]
print(top_k_tokens(docs, k=2))   # ['the', 'cat']
```

Notice what is NOT here: a nested loop, a hand-rolled frequency dict, or a manual sort. Every line earns its place.

:::table {title="Common question types and the Python tool to reach for"}
| Question category | Reach for |
|---|---|
| Frequency / top-K | `collections.Counter.most_common` |
| Sliding window | Two-pointer with running state |
| Large file, one pass | Generator + `yield` |
| Set membership in a loop | Convert list → `set` first |
| Row-wise ops on a DataFrame | `.apply()` or vectorized column ops |
| Elementwise math on arrays | NumPy broadcasting, avoid `for` loops |
| Grouping / aggregation | `pd.DataFrame.groupby` |
:::

## Writing Clean Code Under the Clock

A few habits separate candidates who feel confident from candidates who spiral:

**Start by restating the problem in your own words.** This buys thinking time, surfaces edge cases early, and shows communication skill — all things interviewers explicitly score.

**Write a docstring and a tiny example before the first line of logic.** It forces you to clarify the input/output contract and gives you an instant test case.

**Name variables for humans, not compilers.** `freq_map` beats `d`; `top_tokens` beats `res`. You are being evaluated on code quality, not keystroke count.

**Say what you are about to do before you do it.** "I'll use a Counter here because it gives me most_common for free" signals understanding, not just rote recall.

:::gotcha
Mutable default arguments are the most famous Python trap and interviewers still use them. `def process(data, cache=[])` shares the same list across every call. Use `None` as the default and create the list inside the function body. The same trap bites you with default dicts: `def f(d={})` is almost always a bug.
:::

:::war-story {title="The loop that cost a job offer"}
A candidate at a mid-sized ML platform was given a warm-up task: compute pairwise cosine similarity for a list of 10,000 embedding vectors. They wrote a double Python for-loop, verified it was correct, and submitted. The interviewer ran it. It took over four minutes. "It works, but can you make it fast?" the interviewer asked. The candidate added `tqdm` for a progress bar. The role went to the next person, who reached for `sklearn.metrics.pairwise.cosine_similarity` — a single NumPy-backed call that finished in under a second. The lesson: always ask yourself if the standard library or NumPy already solves your sub-problem.
:::

:::interview-line
"I reach for `Counter.most_common` over a hand-rolled sort because it is O(n log k) instead of O(n log n), and it signals that I know the standard library."
:::

:::qa {q="What is the difference between a list comprehension and a generator expression, and when do you choose each?"}
Both build sequences lazily or eagerly from an iterable. A list comprehension `[x for x in items]` materialises the full list in memory immediately — use it when you need random access or know the result fits in memory. A generator expression `(x for x in items)` yields values one at a time — use it for large sequences or when you only need a single pass, like feeding `sum()` or `Counter()`. In ML pipelines, generators keep memory flat even over multi-GB datasets.
:::

:::qa {q="How would you speed up a Python loop that computes the dot product of two large arrays?"}
Replace it with `np.dot(a, b)` or the `@` operator. NumPy operations execute in compiled C/Fortran and release the GIL, so they are orders of magnitude faster than a Python for-loop. If the arrays are very large and you need GPU speed, move to `torch.matmul`. The key interview point is: never loop over array elements in pure Python when a vectorised operation exists.
:::

:::drill {type="mcq" q="You need to check whether each item in a 50,000-element list appears in a second list of 50,000 items. Which implementation is fastest?"}
- [ ] `[x for x in list_a if x in list_b]` — linear scan per lookup
- [x] Convert `list_b` to a `set`, then `[x for x in list_a if x in set_b]` — O(1) per lookup
- [ ] Sort both lists, then use a two-pointer merge — O(n log n)
- [ ] Use `filter(lambda x: x in list_b, list_a)` — lazy but still O(n²)
:::

:::drill {type="mcq" q="Which of the following is a valid mutable-default-argument bug?"}
- [ ] `def greet(name: str = 'world'): ...`
- [ ] `def add(x: int, y: int = 0): ...`
- [x] `def append_item(item, results=[]): results.append(item); return results`
- [ ] `def repeat(text: str, times: int = 3): return text * times`
:::

:::key-takeaway
The Python coding round rewards knowing the standard library, reaching for vectorised operations instinctively, and writing code that reads like documentation. Master `Counter`, generators, set lookups, and NumPy — then talk through every decision as you go.
:::
