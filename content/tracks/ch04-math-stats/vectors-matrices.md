---
id: vectors-matrices
track: ch04-math-stats
title: "Vectors & matrices, intuitively"
badge: FOUNDATION
minutes: 9
prereqs: []
tags: [linear-algebra, embeddings, dot-product, matrix-multiplication, foundations]
xp: 30
hot2026: false
---

Imagine you've just built a movie recommendation system. A user watches an action thriller, and your job is to instantly surface the ten most similar films from a library of five million. You can't compare every movie line by line — you need a compact, comparable *representation* of each film. That representation is a vector. And the math that lets you compare, combine, and transform those representations? That's linear algebra. It is the language every neural network, embedding model, and recommender system speaks under the hood.

## What a vector actually is

A vector is just a list of numbers that describes something.

A movie might be described as `[0.8, 0.1, 0.9, 0.3]` — high on action, low on romance, high on suspense, low on comedy. That four-number list is a vector. A word embedding might be a list of 768 numbers. A user profile might be 128 numbers. The count doesn't matter — the idea is the same: pack everything you know about a thing into one row of numbers.

In Python / NumPy:

```python {title="Vectors and dot products" run=false}
import numpy as np

# Two movie vectors (action, romance, suspense, comedy)
movie_a = np.array([0.8, 0.1, 0.9, 0.3])   # action thriller
movie_b = np.array([0.7, 0.2, 0.8, 0.1])   # similar thriller
movie_c = np.array([0.1, 0.9, 0.2, 0.8])   # rom-com

# Dot product: higher = more similar (if vectors are unit-length)
print(np.dot(movie_a, movie_b))   # ~0.90 — very similar
print(np.dot(movie_a, movie_c))   # ~0.30 — very different

# Cosine similarity: dot product of unit vectors
def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print(cosine_sim(movie_a, movie_b))   # ~0.998
print(cosine_sim(movie_a, movie_c))   # ~0.287
# Run locally: python3 -m pip install numpy && python3 this_file.py
```

The **dot product** multiplies matching positions and adds them up. When two vectors point in the same direction, the dot product is high. When they point in opposite directions, it's low. That's similarity, expressed as arithmetic.

:::why-prod
Every embedding-based feature in production — semantic search, recommendation, nearest-neighbor retrieval, anomaly detection — boils down to computing dot products between vectors at scale. Understanding the shape of your vectors is how you debug "why is this result so irrelevant?" and how you catch subtle bugs before they hit users.
:::

## What a matrix actually is

A matrix is a table of numbers. Stack several vectors as rows and you have a matrix.

The real power: a matrix *does something* to a vector. When you multiply a matrix by a vector, you transform that vector — rotate it, scale it, project it into a new space. This is exactly what happens inside a neural network layer. The weight matrix learns what transformation to apply so that the output is useful.

:::table {title="Vector vs Matrix at a glance"}
| Concept | Shape notation | Intuition | ML example |
|---|---|---|---|
| Scalar | `()` | A single number | A loss value |
| Vector | `(n,)` | A list of n numbers | A word embedding |
| Matrix | `(m, n)` | m rows, n columns | A weight layer |
| Batch of vectors | `(batch, n)` | Multiple inputs at once | Mini-batch inference |
:::

## Matrix multiplication: the workhorse

When your model runs a forward pass, the dominant operation is matrix multiplication. Input `X` has shape `(batch_size, input_dim)`. Weight matrix `W` has shape `(input_dim, output_dim)`. The result `X @ W` has shape `(batch_size, output_dim)`.

The rule to remember: *inner dimensions must match, outer dimensions give the output shape.*

`(batch, in) @ (in, out)` → `(batch, out)` ✓  
`(batch, in) @ (out, in)` → shape error ✗

This is the most common runtime error in ML code. A mismatched shape usually means a transposed matrix or a batch dimension in the wrong place.

:::gotcha
Confusing `(n,)` with `(n, 1)`. NumPy treats a plain 1-D array `(n,)` differently from a column vector `(n, 1)`. Dot products may work silently in one case and broadcast unexpectedly in the other. Always inspect `.shape` when something feels off — one extra `reshape(-1, 1)` or `unsqueeze(1)` in PyTorch is often the fix.
:::

## The transpose

Transpose just flips a matrix: rows become columns, columns become rows. You see `W.T` everywhere — in attention score computation, in gradient derivations, in the final projection layer of a language model. Visually: if `W` is `(3, 5)`, then `W.T` is `(5, 3)`.

It shows up most often when you want to compute *pairwise dot products* between two sets of vectors. If you have query matrix `Q` of shape `(batch, d)` and key matrix `K` of shape `(batch, d)`, then `Q @ K.T` gives you an `(batch, batch)` similarity matrix — which is exactly the attention score matrix in a transformer.

## Why norms matter

The **norm** of a vector is its length — how far it stretches from the origin. The L2 norm is `sqrt(sum of squares)`. Normalizing a vector to unit length (dividing by its norm) is critical before computing dot-product similarity, because otherwise a longer vector always scores higher regardless of direction. Most embedding models return unit-normalized vectors for this reason.

:::interview-line
"A neural network layer is just a learned linear transformation — matrix multiply plus bias — followed by a non-linearity that gives it expressive power."
:::

:::qa {q="Why does shape matter so much in practice?"}
Shape errors are caught at runtime, not compile time, so a wrong shape can slip through unit tests and explode in production on the first real batch. Keeping shapes explicit — using `.shape` assertions or type annotations — catches transposition bugs, missing batch dimensions, and broadcast surprises before they reach users.
:::

:::qa {q="What is the difference between a dot product and cosine similarity?"}
The dot product is the raw sum of element-wise products. Cosine similarity is the dot product divided by both vectors' norms, so it measures the angle between them regardless of magnitude. For comparing embeddings you almost always want cosine similarity, since embedding magnitude is not semantically meaningful.
:::

:::drill {type="mcq" q="You have matrix A of shape (32, 128) and matrix B of shape (128, 64). What is the shape of A @ B?"}
- [ ] (128, 128)
- [x] (32, 64)
- [ ] (32, 128)
- [ ] (64, 32)
:::

:::drill {type="mcq" q="A movie embedding vector is [2.0, 1.0, 4.0]. You want to use dot products to measure similarity between this vector and others. What should you do first?"}
- [ ] Multiply every element by 0.5 to keep values small
- [ ] Transpose the vector into a column vector
- [x] Normalize it to unit length by dividing by its L2 norm
- [ ] Sort the elements from largest to smallest
:::

:::key-takeaway
Vectors are compact representations of things; matrices are learned transformations. Every embedding lookup, attention score, and neural network forward pass is matrix multiplication in disguise — get comfortable with shapes, and the rest of ML math clicks into place.
:::
