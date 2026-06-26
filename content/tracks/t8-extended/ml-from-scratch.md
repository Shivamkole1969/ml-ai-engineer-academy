---
id: ml-from-scratch
track: t8-extended
title: "Implementing ML from scratch"
badge: HOT
minutes: 9
prereqs: []
tags: [interview, numpy, gradient-descent, linear-regression, logistic-regression, coding-round]
xp: 60
hot2026: true
---

You're forty minutes into an ML coding round. The interviewer says: "Great — now let's skip the library. Can you implement logistic regression from scratch using NumPy?" The scikit-learn safety net is gone. You either know what's happening inside the black box, or you don't.

This lesson is your playbook for that exact moment.

## Why interviewers ask this

Calling `sklearn.LinearRegression().fit(X, y)` is easy. Anyone can do it. Interviewers who ask you to implement from scratch want proof that you understand *what the library is actually doing* — gradient descent, weight updates, loss surfaces, numerical stability.

It's not hazing. It's the fastest signal they have that you won't be mystified when your production model converges to garbage and you need to debug it.

:::why-prod
Production bugs often hide inside assumptions you've delegated to a library. When a model's loss stops decreasing, understanding that gradient descent is updating weights by `w -= lr * gradient` lets you immediately suspect learning rate, batch size, or a gradient explosion — rather than staring at a black box. Scratch implementations are the best debugging education you can get.
:::

## The algorithms you must know cold

These five cover 90 % of what shows up in interviews. Each has a small surface area — just a few key equations.

:::table {title="Scratch Implementation Cheat Sheet"}
| Algorithm | Core equation to implement | Gotcha |
|---|---|---|
| Linear Regression | `w -= lr * X.T @ (X@w - y) / n` | Forget to divide by n → loss scale is wrong |
| Logistic Regression | Same gradient, sigmoid on output | Must clip sigmoid input to avoid `exp` overflow |
| k-Nearest Neighbours | Euclidean distance matrix, `argsort` | Naïve nested loop is O(n²) — use vectorized dist |
| Decision Tree (stump) | Gini or entropy split on each feature | Loop over thresholds, not just feature values |
| Gradient Descent (SGD) | Sample one row per step, update weights | Shuffling data each epoch matters — don't skip it |
:::

## The gradient descent engine

Almost everything else is built on gradient descent. Internalise this loop and you've cracked the hardest part.

```python {title="Linear & Logistic Regression from scratch" run=false}
import numpy as np

# --- helpers ---
def sigmoid(z):
    # Clip to prevent exp overflow on large negative values
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# --- Linear Regression ---
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)       # weight vector
        self.b = 0.0               # bias term
        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            err   = y_hat - y
            # MSE gradients (mean, not sum — keeps lr scale stable)
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()

    def predict(self, X):
        return X @ self.w + self.b

# --- Logistic Regression ---
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p   = sigmoid(X @ self.w + self.b)  # predicted probabilities
            err = p - y                          # cross-entropy gradient shortcut
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# --- Quick sanity check (run locally, no installs needed beyond numpy) ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((200, 3))
    y   = (X @ [1.5, -2.0, 0.5] + rng.standard_normal(200) * 0.3)
    model = LinearRegressionScratch(lr=0.05, epochs=500)
    model.fit(X, y)
    print("Learned weights:", model.w)   # should be close to [1.5, -2.0, 0.5]
```

## What interviewers actually check

They're not grading you on perfect code. They're watching for four things:

1. **Vectorised operations** — no nested Python loops where numpy can do it.
2. **Dividing gradients by n** — shows you understand batch vs. sum.
3. **Numerical guards** — sigmoid clipping, avoiding log(0) in cross-entropy.
4. **Can you explain each line** — they'll ask "why X.T @" and you need to say "that's the chain rule applied to the MSE loss."

:::gotcha
The most common mistake is forgetting the `/n` (dividing by the number of samples) in the gradient. Your loss still decreases, so you think it's working — but the effective learning rate is now 200× larger than intended. Switch datasets, and suddenly it diverges. Always normalise by batch size.
:::

:::war-story {title="The logistic regression that worked on toy data, died on real data"}
A candidate submitted a solid logistic regression implementation during a take-home screen. On their small toy dataset (100 rows, standardised) it converged beautifully. The team ran it on the 50 k-row dataset without pre-scaling. No sigmoid clipping. `exp` of a large positive number overflowed to `inf`, sigmoid returned `nan`, gradients became `nan`, weights became `nan`. The model predicted `nan` for every sample. The candidate had no idea why — they'd never seen the clipping guard before. Five lines of defensive code would have saved the offer.
:::

:::interview-line
"I implement the forward pass first, derive the gradient analytically, then write the update step — and I always add numerical guards before trusting the output."
:::

:::qa {q="What is the gradient of MSE loss with respect to the weights?"}
MSE loss is `(1/n) * ||Xw - y||²`. Taking the derivative with respect to `w` gives `(2/n) * X.T @ (Xw - y)`. The factor of 2 is absorbed into the learning rate in practice, so in code you write `X.T @ (y_hat - y) / n`. This is exactly the matrix form of summing each sample's contribution to the weight update.
:::

:::qa {q="Why use sigmoid in logistic regression instead of a hard threshold?"}
A hard threshold (step function) has zero gradient almost everywhere — you can't do gradient descent on it. Sigmoid is smooth and differentiable, so gradients flow back through it. It also outputs a calibrated probability between 0 and 1, which is more useful in production than a raw score when you need to set decision thresholds dynamically.
:::

:::drill {type="mcq" q="You implement linear regression from scratch and notice training loss stops decreasing after epoch 5 but is still high. What is the most likely cause?"}
- [ ] Your sigmoid has a numerical overflow
- [ ] You forgot to shuffle the data each epoch
- [x] Your learning rate is too small or you forgot to divide the gradient by n, making effective updates tiny
- [ ] Decision trees don't use gradient descent so the loop is wrong
:::

:::drill {type="mcq" q="Which NumPy expression correctly computes the Euclidean distance between every row of X_train (shape n×d) and a single query vector q (shape d)?"}
- [ ] `np.dot(X_train, q)`
- [ ] `np.sum(X_train - q)`
- [x] `np.sqrt(np.sum((X_train - q) ** 2, axis=1))`
- [ ] `np.linalg.norm(X_train)`
:::

:::key-takeaway
Implement the forward pass, write the gradient analytically, then code the update — and add numerical guards (clip, epsilon) before you trust any output. That three-step ritual is all "from scratch" interviews test.
:::
