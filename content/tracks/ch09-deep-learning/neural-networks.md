---
id: neural-networks
track: ch09-deep-learning
title: "Neural networks from first principles"
badge: CORE
minutes: 9
prereqs: []
tags: [deep-learning, neural-networks, perceptron, activation, forward-pass, weights]
xp: 45
hot2026: false
---

Imagine you are trying to predict whether a customer will churn next month. You hand a spreadsheet to a junior analyst with 50 columns and say "figure it out." They stare at it and try a few rules: "if age < 30 AND last_login > 60 days, probably churning." Good instinct, but those hand-crafted rules break the moment the data shifts. A neural network does something remarkably similar — but it learns its own rules, automatically, from examples.

## What a neuron actually does

A single artificial neuron is just a weighted sum followed by a squish.

That is the whole idea. Take all your input numbers, multiply each by a learned weight, add them up, then pass the total through a non-linear function (the *activation*). The output is a single number.

```python {title="A single neuron from scratch" run=false}
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(inputs, weights, bias):
    # Step 1: weighted sum
    total = sum(w * x for w, x in zip(weights, inputs)) + bias
    # Step 2: non-linear squish (sigmoid maps any number into (0, 1))
    return sigmoid(total)

# Example: 3 inputs, 3 weights
inputs  = [0.5, 1.2, -0.3]
weights = [0.8, -0.4, 1.1]
bias    = 0.1

print(neuron(inputs, weights, bias))  # → ~0.63
```

The bias is just an extra knob so the neuron can shift its threshold. Think of it as the neuron's "default opinion" before it sees any data.

## Stacking neurons into a network

One neuron can only draw one straight line through your data. Real problems are not straight lines.

Stack neurons side by side into a *layer*, then stack layers on top of each other — and you get a *feed-forward neural network* (also called a multi-layer perceptron, or MLP). Each layer transforms the data into a new representation, and the next layer builds on that.

- **Input layer** — raw features (pixel values, embeddings, numbers)
- **Hidden layers** — learned intermediate representations
- **Output layer** — your prediction (a class probability, a number, etc.)

The term *deep* in deep learning just means "many hidden layers." Two layers is shallow. Fifty layers is deep.

:::why-prod
In production, the number of layers and neurons directly controls model size, inference latency, and RAM footprint. A bigger network is not always better — it may overfit on small data, cost ten times more to serve, and respond 200 ms slower. Knowing what each layer does helps you make that trade-off with eyes open.
:::

## Activation functions: the non-linear magic

Without an activation function, stacking layers accomplishes nothing — you are just doing one big matrix multiply. Activations let layers learn curved, twisted, non-linear decision boundaries.

:::table {title="Common activation functions"}
| Activation | Formula (simplified) | Where you see it |
|---|---|---|
| Sigmoid | 1 / (1 + e^-x) → (0, 1) | Binary output layer |
| Tanh | (e^x - e^-x) / (e^x + e^-x) → (-1, 1) | RNNs, old-school hidden layers |
| ReLU | max(0, x) | Hidden layers in most modern nets |
| GELU | smooth ReLU-like curve | Transformers, BERT, GPT |
| Softmax | normalizes a vector to sum = 1 | Multi-class output layer |
:::

**ReLU** (Rectified Linear Unit) is the workhorse. It is fast to compute, does not saturate for large positive values, and just works well empirically. When in doubt, use ReLU for hidden layers.

## The forward pass

When data flows from input → hidden layers → output to produce a prediction, that is called the **forward pass**. This is what happens at inference time — your deployed model runs forward passes millions of times a day.

The weights start as small random numbers. Without training, the network outputs garbage. Training (covered in the Backpropagation lesson) adjusts those weights so the output gets closer to the true label.

:::gotcha
A very common beginner mistake: initializing all weights to zero. If every neuron starts identical, they learn identical things — the entire hidden layer collapses into a single neuron regardless of its width. Always use random initialization (or better, Xavier/He initialization for deep nets).
:::

## Why depth beats width (usually)

You can technically approximate any function with a *single* hidden layer if you make it wide enough — this is the Universal Approximation Theorem. But in practice, deeper networks learn hierarchical features far more efficiently.

Think about images. Layer 1 detects edges. Layer 2 combines edges into corners and curves. Layer 3 combines those into eyes and wheels. Layer 10 recognizes a face or a car. A single enormous flat layer would have to learn all of that simultaneously — it works in theory but explodes in parameter count.

Depth = efficient composition of simple features into complex ones.

:::interview-line
"A neural network is a series of learned linear transformations separated by non-linearities — the non-linearities are what give it the power to approximate arbitrary functions."
:::

:::qa {q="What is the role of the bias term in a neuron?"}
The bias is an offset that lets the neuron fire even when all inputs are zero, or stay quiet even when they are large. Without bias, every decision boundary in the network is forced to pass through the origin, which severely limits what the model can learn. In practice, bias = a learned constant added after the weighted sum.
:::

:::qa {q="Why can't you just use a linear activation everywhere?"}
Because composing linear functions always produces another linear function — no matter how many layers you add, the network would reduce to a single linear transform. Non-linear activations break this collapse and let the network model curved, complex relationships in data. Without them, depth buys you nothing.
:::

:::qa {q="What happens during the forward pass?"}
Input data flows layer by layer from left to right: each layer computes a weighted sum of its inputs, adds a bias, applies an activation function, and passes the result to the next layer. The final layer produces a prediction. No weights change during a forward pass — that only happens during backpropagation.
:::

:::drill {type="mcq" q="You initialize all weights in a 3-layer network to exactly 0.0. What happens during training?"}
- [ ] The network trains normally but converges slowly
- [ ] Only the output layer learns; hidden layers stay frozen
- [x] All neurons in each hidden layer learn identical features — the layers never diversify
- [ ] Weights explode to infinity on the first gradient step
:::

:::drill {type="mcq" q="Which activation function is most commonly used in hidden layers of modern feed-forward networks?"}
- [ ] Sigmoid, because it maps outputs cleanly to (0, 1)
- [ ] Tanh, because it is zero-centered
- [x] ReLU, because it is fast, avoids vanishing gradients for positive values, and works well in practice
- [ ] Softmax, because it normalizes across all neurons
:::

:::key-takeaway
A neural network is a stack of neurons, each computing a weighted sum plus a non-linear squish. Depth lets layers build hierarchical representations efficiently. Weights start random and garbage — training (backpropagation) is what makes the network useful.
:::
