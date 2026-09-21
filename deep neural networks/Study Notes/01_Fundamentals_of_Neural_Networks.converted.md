# Fundamentals of Neural Networks

**Course:** AMLSIZG511 – Deep Neural Networks (BITS Pilani WILP)
**Sources:** Contact Session 1–2 slide decks (`CS-1 and 2-DNN.pdf`), Course Handout (T1: Zhang et al., *Dive into Deep Learning*; R1: Goodfellow et al., *Deep Learning*), Supplementary notebooks (`Copy_of_AND/OR/XOR_Gate_Implementation.ipynb`)

---

## Table of Contents (this module)
1. Core Concept — What is Deep Learning, and why now?
2. The Anatomy of a Learning Problem (Data, Model, Objective, Optimizer)
3. Types of Learning Problems (Supervised / Unsupervised / Semi-supervised / RL)
4. Biological vs Artificial Neuron
5. The Perceptron & Perceptron Learning Algorithm
6. Logic Gates as Linear Classifiers (AND / OR / NAND / NOR)
7. The XOR Problem and the Birth of the MLP
8. Multilayer Perceptron (MLP), Universal Approximation
9. Issues of Depth vs Width
10. Code, Practice Problems, Key Takeaways

---

## 1. Core Concept

### 1.1 AI → ML → DL
- **AI**: the science of making machines perform tasks that (currently) require human intelligence.
- **ML**: an approach to AI where the machine *learns* the mapping from data instead of being explicitly programmed.
- **DL (Deep Learning)**: a subfield of ML that learns **successive layers of increasingly meaningful representations** directly from raw data. The "deep" refers to the number of layers (the **depth**) that transform the data.

> **Definition (working):** Deep Learning is a class of machine learning algorithms based on artificial neural networks with **three or more layers** in which each layer learns to transform its input representation into a slightly more abstract one, so the network as a whole learns a hierarchy of features (edges → textures → parts → objects, for example).

### 1.2 Why Deep Learning, and why now?
The ideas behind neural networks are 60+ years old (Perceptron, 1958). What changed to make deep learning practical circa 2012 onward:

| Driver | Why it matters |
|---|---|
| **Data** | Huge volumes of unstructured data (images, text, audio, video) now available (ImageNet, Common Crawl, etc.) |
| **Compute** | Cheap GPUs/TPUs, distributed clusters — training a network that used to take weeks now takes hours |
| **Storage** | Cheap large-scale storage to hold datasets and model checkpoints |
| **Automated feature learning** | No manual feature engineering — the network learns its own features from raw data |
| **Scalability** | Performance keeps improving with more data + more parameters (unlike classic ML, which plateaus) |

**Historical breakthroughs** (from slides, credit CMU 11-785): ImageNet/AlexNet (vision), AlphaGo & AlphaGo Zero (game-playing/self-play RL), neural image captioning (2015), and generative models (Stable Diffusion, ChatGPT).

### 1.3 Common Misconceptions
- ❌ "Deep learning is a black box that can't be understood at all." → It is *harder* to interpret than linear models, but the computation is fully deterministic and mathematically transparent (this whole document derives it!). Interpretability is an active research area, not an impossibility.
- ❌ "More layers always means a better model." → Depth increases *capacity*, but without enough data/regularization it causes overfitting, vanishing gradients, and harder optimization (see Modules 3–4).
- ❌ "Neural networks work like the brain." → They are only loosely *inspired* by biological neurons; the mechanisms (backpropagation, weight sharing) have no verified biological analogue.

---

## 2. The Anatomy of a Learning Problem

Every deep learning problem — regardless of the task — reduces to four core components:

1. **Data** — a collection of `m` examples `{(x^((i)), y^((i)))}_(i=1)^m`, converted into a numeric representation. Each example has **features** (or covariates) and, for supervised problems, a **label**/target.
   - *Fixed-length* data (images with fixed resolution) vs *variable-length* data (text, audio) — the model architecture must match the data's structure.
2. **Model** — the computational machinery (a parameterized function `f_θ`) that maps input data to predictions. In DL, `f_θ` is built from many **chained transformations** (layers).
3. **Objective / Loss function** — a formal, differentiable measure of how good/bad the model's predictions are. By convention, **lower is better** (hence "loss").
   - Regression → commonly **Mean Squared Error (MSE)**.
   - Classification → commonly **Cross-Entropy / error rate**.
4. **Optimization algorithm** — a procedure (typically a variant of **gradient descent**) that searches for parameters `θ` minimizing the loss on the training data.

### Intuition — worked mini-example (wake-word detector)
1. Precisely define inputs (audio snippets) and outputs (yes/no).
2. Choose a flexible **model family** (a set of programs parameterized by `θ`) — same family can serve "Hey Siri" or "Alexa" style tasks.
3. Collect labeled examples (audio + wake/no-wake label).
4. Run a **learning algorithm** — a meta-program that uses the data to select the parameter values `θ` that make the model behave well, i.e., **training**.

### Generalization
Doing well on the *training set* is not the goal — the goal is to generalize to **unseen data**. A model that fits training data very well but performs poorly on new data is said to be **overfitting** (see Module 4 — Regularization).

---

## 3. Types of Learning Problems

| Type | Signal available | Example |
|---|---|---|
| **Supervised** | Input `x` **and** label `y` | Classification, regression |
| **Unsupervised** | Only input `x`, no labels | Clustering, dimensionality reduction, representation learning |
| **Semi-supervised** | Small labeled set + large unlabeled set | Using unlabeled data to regularize/pre-train, then fine-tune with few labels |
| **Reinforcement Learning (RL)** | An **agent** interacts with an **environment**, receiving **rewards** instead of direct labels | AlphaGo, robotics control |

### When to consider (deep) neural networks?
- Input is high-dimensional (discrete or real-valued), possibly noisy.
- Output is discrete, real-valued, or a vector.
- The **form** of the true target function is unknown / complex.
- Human interpretability of the *mechanism* is not critical (only the *output* matters).
- Examples: speech phoneme recognition, image classification, financial prediction.

---

## 4. Biological vs Artificial Neuron

### 4.1 Biological Neuron — Key Components
| Component | Function |
|---|---|
| **Dendrites** | Receive signals from other neurons |
| **Soma (Cell body)** | Integrates/processes incoming signals |
| **Axon** | Transmits the output signal |
| **Synapse** | Connection point to other neurons; synaptic strength encodes learned knowledge |

### 4.2 Mapping to the Artificial Neuron

```
Biological Neuron        Artificial Neuron
------------------        ------------------
Dendrites          →      Inputs (x1, x2, ..., xn)
Synapse (strength) →      Weights (w1, w2, ..., wn)
Soma (integration) →      Weighted sum + bias  (z = Σ wi*xi + b)
Axon (firing)      →      Activation function f(z) → output ŷ
```

### 4.3 Mathematical model of an artificial neuron

> **`z = Σ(i=1..n) w_i x_i + b, ŷ = f(z)`**

Where:
- `x_i`: input feature `i`
- `w_i`: weight (importance) associated with feature `i`
- `b`: bias term (shifts the decision boundary)
- `f(·)`: activation function (identity, sigmoid, step, ReLU, ... — see Module 2)
- `ŷ`: neuron's output (prediction)

**Intuition:** the neuron computes a *weighted vote* over its inputs, shifts the vote by a bias, then squashes/transforms the result through a non-linearity to decide its output.

### 4.4 Comparison Table

| Biological Neuron | Artificial Neuron |
|---|---|
| Complex biochemical processes | Simple mathematical model |
| Analog signal processing | Digital computation |
| Adaptive synaptic strengths | Adjustable numeric weights |
| Massively parallel | Parallel computation possible (matrix ops) |
| Learns through experience | Learns through algorithms (gradient descent) |
| Fault tolerant | Deterministic behaviour |

### 4.5 Connectionism
A **Connectionist Machine** is a network of simple processing elements (artificial neurons) whose *world knowledge* is stored entirely in the **connections (weights)** between them, not in explicit symbolic rules. Neural networks are the quintessential connectionist model.

**Properties of ANNs:** many neuron-like threshold/switching units; many weighted interconnections; highly parallel, distributed processing; emphasis on *automatically* tuning weights (learning), not hand-coding them.

---

## 5. The Perceptron & Perceptron Learning Algorithm

### 5.1 Core Concept
Introduced by **Frank Rosenblatt (1958)**, the Perceptron is the simplest artificial neuron: a linear threshold unit.

> **`z = Σ(i) w_i x_i + b, h = step(z) = 1 if z ≥ 0 ; 0 if z < 0`**

Geometrically, the perceptron represents a **hyperplane decision surface** in `n`-dimensional input space: it outputs `1` for points on one side of the hyperplane, and `0` (or `-1`) for points on the other side. It can only separate classes that are **linearly separable**.

### 5.2 Perceptron Learning Algorithm — Derivation

**Update rule** (for target `t`, output `h`, learning rate `η`):

> **`Δ w_i = η (t - h) x_i, Δ w₀ = η (t-h)\ (bias term, x₀=1)`**

> **`w_i ≤ftarrow w_i + Δ w_i`**

**Derivation / intuition:** The perceptron uses the **error-correction rule**:
- If prediction is correct (`t = h`): error `= 0` → **no update**.
- If the perceptron predicts `0` but target is `1` (`t-h=1`): increase `w_i` in proportion to `x_i` (push `z` upward for inputs that were "on").
- If the perceptron predicts `1` but target is `0` (`t-h=-1`): decrease `w_i` in proportion to `x_i`.

This is a **stochastic gradient descent** step on the (non-differentiable) 0/1 error using the "perceptron criterion" — it does not require calculus because the step function's sub-gradient is approximated by the sign of the error.

**Convergence (Perceptron Convergence Theorem):** If the training data is **linearly separable**, the perceptron learning rule is guaranteed to find a separating hyperplane in a finite number of steps. If the data is **not linearly separable** (e.g., XOR), the algorithm **never converges** — weights oscillate forever.

### 5.3 Worked Example — Learning NOT gate (from CS-1/2 slides)

Single input `x₁`, target `t = NOT(x₁)`, `η = 1`, initial `w₁ = w₀ = 0` (here `w₀` is the bias):

**Epoch 1:**
| `x₁` | `t` | `w₁` | `w₀` | `z=w_1x₁+w₀` | `h=step(z)` | correct? | update |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 1 | ✅ | none |
| 1 | 0 | 0 | 0 | 0 | 1 | ❌ | `Δ w₁ = 1(0-1)(1) = -1`, `Δ w₀ = 1(0-1) = -1` → `w₁=-1, w₀=-1` |

**Epoch 2** (`w₁=-1, w₀=-1`):
| `x₁` | `t` | `z` | `h` | correct? | update |
|---|---|---|---|---|---|
| 0 | 1 | `-1` | 0 | ❌ | `Δ w₁ = 1(1-0)(0)=0`, `Δ w₀=1(1-0)=1` → `w₁=-1, w₀=0` |
| 1 | 0 | `-1+0=-1` | 0 | ✅ | none |

**Epoch 3** (`w₁=-1, w₀=0`): both examples now classified correctly → **converged**. Final: `w₁=-1, w₀=0`, i.e. `h = step(-x₁)`, which correctly implements NOT.

### 5.4 Representing AND / OR gates as perceptrons

Truth tables:

**AND**
| A | B | A∧B |
|---|---|---|
|0|0|0|
|0|1|0|
|1|0|0|
|1|1|1|

**OR**
| A | B | A∨B |
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|1|

Both are **linearly separable** (a single straight line can separate the 1-class from the 0-class), so a single perceptron suffices. Example weight sets that work (many valid solutions exist):
- AND: `w₁=1, w₂=1, b=-1.5` → `z = x₁+x₂-1.5`; only `(1,1)` gives `z>0`.
- OR: `w₁=1, w₂=1, b=-0.5` → `z = x₁+x₂-0.5`; any input with at least one 1 gives `z>0`.

The decision boundary is the line `w_1x₁ + w_2x₂ + b = 0` (equivalently `ax₁+bx₂+d=0`, slope `m=-a/b`, intercept `c=-d/b`).

**Exercise (from slides):** derive parameters for NOR and NAND gates yourself — they are just the *complements* of OR and AND, so negate all weights and bias of the corresponding gate.

---

## 6. The XOR Problem

### 6.1 Why a single perceptron fails

**XOR** truth table:
| A | B | A⊕B |
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

**Proof of non-linear-separability:** Suppose a line `w_1x₁+w_2x₂+b=0` could separate class-1 points `{(0,1),(1,0)}` from class-0 points `{(0,0),(1,1)}`. Then we need:

```
b<0 (from (0,0)), w₁+b≥ 0 (from (1,0)), w₂+b≥ 0 (from (0,1)), w₁+w₂+b<0 (from (1,1))
```

Adding the middle two inequalities: `w₁+w₂+2b ≥ 0 ⇒ w₁+w₂+b ≥ -b > 0` (since `b<0`). This **contradicts** the fourth inequality `w₁+w₂+b<0`. Hence **no** linear separator exists — XOR is **not linearly separable**.

Geometrically, the two "1" points and two "0" points of XOR sit at *diagonally opposite corners* of the unit square — no single straight line can isolate one diagonal from the other.

### 6.2 The Solution: Multilayer Perceptron (MLP)

Stack a **hidden layer** between input and output. The hidden layer performs a non-linear transformation of the input into a new (higher-dimensional or re-mapped) space in which the transformed problem *becomes* linearly separable at the output layer.

```
        Input Layer      Hidden Layer        Output Layer
        x1 ──●╲          ●  h1  ╲
             ╲ ╲        ╱  \    ╲
        x2 ──●──╲──────●    h2 ──●── ŷ
                 ╲      ╲  ╱    ╱
                  ╲      ●h3  ╱
                   ╲    ╱ ...╱
                    (weights W1)  (weights W2)
```

This is why the hidden layer is called "hidden" — its outputs `h₁, h₂, …` are not directly observed as final predictions; they simply feed the next neuron.

**Key conceptual leap:** the network *learns its own intermediate features* (`h₁,h₂,…`) instead of a human hand-designing them.

---

## 7. Multilayer Perceptron (MLP): Classifiers & Universal Approximators

### 7.1 MLP as a classifier
An MLP with 1+ hidden layers, each followed by a non-linear activation, can carve out **arbitrarily complex, non-linear decision regions** — unlike a single perceptron restricted to a single hyperplane.

### 7.2 Universal Approximation Theorem (statement & intuition)
> **Theorem (Cybenko 1989 / Hornik 1991, informal statement):** A feedforward network with a single hidden layer containing a **finite** number of neurons, using a (non-polynomial, e.g. sigmoid) activation function, can approximate **any continuous function** on a compact subset of `ℝ^n` to **arbitrary accuracy**, given enough hidden units.

**Intuition:** think of each hidden neuron with a sigmoid activation as producing a smooth "step" (ridge) in some direction of input space. A weighted sum of many such steps, positioned and scaled appropriately, can approximate any smooth surface — analogous to how a Fourier series approximates a periodic function using enough sinusoids, or how enough small rectangular blocks can approximate any 2D shape.

**Caveats:**
- The theorem guarantees **existence**, not that gradient descent will *find* such a network, nor how many hidden units are needed (could be exponentially many).
- This is why **depth** (multiple hidden layers) is preferred in practice over pure **width** (one very wide hidden layer) — depth can represent the same function with **exponentially fewer** total neurons, because it reuses/composes simpler features hierarchically.

---

## 8. Issues of Depth and Width

**Definitions** (from slides):
- **Depth** `h` of a network = number of layers, **including** the output layer but **excluding** the input layer.
- **Width** `d_m` of a network = the maximum number of neurons in any single layer.

Two ways to increase model capacity:

| Increase **Width** | Increase **Depth** |
|---|---|
| More parallel features per layer | Hierarchical, composed features |
| Higher raw capacity | More expressive per-parameter |
| Risk of overfitting grows quickly with width | Generally better generalization for the same parameter count |
| Easier to parallelize / train | Sequential processing, but can represent complex functions with far fewer neurons overall |

**Practical implication:** A **deep** network can represent complex functions using **far fewer neurons** than a **shallow, wide** network representing the same function (this follows from composition — reusing a small vocabulary of features at each layer to build higher-order features, rather than needing an exponential number of "flat" features).

**Advantages of Deep Learning:** excels on unstructured/complex data — images, video, natural language, audio.

**Disadvantages:** the "black-box" interpretability problem, overfitting risk, need for large data, and high computational cost.

---

## 9. Code Implementation

### 9.1 Perceptron for AND gate — pseudocode

```
Initialize w1, w2, b randomly (or zero)
for epoch in 1..E:
    for each training example (x1, x2, t):
        z = w1*x1 + w2*x2 + b
        h = step(z)            # or sigmoid(z) for a "soft" perceptron
        error = t - h
        w1 += lr * error * x1
        w2 += lr * error * x2
        b  += lr * error
```

### 9.2 Full working Python: AND gate via gradient-based perceptron
*(Adapted from supplementary notebook `Copy_of_AND_Gate_Implementation.ipynb`, Webinar-1 material)*

```python
# Topic: Perceptron (soft, sigmoid) — AND gate
# Purpose: Learn AND-gate mapping via gradient descent on binary cross-entropy
# Dependencies: numpy, matplotlib
# Source: Supplementary notebook Copy_of_AND_Gate_Implementation.ipynb

import numpy as np

# 1. AND GATE DATA
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [0], [0], [1]], dtype=float)

# 2. Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Initialize weights & bias
np.random.seed(42)
w1, w2, b = np.random.randn(), np.random.randn(), np.random.randn()

learning_rate = 0.5
epochs = 1000

for epoch in range(1, epochs + 1):
    # ---- Forward propagation ----
    z = w1 * X[:, 0] + w2 * X[:, 1] + b
    y_pred = sigmoid(z)

    # ---- Binary cross-entropy loss ----
    loss = -np.mean(
        y.flatten() * np.log(y_pred + 1e-8)
        + (1 - y.flatten()) * np.log(1 - y_pred + 1e-8)
    )

    # ---- Backpropagation (gradients of BCE w.r.t. z reduce to (y_pred - y)) ----
    error = y_pred - y.flatten()
    dw1 = np.mean(error * X[:, 0])
    dw2 = np.mean(error * X[:, 1])
    db = np.mean(error)

    # ---- Gradient descent update ----
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db

print(f"Final: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")

# Example usage — verify learned AND gate:
z = w1 * X[:, 0] + w2 * X[:, 1] + b
probs = sigmoid(z)
preds = (probs >= 0.5).astype(int)
for i in range(4):
    print(X[i], "->", probs[i], "pred:", preds[i], "target:", int(y[i, 0]))
# Expected output: predictions match [0,0,0,1] closely after training
```

### 9.3 Full working Python: solving XOR with a 2-layer MLP (manual backprop)
*(Adapted from supplementary notebook `Copy_of_XOR_gate_implementation.ipynb` — demonstrates why a hidden layer is required)*

```python
# Topic: 2-layer MLP solving XOR via manual forward/backward propagation
# Purpose: Show that a hidden layer of nonlinear units solves the non-linearly-
#          separable XOR problem, where a single perceptron cannot.
# Dependencies: numpy
# Source: Supplementary notebook Copy_of_XOR_gate_implementation.ipynb

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR targets

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):     # a = sigmoid(z) already computed
    return a * (1 - a)

np.random.seed(42)
input_neurons, hidden_neurons, output_neurons = 2, 4, 1

W1 = np.random.randn(input_neurons, hidden_neurons) * 0.5
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.5
b2 = np.zeros((1, output_neurons))

learning_rate = 1.0
epochs = 2500

for epoch in range(1, epochs + 1):
    # ---- FORWARD PROPAGATION ----
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)                 # hidden activations
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)                 # network output

    # ---- LOSS (binary cross-entropy) ----
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

    # ---- BACKPROPAGATION (chain rule, layer by layer) ----
    dZ2 = A2 - y                                  # dL/dZ2 (BCE + sigmoid simplifies nicely)
    dW2 = np.dot(A1.T, dZ2) / len(X)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)                       # backprop error into hidden layer
    dZ1 = dA1 * sigmoid_derivative(A1)             # apply local derivative of sigmoid
    dW1 = np.dot(X.T, dZ1) / len(X)
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # ---- UPDATE WEIGHTS ----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch in [1, 10, 100, 1000, 2500]:
        print(f"Epoch: {epoch:5d} | Loss: {loss:.6f}")

# Final predictions
Z1 = np.dot(X, W1) + b1; A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2) + b2; A2 = sigmoid(Z2)
predictions = (A2 >= 0.5).astype(int)
print(predictions.flatten())   # Expected: [0 1 1 0]  -- matches XOR!
```

### 9.4 Code Walkthrough
- **Forward pass:** input → linear transform (`Z₁ = XW₁+b₁`) → nonlinearity (`A₁=σ(Z₁)`) → linear transform (`Z₂=A_1W₂+b₂`) → nonlinearity (`A₂=σ(Z₂)`).
- **Loss:** Binary Cross-Entropy, appropriate for binary classification with a sigmoid output.
- **Backpropagation:** because BCE loss combined with a sigmoid output has the elegant simplification `∂ L/∂ Z₂ = A₂ - y` (derived in Module 2), the output-layer gradient is just the prediction error. This error is then **propagated backward** through `W₂^T` into the hidden layer, and multiplied by the **local derivative** of the hidden activation (`σ'(Z₁) = A₁(1-A₁)`) — this is the chain rule in action (fully derived with numbers in `02_Deep_Feedforward_Networks.md`).
- **Why this works for XOR but a single perceptron fails:** `W₁` maps the 2D XOR input into a 4-dimensional hidden representation in which the classes become linearly separable; `W₂` then just needs to draw one hyperplane in that new space.

---

## 10. Practical Examples

### 10.1 Simple hand-calculated example (OR gate)
Choose `w₁=1, w₂=1, b=-0.5` (a valid solution — not unique):

| `x₁` | `x₂` | `z=x₁+x₂-0.5` | step(`z`) | target |
|---|---|---|---|---|
|0|0|-0.5|0|0|
|0|1|0.5|1|1|
|1|0|0.5|1|1|
|1|1|1.5|1|1|

All four rows match target → this weight vector is a valid OR-gate perceptron.

### 10.2 Real-world application
- Perceptron-style linear classifiers are the foundation of **logistic regression** used in Assignment 1 (binary classification on heart-disease / diabetes datasets) — a "soft" perceptron with sigmoid output IS logistic regression.
- MLPs solving XOR-like non-linear boundaries generalize directly to real tabular datasets where classes are not linearly separable (e.g., loan default prediction).

### 10.3 Common Variations
- **Step-function perceptron** (Rosenblatt's original, non-differentiable, uses error-correction rule) vs **sigmoid perceptron** (differentiable, trained with gradient descent — bridges to logistic regression and modern NNs).
- Multi-class extension of the output layer uses **softmax** instead of sigmoid (see Module 2).

---

## 11. Connections to Other Topics
- **Deep Feedforward Networks (Module 2):** generalizes the MLP here into arbitrary depth, formalizes forward/backward propagation with computation graphs, and introduces more activation functions.
- **Optimization (Module 3):** the gradient descent update used to train the perceptron/MLP above is the simplest form of the optimization algorithms studied there (SGD, momentum, Adam).
- **Regularization (Module 4):** overfitting introduced in Section 2 motivates L1/L2, dropout, and other techniques.
- **CNNs (Module 5):** CNNs are MLPs with structured, weight-shared connectivity suited to grid-like data (images); the same forward/backward propagation principles apply.

---

## 12. Key Takeaways
1. Deep learning = learning hierarchical, successive representations of data via layered neural networks; enabled today by data + compute + storage abundance.
2. Every learning problem reduces to **Data + Model + Objective + Optimizer**.
3. A single perceptron can only represent **linearly separable** functions (proved impossible for XOR); this motivated the **multilayer perceptron**.
4. The **Universal Approximation Theorem** guarantees a sufficiently wide single-hidden-layer network can approximate any continuous function — but **depth** achieves this far more parameter-efficiently than width alone.
5. Depth vs width is a capacity/generalization trade-off: width → parallel features, higher risk of overfitting; depth → hierarchical composition, generally better generalization per parameter.

## 13. Common Mistakes & Misconceptions
- **Mistake:** Believing the perceptron learning rule always converges. **Correction:** It only converges if data is linearly separable (Perceptron Convergence Theorem); otherwise it oscillates forever — always check separability or switch to a smooth loss + gradient descent (which converges to a local minimum region even for non-separable data).
- **Mistake:** Assuming a hidden layer must be "observed"/interpreted directly. **Correction:** hidden units learn arbitrary, often uninterpretable, intermediate features — that's expected and desired (automated feature learning).
- **Mistake:** Confusing depth definition. **Correction:** depth counts hidden **and** output layers, but **not** the input layer.

## 14. Practice Problems

1. **Compute perceptron parameters for NAND.** *(Solution: NAND = NOT(AND); negate AND's weights and bias: if AND used `w₁=1,w₂=1,b=-1.5`, NAND uses `w₁=-1,w₂=-1,b=1.5`. Check: `(1,1)→ z=-1-1+1.5=-0.5<0→ h=0` ✓; all others `→ z≥ 0 → h=1` ✓.)*
2. **Prove NOR is linearly separable and give a valid weight set.** *(Solution: NOR = NOT(OR). Using OR's `w₁=1,w₂=1,b=-0.5`, NOR's weights are `w₁=-1,w₂=-1,b=0.5`. Verify all 4 rows.)*
3. **Why can't a 2-input, 0-hidden-layer perceptron solve XOR, but a 2-input-2-hidden-neuron-1-output MLP can?** *(Solution: XOR's positive class `{(0,1),(1,0)}` and negative class `{(0,0),(1,1)}` are not linearly separable — proved by contradiction in §6.1. A hidden layer with (e.g.) an AND-like unit and an OR-like unit lets the output neuron compute (OR AND NOT-AND) = XOR, a linear combination in the new 2D hidden space where classes ARE separable.)*
4. **Given a trained perceptron with `w₁=2, w₂=-1, b=0`, sketch/describe the decision boundary.** *(Solution: boundary is line `2x₁ - x₂ = 0 ⇒ x₂ = 2x₁`, a line through the origin with slope 2; points above the line get `h=0`, below get `h=1`, depending on sign convention.)*
5. **Depth vs Width:** *A network with 1 hidden layer of 1000 neurons vs. a network with 10 hidden layers of 50 neurons each — same total neuron count (1000 vs 500... adjust to compare fairly). Which is likely to generalize better on image data, and why?* *(Solution: the deep, narrower network typically generalizes better on structured/hierarchical data like images because it composes simple features into complex ones across layers, requiring exponentially fewer total units than a shallow network attempting to represent the same function directly, per the depth-efficiency argument in §7.2/§8.)*
