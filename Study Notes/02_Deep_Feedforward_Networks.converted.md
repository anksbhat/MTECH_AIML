# Deep Feedforward Neural Networks

**Course:** AMLSIZG511 – Deep Neural Networks (BITS Pilani WILP)
**Sources:** Contact Session 3 (`CS03_DNN.pdf` — Multi-class classification, DFNN, Activation Functions, Computational Graph, Forward Propagation), Module 5 Numerical Practice deck (`DNN_M5_DFNN_Numerical.pdf`), Course Handout (T1 Ch.3.4/Ch.4)

---

## Table of Contents (this module)
1. Core Concept — What is a Deep Feedforward Network?
2. Computational Graphs
3. Forward Propagation — scalar, vectorized, and matrix-form
4. Activation Functions (full comparison + derivatives)
5. Loss Functions and Softmax Regression
6. Backward Propagation — full chain-rule derivation
7. Complete Worked Numerical Example (forward + backward pass)
8. Gradient Flow / Impact of Depth
9. Designing a Network — layer & neuron sizing heuristics
10. Code, Practice Problems, Key Takeaways

---

## 1. Core Concept

A **Deep Feedforward Network** (DFNN), also called a **Multilayer Perceptron (MLP)**, is the quintessential deep learning model. Information flows in **one direction only** — from the input, through one or more **hidden layers**, to the output — with **no feedback/recurrent connections** (that distinguishes it from RNNs, Module 6).

**Goal:** approximate some function `f^*` (e.g., a classifier `y = f^*(x)`) by defining a mapping `ŷ = f(x;θ)` and learning the parameters `θ` that give the best approximation.

**Why "deep"?** The network is a *composition* of simpler functions:

> **`f(x) = f^((L))(f^((L-1))(·s f^((1))(x)·s))`**

Each `f^((l))` is one **layer**. The overall **depth** of the network is `L` (number of layers, excluding the input).

---

## 2. Computational Graphs

A **computation graph** formalizes how a complex expression is built from elementary operations, and it is the key data structure that makes automatic differentiation (backpropagation) possible.

**Definition:**
- Each **node** represents a variable (scalar, vector, matrix, tensor).
- Each **edge**/operation is a simple function of one or more input variables, returning **exactly one** output variable.
- If `y` is computed by applying an operation to `x`, we draw a directed edge `x → y`.

### 2.1 Worked toy example
Let `J(a,b,c) = 3(a+bc)`. Decompose into elementary operations:

> **`u = bc, v = a+u, J = 3v`**

```
a ──────╮
        ├── + ──► v ──► ×3 ──► J
b ─╮    │
   ├─×─►u
c ─╯
```

**Forward pass** (left→right): plug in numbers for `a,b,c`, compute `u`, then `v`, then `J`.

**Backward pass** (right→left, i.e., **backpropagation**): to find `∂ J/∂ v`, we "nudge" `v` slightly and observe the resulting change in `J`: since `J=3v`, `∂ J/∂ v = 3`. Then, using the **chain rule**, we push this derivative backward:

```
(∂ J)/(∂ u) = (∂ J)/(∂ v)·(∂ v)/(∂ u) = 3· 1 = 3, 
(∂ J)/(∂ a) = (∂ J)/(∂ v)·(∂ v)/(∂ a) = 3· 1 = 3
```

```
(∂ J)/(∂ b) = (∂ J)/(∂ u)·(∂ u)/(∂ b) = 3· c, 
(∂ J)/(∂ c) = (∂ J)/(∂ u)·(∂ u)/(∂ b) = 3· b
```

**Intuition:** every gradient in a neural network — no matter how deep — is computed exactly this way: propagate a local derivative backward through the graph, multiplying by the chain rule at every edge, and **accumulating** contributions when a node feeds into multiple downstream nodes.

### 2.2 Computational graph for binary classification (one neuron)

> **`z = w^Tx + b, a = ŷ = σ(z), L(a,y) = -[ylog a + (1-y)log(1-a)]`**

```
x1 ─╮
w1 ─┼─►(×,+ with b)──► z ──►σ──► a ──► L(a,y)
b  ─╯
```

---

## 3. Forward Propagation

### 3.1 Notation (formal)
For layer `l = 1, …, L`:
- `n^[l]`: number of units (neurons) in layer `l` (`n^[0] = n_x` = number of input features)
- `W^[l] ∈ ℝ^n^[l]× n^[l-1]`: weight matrix of layer `l`
- `b^[l] ∈ ℝ^n^[l]`: bias vector of layer `l`
- `z^[l]`: pre-activation ("net input") of layer `l`
- `a^[l] = g^[l](z^[l])`: activation (output) of layer `l`, with `a^[0] = x`
- `g^[l](·)`: activation function of layer `l` (may differ per layer, e.g., ReLU in hidden layers, sigmoid/softmax at output)

### 3.2 Per-layer forward equations (single example)

```
boxedz^[l] = W^[l]a^[l-1] + b^[l], boxeda^[l] = g^[l](z^[l]), l = 1,…,L
```

Final prediction: `ŷ = a^[L]`.

### 3.3 Vectorizing over `m` training examples
Stack the `m` examples column-wise: `A^[0] = X ∈ ℝ^(n_x × m)`. Then for each layer:

> **`Z^[l] = W^[l]A^[l-1] + b^[l], A^[l] = g^[l](Z^[l])`**

where `b^[l]` broadcasts across the `m` columns. This removes the explicit Python loop over training examples ("for i = 1 to m") and lets the whole batch be processed via matrix multiplication — critical for GPU efficiency.

**Dimension bookkeeping example** (from CS03 slides), for a 3-hidden-layer network with `n_x=2` inputs:

| Tensor | Shape |
|---|---|
| `a^[0]=x` | `(2,)` |
| `W^[1]` | `(5,2)` |
| `b^[1], z^[1], a^[1]` | `(5,)` |
| `W^[2]` | `(4,5)` |
| `b^[2], z^[2], a^[2]` | `(4,)` |
| `W^[3]` | `(3,4)` |
| `b^[3], z^[3], ŷ = a^[3]` | `(3,)` |

General rule: `W^[l]` has shape `(n^[l], n^[l-1])`, and `b^[l], z^[l], a^[l]` have shape `(n^[l],)` (or `(n^[l], m)` when vectorized over `m` examples).

### 3.4 Computational cost of the forward pass
- Forward pass = `L` matrix multiplications, each followed by an activation function.
- Matrix multiplication is highly parallelizable (GPUs) → forward pass is fast in practice.
- **Complexity is linear in depth**: `O(L)` sequential matrix-multiply + activation steps.

---

## 4. Activation Functions

### 4.1 Why non-linearity is essential
If every `g^[l]` were the identity function, then `a^[L] = W^[L]W^[L-1]·s W^[1]x + (bias terms)` — a composition of *linear* maps is itself just **one** linear map. Stacking layers would add no representational power at all. Non-linear activations are what let depth actually increase the function class the network can represent (recall the Universal Approximation discussion, Module 1).

### 4.2 Sigmoid (logistic)

> **`σ(z) = frac11+e^(-z), σ'(z) = σ(z)(1-σ(z))`**

- Range: `(0,1)`. Not zero-centered. Max derivative `= 0.25` at `z=0`.
- Used for binary classification output (probability interpretation).
- **Problem:** saturates for large `|z|` → gradient `→ 0` → vanishing gradients in deep nets (see §8).

### 4.3 Tanh (hyperbolic tangent)

> **`tanh(z) = frace^z-e^(-z)e^z+e^(-z), tanh'(z) = 1-tanh²(z)`**

- Range: `(-1,1)`. **Zero-centered** (an advantage over sigmoid for hidden layers — keeps gradients more balanced in sign).
- Max derivative `=1.0` at `z=0`. Still saturates for large `|z|`.
- Odd/anti-symmetric: `tanh(-z) = -tanh(z)`; derivative is even: `tanh'(-z)=tanh'(z)`.
- Worked values: `tanh(2.0)≈0.964`, `tanh(-1.5)≈-0.905`, `tanh(0.8)≈0.664`, `tanh(-0.3)≈-0.291`.

### 4.4 ReLU (Rectified Linear Unit)

> **`ReLU(z) = max(0,z), ReLU'(z) = 1 if z>0 ; 0 if z≤ 0`**

- No saturation for `z>0` → mitigates vanishing gradients, computationally cheap, default choice for hidden layers in modern DNNs.
- **Problem:** "dying ReLU" — if a unit's `z<0` persistently, its gradient is permanently 0 and it stops learning (leads to Leaky ReLU / Parametric ReLU variants: `f(z)=max(α z, z)` with small `α`, e.g., 0.01, so the unit still receives a small gradient when `z<0`).

### 4.5 Softmax (multi-class output)

> **`softmax(z)_j = frace^(z_j)Σ(k=1..K) e^(z_k)`**

- Converts a vector of raw scores ("logits") into a **probability distribution** over `K` classes: outputs are non-negative and sum to 1.
- For numerical stability, always subtract `max(z)` before exponentiating: `softmax(z)_j = e^(z_j-max(z))/Σ(k) e^(z_k-max(z))` (shown in the worked example, §7.2).

### 4.6 Comparison Table
*(from `DNN_M5_DFNN_Numerical.pdf`, "Activation Functions: Comparison")*

| Property | Sigmoid | Tanh | Softmax |
|---|---|---|---|
| Formula | `frac11+e^(-z)` | `frace^z-e^(-z)e^z+e^(-z)` | `frace^(z_j)Σ(k) e^(z_k)` |
| Range | `(0,1)` | `(-1,1)` | `(0,1)`, sums to 1 |
| Zero-centered? | No | Yes | No |
| Derivative | `σ(1-σ)` | `1-tanh²` | Complex (Jacobian) |
| Max derivative | 0.25 | 1.0 | Varies |
| Use case | Binary classification output | Hidden layers | Multi-class output |

| Loss | Formula | Derivative w.r.t. `ŷ` |
|---|---|---|
| MSE | `(1)/(2)(ŷ - y)²` | `ŷ - y` |
| MAE | `|ŷ - y|` | `sign(ŷ-y)` |
| BCE | `-[ylogŷ + (1-y)log(1-ŷ)]` | `(ŷ-y)/(ŷ(1-ŷ))` |
| Cross-Entropy (multi-class) | `-Σ(c) y_clog ŷ_c` | `-y_j/ŷ_j` (per class) |

> **Special, extremely important simplification:** when **Softmax is paired with Cross-Entropy loss**, or **Sigmoid is paired with Binary Cross-Entropy**, the gradient of the loss w.r.t. the *pre-activation logits* `z` collapses to the beautifully simple form:
> 
> **`(∂ L)/(∂ z) = ŷ - y`**

> This is why virtually every classification network uses this pairing — it avoids ever computing the individually messy activation/loss derivatives separately.

---

## 5. Loss Functions & Softmax Regression

**Softmax Regression** = multi-class generalization of logistic regression: linear layer `z = Wx+b` followed by softmax, trained with cross-entropy loss.

> **`L = -Σ(c=1..K) y_c log(ŷ_c), ŷ = softmax(z)`**

Because `y` is one-hot, this reduces to `L = -log(ŷ_(true class))` — the loss only "cares about" the predicted probability mass assigned to the correct class.

**Practical guidance** (from slides' Summary):
- Regression → MSE or MAE.
- Binary classification → Sigmoid + BCE.
- Multi-class classification → Softmax + Cross-Entropy.

---

## 6. Backward Propagation — Full Derivation

### 6.1 The two-way flow of computation
- **Forward** (left→right): compute the output `ŷ` and the loss `L`.
- **Backward** (right→left): compute `∂ L/∂θ` for every parameter `θ = {W^[l], b^[l]}`, using the chain rule, so that gradient descent can update them.

### 6.2 General layer-wise backprop equations
Define the **error signal** at layer `l`: `δ^[l] = dfrac∂ L∂ z^[l]`.

**Output layer** (`l=L`): for the sigmoid+BCE or softmax+CE special case,

> **`δ^[L] = a^[L] - y`**

**Hidden layers** (`l = L-1, …, 1`), propagate the error backward:

> **`δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g^([l]prime)(z^[l])`**

(`⊙` = element-wise product; this is exactly the computation graph's backward pass applied layer by layer.)

**Parameter gradients:**

> **`frac∂ L∂ W^[l] = δ^[l] (a^[l-1])^T, frac∂ L∂ b^[l] = δ^[l]`**

**Intuition:** `δ^[l]` measures "how much layer `l`'s pre-activation is to blame for the final loss." It is obtained by taking the *next* layer's blame `δ^[l+1]`, mapping it backward through that layer's weights `(W^[l+1])^T` (undoing the forward linear map), then multiplying by how sensitive layer `l`'s own activation function was at that point (`g^([l]prime)(z^[l])`) — exactly mirroring the chain rule derivation in §2.1.

### 6.3 Gradient descent parameter update

```
W^[l] ≤ftarrow W^[l] - η frac∂ L∂ W^[l], b^[l] ≤ftarrow b^[l] - η frac∂ L∂ b^[l]
```

where `η` is the learning rate (see Module 3 — Optimization for more advanced update rules).

---

## 7. Complete Worked Numerical Example
*(Adapted from `DNN_M5_DFNN_Numerical.pdf` — "Complete Neural Network Training")*

### 7.1 Network setup
2-layer network: Input (2 neurons) → Hidden (3 neurons, **ReLU**) → Output (2 neurons, **Sigmoid**).

```
x=[1.0; 0.5], 
W^[1]=[0.5, 0.2; -0.3, 0.8; 0.1, -0.4], 
b^[1]=[0.1; -0.2; 0.3]
```

```
W^[2]=[0.4, -0.1, 0.6; 0.2, 0.7, -0.3], 
b^[2]=[0.2; -0.1], y=[1; 0]
```

### 7.2 Forward Propagation

**Step 1 — Layer 1 pre-activation:**

```
z^[1] = W^[1]x+b^[1] = [0.5(1.0)+0.2(0.5); -0.3(1.0)+0.8(0.5); 0.1(1.0)-0.4(0.5)]+[0.1; -0.2; 0.3]
=[0.6; 0.1; -0.1]+[0.1; -0.2; 0.3]=[0.7; -0.1; 0.2]
```

**Step 2 — Apply ReLU:**

```
a^[1] = ReLU(z^[1]) = [max(0,0.7); max(0,-0.1); max(0,0.2)] = [0.7; 0.0; 0.2]
```

Note: neuron 2 is **"dead"** for this input (`z^[1]₂<0 ⇒ a^[1]₂ = 0`).

**Step 3 — Layer 2 pre-activation:**

```
z^[2] = W^[2]a^[1]+b^[2] = [0.4(0.7)-0.1(0)+0.6(0.2); 0.2(0.7)+0.7(0)-0.3(0.2)]+[0.2; -0.1]
= [0.4; 0.08]+[0.2; -0.1]=[0.6; -0.02]
```

**Step 4 — Apply Sigmoid:**

```
a^[2] = σ(z^[2]) = [frac11+e^(-0.6); frac11+e^(0.02)]=[0.646; 0.495]=ŷ
```

**Forward pass complete: `ŷ = [0.646,\ 0.495]^T`.**

### 7.3 Backward Propagation

**Step 1 — Loss (BCE):**

```
L=-(1)/(2)[y_1log(a₁^[2])+(1-y₁)log(1-a₁^[2]) + y_2log(a₂^[2])+(1-y₂)log(1-a₂^[2])]
```

> **`= -(1)/(2)[log(0.646)+log(0.505)] = -(1)/(2)[-0.437-0.683]=0.560`**

**Step 2 — Output layer error** (using the sigmoid+BCE shortcut `δ^[2]=a^[2]-y`):

> **`δ^[2] = [0.646; 0.495]-[1; 0]=[-0.354; 0.495]`**

**Step 3 — Output layer gradients:**

```
frac∂ L∂ W^[2] = δ^[2](a^[1])^T = [-0.354; 0.495][0.7, 0.0, 0.2]
=[-0.248, 0.000, -0.071; 0.347, 0.000, 0.099]
```

> **`frac∂ L∂ b^[2] = δ^[2] = [-0.354; 0.495]`**

**Step 4 — Hidden layer error** (backprop through `W^[2]` then apply ReLU′):

```
(W^[2])^Tδ^[2] = [0.4, 0.2; -0.1, 0.7; 0.6, -0.3][-0.354; 0.495]
= [-0.043; 0.382; -0.361]
```

> **`g^([1]prime)(z^[1]) = [1, (z₁=0.7>0); 0, (z₂=-0.1≤0); 1, (z₃=0.2>0)]`**

```
δ^[1] = (W^[2])^Tδ^[2]⊙ g^([1]prime)(z^[1]) = [-0.043; 0.382; -0.361]⊙[1; 0; 1]=[-0.043; 0.000; -0.361]
```

Note: the **dead ReLU neuron's gradient is forced to zero** — it receives no learning signal this step, a concrete illustration of the "dying ReLU" issue.

**Step 5 — Hidden layer gradients:**

```
frac∂ L∂ W^[1] = δ^[1]x^T = [-0.043; 0.000; -0.361][1.0, 0.5]
=[-0.043, -0.022; 0.000, 0.000; -0.361, -0.181], 
frac∂ L∂ b^[1] = δ^[1] = [-0.043; 0.000; -0.361]
```

These four gradients (`∂ L/∂ W^[1], ∂ L/∂ b^[1], ∂ L/∂ W^[2], ∂ L/∂ b^[2]`) are exactly what a gradient-descent optimizer needs to update all parameters (see Module 3).

### 7.4 Bonus worked example: Softmax + Cross-Entropy
Given logits `z = [2.0, 1.0, 0.1, -1.0]^T` and one-hot target `y=[0,1,0,0]^T`:

**Stabilized softmax:** subtract `max(z)=2.0`: `z_(stable)=[0,-1.0,-1.9,-3.0]`

> **`e^z_(stable)=[1.000, 0.368, 0.150, 0.050], Σ = 1.568`**

> **`softmax(z) = [0.638, 0.235, 0.096, 0.032] (sums to ≈1)`**

**Cross-entropy loss:** only the true class (`j=2`) contributes:

> **`L = -log(0.235) = 1.447`**

**Gradient (the "beautiful" result):**

```
(∂ L)/(∂ z) = ŷ - y = [0.638, 0.235, 0.096, 0.032]-[0,1,0,0] = [0.638, -0.765, 0.096, 0.032]
```

---

## 8. Gradient Flow / Impact of Depth

### 8.1 Vanishing gradients — worked numerical illustration
*(from `DNN_M5_DFNN_Numerical.pdf` — "Gradient Flow Analysis")*

Consider a 4-layer sigmoid network where every weight matrix has spectral norm `‖W^[l]‖=0.8`, and sigmoid's derivative is at its **maximum possible value** `g'=0.25` everywhere. The gradient propagation formula:

> **`|δ^[l-1]| = |δ^[l]|·‖W^[l]‖·|g'^[l-1]|`**

Starting from `|δ^[4]|=1.0` at the output:

| Layer | Computation | Magnitude |
|---|---|---|
| 4 (output) | given | 1.0 |
| 3 | `1.0×0.8×0.25` | 0.2 |
| 2 | `0.2×0.8×0.25` | 0.04 |
| 1 | `0.04×0.8×0.25` | 0.008 |

**Analysis:** the gradient shrinks by a factor `0.8×0.25=0.2` **at every layer**; after just 3 backward steps it is down to **0.8%** of its original size. The first (earliest) layer receives an almost negligible learning signal — this is the **vanishing gradient problem**, and it gets exponentially worse with depth when using saturating activations like sigmoid/tanh. (This directly motivates ReLU-family activations, careful weight initialization, batch normalization, and residual/skip connections — covered in Modules 4 and 5.)

### 8.2 Impact of depth — summary
- **Complexity** of forward and backward passes is `O(L)` — linear in depth, but the constant hidden in "`O`" (per-layer matmul cost) can be large.
- **Representational benefit** of depth: exponentially more efficient function representation vs. width alone (Module 1, §7–8).
- **Optimization cost** of depth: gradients must propagate through `L` nonlinear layers — the deeper the network, the more prone it is to vanishing/exploding gradients unless mitigated (see Module 4).

---

## 9. Designing a Network — Layer & Neuron Sizing Heuristics
*(from `DNN_M5_DFNN_Numerical.pdf` — "Design a Neural Network")*

A commonly used **rule-of-thumb constraint** relates dataset size `M`, input dimension `d`, and number of classes `c` to a reasonable **parameter budget** (to control overfitting): total parameters should not greatly exceed roughly `M/5`, and a bound on depth is:

> **`L ≤ log₂((M)/(d· c))+1`**

**Worked Example 1 — Small tabular dataset** (`d=20`, `M=800`, `c=3`):

> **`L ≤ log₂(800/60)+1 = log₂(13.33)+1≈ 4.7 ⇒ L≤4`**

Try `[20,40,3]`: parameters `=20×40+40+40×3+3=963` — exceeds budget `800/5=160`. Iteratively shrink: `[20,20,3]→483` (too many) → `[20,10,3]→243` (too many) → `[20,6,3]→147` ✅. **Final: `[20,6,3]` with regularization.**

**Worked Example 2 — Medium image dataset** (`d=784`, `M=5000`, `c=10`, MNIST-like):
Naively wide `[784,2352,784,10]` gives `≈3.7`M parameters — vastly over budget (`5000/5=1000`). Shrinking down: `[784,128,64,10]→109,000` → `[784,64,32,10]→52,458` → `[784,32,10]→25,450` → `[784,16,10]→12,714`. **Even the smallest options need regularization** (dropout, L2) since parameter counts are still far above the naive `M/5` heuristic — in practice deep-learning image models rely on architectural priors (CNNs, Module 5) rather than raw MLPs to control effective capacity.

**Worked Example 3 — Large text dataset** (`d=1000`, `M=50000`, `c=5`):

> **`L≤log₂(50000/5000)+1=log₂(10)+1≈4.3⇒ L≤4`**

Naive `[1000,5000,2000,1000,5]→≈17`M parameters — far too many. Reasonable choice: `[1000,512,256,128,5]→675,000` parameters, i.e. **13.5 parameters/sample** — a workable ratio. **Recommendation: use with batch normalization and dropout.**

**Takeaway:** these formulas are heuristics (not guarantees) — always validate final choices with held-out validation performance and regularization (Module 4).

---

## 10. Code Implementation

### 10.1 Pseudocode — generic L-layer forward + backward pass
```
# Forward
A[0] = X
for l in 1..L:
    Z[l] = W[l] @ A[l-1] + b[l]
    A[l] = g_l(Z[l])
Y_hat = A[L]
L_loss = loss(Y_hat, Y)

# Backward
delta[L] = Y_hat - Y                       # for sigmoid/softmax + matching loss
for l in L..1:
    dW[l] = delta[l] @ A[l-1].T / m
    db[l] = mean(delta[l], axis=1)
    if l > 1:
        delta[l-1] = (W[l].T @ delta[l]) * g_(l-1)_prime(Z[l-1])

# Update
for l in 1..L:
    W[l] -= eta * dW[l]
    b[l] -= eta * db[l]
```

### 10.2 Full Python/NumPy implementation reproducing the §7 worked example

```python
# Topic: Generic L-layer feedforward network — forward + backward propagation
# Purpose: Reproduce, in code, the hand-derived numerical example of §7
#          (2-layer net, ReLU hidden layer, Sigmoid output, BCE loss)
# Dependencies: numpy

import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([[1.0], [0.5]])
y = np.array([[1.0], [0.0]])

W1 = np.array([[0.5, 0.2], [-0.3, 0.8], [0.1, -0.4]])
b1 = np.array([[0.1], [-0.2], [0.3]])
W2 = np.array([[0.4, -0.1, 0.6], [0.2, 0.7, -0.3]])
b2 = np.array([[0.2], [-0.1]])

# ---- FORWARD PROPAGATION ----
Z1 = W1 @ x + b1
A1 = relu(Z1)
Z2 = W2 @ A1 + b2
A2 = sigmoid(Z2)          # = y_hat

print("z1 =", Z1.ravel())   # [ 0.7 -0.1  0.2]
print("a1 =", A1.ravel())   # [0.7 0.0 0.2]
print("z2 =", Z2.ravel())   # [ 0.6 -0.02]
print("y_hat =", A2.ravel())# [0.646 0.495]

# ---- LOSS (BCE, averaged over 2 outputs) ----
L = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
print("Loss =", L)          # 0.560

# ---- BACKWARD PROPAGATION ----
delta2 = A2 - y                              # output-layer error (sigmoid+BCE shortcut)
dW2 = delta2 @ A1.T
db2 = delta2

dA1 = W2.T @ delta2
delta1 = dA1 * relu_derivative(Z1)           # dead-ReLU neuron -> gradient forced to 0
dW1 = delta1 @ x.T
db1 = delta1

print("dW2 =\n", dW2)
print("dW1 =\n", dW1)

# ---- GRADIENT DESCENT UPDATE (single step, lr = 0.1) ----
lr = 0.1
W1 -= lr * dW1; b1 -= lr * db1
W2 -= lr * dW2; b2 -= lr * db2
```

**Expected output** matches the hand-derived numbers in §7 exactly (up to rounding): `z1=[0.7,-0.1,0.2]`, `a1=[0.7,0.0,0.2]`, `y_hat=[0.646,0.495]`, `Loss=0.560`, `dW2=[[-0.248,0.0,-0.071],[0.347,0.0,0.099]]`.

### 10.3 Softmax + Cross-Entropy from scratch
```python
# Topic: Numerically stable Softmax + Cross-Entropy, with its gradient
# Purpose: Reproduce worked example in §7.4
import numpy as np

def softmax(z):
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)

z = np.array([2.0, 1.0, 0.1, -1.0])
y = np.array([0, 1, 0, 0])              # one-hot target (class index 1)

probs = softmax(z)
loss = -np.log(probs[np.argmax(y)])
grad = probs - y                        # beautifully simple gradient

print("softmax:", np.round(probs, 3))    # [0.638 0.235 0.096 0.032]
print("loss:", round(loss, 3))           # 1.447
print("grad:", np.round(grad, 3))        # [ 0.638 -0.765  0.096  0.032]
```

### 10.4 Code Walkthrough
- `relu_derivative(Z1)` returns a 0/1 mask, directly implementing `g'[z]` from §6.2 — this is the exact same mechanism that zeroed out neuron 2's gradient in the hand derivation.
- The BCE + sigmoid shortcut (`delta2 = A2 - y`) and the softmax + CE shortcut (`grad = probs - y`) are used everywhere in practice instead of computing the activation and loss derivatives separately via the full chain rule — both are algebraically equivalent to doing so, but numerically simpler and more stable.
- `W2.T @ delta2` implements the backward pass through the linear layer (`(W^[l+1])^Tδ^[l+1]` from §6.2).

---

## 11. Practical Examples

### 11.1 Real dataset application
The same forward/backward mechanics above are exactly what students implement "from scratch" in the course's Programming Assignment 1 (`Deep Neural Networks-Assignment-1.pdf`): a **single-neuron linear network** for regression (MSE loss, batch gradient descent) and a **single-neuron sigmoid network** for binary classification (BCE loss, SGD) — both are the `L=1` special case of the general layer equations in §3–6.

### 11.2 Common Variations
- Different activation per layer is normal (ReLU in hidden layers, sigmoid/softmax only at the output for classification, or linear/identity at the output for regression).
- Mini-batch vs. full-batch vs. single-example (online/stochastic) forward-backward passes — all use identical equations, just different `m` in the vectorized forms (Module 3).

---

## 12. Connections to Other Topics
- **Module 1 (Fundamentals):** the 2-layer XOR MLP there is the `L=2` special case of the general equations derived here.
- **Module 3 (Optimization):** the gradients computed via backprop here are exactly the inputs consumed by SGD/Momentum/Adam.
- **Module 4 (Regularization):** vanishing gradients (§8.1) directly motivate careful weight initialization, batch normalization, and residual connections.
- **Module 5 (CNNs) / Module 6 (RNNs):** both reuse the same forward/backward propagation & computational-graph machinery, just with structured (convolutional / recurrent) weight sharing instead of dense `W^[l]`.

---

## 13. Key Takeaways
1. A DFNN is a chain of linear transforms + non-linear activations; **computation graphs** make the forward and backward passes systematic and automatable.
2. Backpropagation is **repeated application of the chain rule**, propagating an "error signal" `δ^[l]` backward and multiplying by local activation derivatives at each layer.
3. Pairing **sigmoid+BCE** or **softmax+cross-entropy** yields the clean gradient `ŷ - y` at the output — always prefer these standard pairings.
4. **ReLU** avoids the vanishing-gradient problem of sigmoid/tanh for `z>0`, but can "die" for `z≤0` — both illustrated numerically in §7 and §8.
5. Layer/neuron count should be chosen with a **parameter-budget** relative to dataset size in mind — bigger is not automatically better; regularization is usually still required (Module 4).

## 14. Common Mistakes & Misconceptions
- **Mistake:** forgetting to subtract `max(z)` before exponentiating in softmax. **Correction:** always use the numerically-stable form to avoid overflow.
- **Mistake:** treating the "beautiful" gradient `ŷ - y` as specific to any activation in isolation. **Correction:** it only holds for the **matched pairs** (sigmoid+BCE or softmax+CE) — using, e.g., MSE with a sigmoid output does **not** simplify this way.
- **Mistake:** assuming deeper is always better. **Correction:** depth increases both representational power *and* vanishing-gradient risk; §8.1's numeric example shows gradients can shrink to <1% within 3 layers under naive sigmoid stacking.

## 15. Practice Problems

1. **Given** `z^[1]=[1.2,-0.5,0.3]` with ReLU activation, compute `a^[1]` and `g'^[1](z^[1])`.
   *Solution:* `a^[1]=[1.2,0,0.3]`; `g'=[1,0,1]`.
2. **For the §7 network**, if instead `W^[2]` used **no bias** (`b^[2]=0`), recompute `z^[2]` and `a^[2]`.
   *Solution:* `z^[2]=[0.4,0.18]`, `a^[2]=σ(0.4)=0.599,\ σ(0.18)=0.545`.
3. **Compute the cross-entropy loss** for logits `z=[1.0,3.0,0.2]`, true class index 1.
   *Solution:* stabilize: `z-max=[-2.0,0,-2.8]`; `e^z=[0.135,1,0.061]`; sum`=1.196`; softmax`=[0.113,0.836,0.051]`; `L=-log(0.836)=0.179`.
4. **Vanishing gradient extension:** repeat the §8.1 calculation for a 6-layer network under the same assumptions (`‖W‖=0.8`, `g'=0.25`). What is `|δ^[1]|`?
   *Solution:* factor `0.2` per layer, 5 backward steps from layer 6 to layer 1: `|δ^[1]| = 1.0×0.2⁵ = 0.00032` (0.032% of original) — even more severe vanishing.
5. **Design problem:** for `d=50` features, `M=2000` samples, `c=4` classes, apply the depth heuristic `L≤log₂(M/(d c))+1` and propose one candidate architecture with parameter count checked against `M/5`.
   *Solution:* `L≤log₂(2000/200)+1=log₂(10)+1≈4.3⇒ L≤4`. Try `[50,20,4]`: params `=50×20+20+20×4+4=1084` vs budget `2000/5=400` — too many; try `[50,8,4]→50×8+8+8×4+4=444` — still slightly over; try `[50,6,4]→50×6+6+6×4+4=334≤400` ✅.
