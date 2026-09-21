# Regularization for Deep Models

**Course:** AMLSIZG511 – Deep Neural Networks (BITS Pilani WILP)
**Sources:** Module 11 slide deck (`DNN_CS05_Regularization.pdf`), `Regularization practice numbericals.pdf` + `regularization_15_solutions_line_by_line.pdf` (full worked solutions), Course Handout (T1 Ch.4, 7.5; R1 Ch.7 Goodfellow et al.)

---

## Table of Contents (this module)
1. Model Selection, Underfitting & Overfitting
2. Challenges: Vanishing/Exploding Gradients, Covariate Shift
3. Parameter Initialization (Xavier/Glorot, He)
4. L1 & L2 Regularization — full derivation
5. Dropout — algorithm & intuition
6. Batch Normalization & Layer Normalization
7. Early Stopping & Data Augmentation
8. Practical Guidance (selection, tuning, interactions, pitfalls)
9. Code, 15 Full Worked Numerical Practice Problems, Key Takeaways

---

## 1. Model Selection, Underfitting & Overfitting

**What is regularization?** *Adding constraints to prevent models from becoming too complex and memorizing training data.* Analogies from the slides: speed limits on roads (prevent overly aggressive driving), grammar rules (guide but don't fully restrict), budget constraints (force prioritization). In ML terms: constrains model complexity, improves generalization, adds useful inductive bias — often at the cost of *slightly worse* training performance in exchange for *better* test performance.

```
Underfitting            Just Right              Overfitting
   x → y                  x → y                    x → y
 (too simple,          (captures true          (too complex, fits
  high bias)            signal well)            noise, high variance)
```

**Why is regularization needed?** Modern networks have millions–billions of parameters — enough capacity to *memorize* an entire training set. Without regularization: perfect training accuracy, poor test performance, unstable training, high sensitivity to data variations. **The paradox:** more powerful models need *stronger* constraints to generalize well.

### 1.1 Regularization Taxonomy
```
Regularization
├── Explicit Regularization
│   ├── Parameter Penalties (L1, L2)
│   └── Structural (Dropout, DropBlock)
└── Implicit Regularization
    ├── Training Procedures (Early Stopping)
    └── Normalization (Batch/Layer Norm)
```

### 1.2 Bias–Variance Tradeoff

> **`Total Error = Bias² + Variance + Noise`**

Increasing model complexity typically **reduces bias but increases variance**. Regularization moves the operating point **left** on the complexity axis — it deliberately trades a small increase in bias for a larger reduction in variance, ideally landing near the sweet spot (minimum total error).

### 1.3 Model selection pipeline
`Dataset → Train/Val/Test split (e.g., 80/10/10) → Train models across multiple λ (regularization strengths) → Select best on validation → Final test`.

**K-Fold Cross-Validation:**
1. Divide training data into `k` folds.
2. For each fold `i`: train on the other `k-1` folds, validate on fold `i`.
3. Average validation performance across all `k` folds.
4. Select the hyperparameters with the best average performance.

**Benefits:** more reliable performance estimates, better use of limited data, reduced variance in hyperparameter selection.

### 1.4 Domain-specific regularization needs

| Domain | Primary challenges | Key techniques |
|---|---|---|
| Computer Vision | Large images, spatial correlations, translation invariance | Data augmentation, BatchNorm, DropBlock |
| NLP | Sequential dependencies, variable length, context | Dropout, LayerNorm, Attention Dropout |
| Time Series | Temporal patterns, non-stationarity, long dependencies | Recurrent Dropout, Early Stopping, Weight Decay |
| Tabular Data | Mixed types, feature interactions, small datasets | L1/L2, Dropout, Feature Selection |

---

## 2. Challenges in Deep Learning

### 2.1 Vanishing Gradient Problem
**Mathematical cause:** by the chain rule, the gradient at the first layer is a **product** of Jacobians across all layers:

```
frac∂ L∂ W^((1)) = frac∂ L∂ h^((L))(Π(l=2..L)frac∂ h^((l))∂ h^((l-1)))frac∂ h^((1))∂ W^((1))
```

If `‖frac∂ h^((l))∂ h^((l-1))‖ < 1` consistently, the product shrinks **exponentially** with depth `L` — gradients vanish, early layers learn very slowly (numerically demonstrated in Module 3, §8.1: a factor of `0.2` per layer shrank the gradient to 0.8% of its size within 3 layers). Common with sigmoid/tanh activations, RNNs over long sequences, and poor initialization.

### 2.2 Exploding Gradient Problem
**Cause:** the mirror-image case — if `‖frac∂ h^((l))∂ h^((l-1))‖ > 1`, the product grows exponentially. **Symptoms:** loss becomes NaN/Inf, very large weight updates, unstable training curves. **Solutions:** gradient clipping (Module 3, §8.1), better initialization (§3 below), batch normalization, residual/skip connections.

### 2.3 (Internal) Covariate Shift
**Definition:** the change in the distribution of a layer's inputs (activations) as the parameters of *earlier* layers keep changing during training. As layer 1's weights update, the input distribution seen by layer 2 shifts — forcing layer 2 to continually re-adapt to a moving target, which slows training and demands careful initialization/learning rates. **Solution:** Batch Normalization directly addresses this by normalizing layer inputs (§6).

### 2.4 Practical challenges
- **Hyperparameter sensitivity:** wrong regularization strength can hurt more than help; interactions between multiple techniques compound this. Mitigate via grid/random search, Bayesian optimization, LR scheduling, early stopping.
- **Computational overhead** of common techniques:

| Technique | Memory | Training time | Inference | Implementation |
|---|---|---|---|---|
| L1/L2 | +0% | +1–2% | 0% | Easy |
| Dropout | +5% | +10% | 0% | Easy |
| Batch Norm | +15% | +15% | +5% | Medium |
| Data Augmentation | +50% | +50% | 0% | Medium |
| Advanced Dropout (DropBlock etc.) | +20% | +20% | 0% | Hard |

- **Training instability:** gradient explosion/vanishing (batch norm + gradient clipping + good init), LR sensitivity (regularization changes the *optimal* LR — often need to reduce it), batch-size dependencies (batch norm is sensitive to batch size; layer norm is a more stable alternative).

---

## 3. Parameter Initialization

**Why it matters:** poor initialization can directly *cause* vanishing/exploding gradients, slows/breaks convergence, and if all weights start identical, neurons compute the same function forever (**symmetry problem**).

### 3.1 Common (naive) approaches and their problems
| Method | Issue |
|---|---|
| Zero init: `W=0` | All neurons identical → symmetry never breaks |
| Small random: `W~mathcal N(0,0.01²)` | Activations/gradients shrink toward zero through depth → vanishing gradients |
| Large random: `W~mathcal N(0,1²)` | Activations saturate (sigmoid/tanh) → exploding gradients / saturation |

### 3.2 Xavier / Glorot Initialization (for sigmoid / tanh)
**Principle:** keep the **variance** of activations (forward) and gradients (backward) roughly constant across layers.

> **`W~mathcal N(0,\ frac1n_(in)) or W~mathcal N(0,\ frac2n_(in)+n_(out))`**

> **`W~mathcal U(-√frac6n_(in)+n_(out),\ √frac6n_(in)+n_(out))`**

where `n_(in), n_(out)` are the fan-in/fan-out of the layer.

**Derivation sketch (why `1/n_(in)`):** for a linear neuron `z=Σ(i=1)^n_(in) w_i x_i` with i.i.d. `w_i,x_i` of mean 0, `Var(z) = n_(in)·Var(w)·Var(x)`. To keep `Var(z)=Var(x)` (activation variance preserved layer-to-layer), we need `Var(w) = 1/n_(in)`. Averaging the forward (`1/n_(in)`) and backward (`1/n_(out)`) requirements gives the symmetric form `2/(n_(in)+n_(out))`.

### 3.3 He Initialization (for ReLU)

```
W~mathcal N(0,\ frac2n_(in)) or W~mathcal U(-√frac6n_(in),\ √frac6n_(in))
```

**Reasoning:** ReLU zeros out (kills) roughly half the pre-activations on average (`z≤0 ⇒ a=0`), which halves the variance passed forward compared to a linear/symmetric activation. He initialization compensates by **doubling** the variance (`2/n_(in)` instead of Xavier's `1/n_(in)`) so the *effective* forward variance (after ReLU) still matches the input variance.

**Guideline:** ReLU/Leaky ReLU → **He** initialization; Sigmoid/Tanh → **Xavier/Glorot** initialization.

---

## 4. L1 and L2 Regularization

### 4.1 Formulation and comparison

| Property | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Penalty formula | `λΣ(i) w_i²` | `λΣ(i) lvert w_irvert` |
| Gradient of penalty | `2λ w` | `λ·sign(w)` |
| Effect on weights | Shrinks proportionally toward 0 | Drives weights to **exactly** 0 |
| Sparsity | No | Yes |
| Automatic feature selection | No | Yes |
| Geometry of constraint region | Circle/sphere (`Σ w_i²≤ t`) | Diamond/octahedron (`Σlvert w_irvert≤ t`) |
| Differentiable everywhere | Yes | No (kink at `w=0`) |
| Typical `λ` | `10⁻⁵`–`10⁻²` | `10⁻⁴`–`10⁻¹` |
| Use case | General smoothness | Feature selection |

### 4.2 Derivation of the weight update
Total loss: `L_(total)(w) = L(w) + λΣ(i) w_i²` (L2) or `L_(total)(w) = L(w)+λΣ(i)lvert w_irvert` (L1).

**L2 gradient:** `(∂)/(∂ w)[λ w²] = 2λ w`, giving the update:

> **`w ≤ftarrow w - η((∂ L)/(∂ w) + 2λ w) = (1-2ηλ)w - η(∂ L)/(∂ w)`**

This is exactly **weight decay**: at every step the weight is multiplicatively shrunk by a factor `(1-2ηλ) < 1` *in addition to* the usual gradient step — hence "L2 regularization ≡ weight decay" for plain SGD (this equivalence breaks for adaptive optimizers like Adam, motivating AdamW — Module 3, §6.6).

**L1 gradient (subgradient at 0):** `(∂)/(∂ w)[λlvert wrvert] = λ·sign(w)`, giving:

> **`w ≤ftarrow w - η((∂ L)/(∂ w) + λ sign(w))`**

Unlike L2 (which shrinks proportionally to the current magnitude, so it shrinks slower and slower as `w→0`), L1 subtracts a **constant** amount `ηλ` regardless of `w`'s magnitude — this constant "push toward zero" is precisely what can drive small weights to land exactly at 0, producing **sparsity**.

### 4.3 Geometric intuition for L1 sparsity
The L1 constraint region `{|w₁|+|w₂|≤ t}` is a diamond with **corners on the axes**. The loss function's contours (ellipses, generically) are much more likely to first touch this diamond **at a corner** (where one coordinate is exactly zero) than the L2 constraint region (a smooth circle, which the ellipse can touch anywhere on its boundary with no preference for the axes). This is why L1 tends to produce **sparse** solutions and L2 does not.

### 4.4 Elastic Net
Combines both penalties:

> **`L_(total) = L + λ[αΣ(i)lvert w_irvert + (1-α)Σ(i) w_i²]`**

`α∈[0,1]` interpolates between pure Lasso (`α=1`) and pure Ridge (`α=0`) — combining L1's sparsity with L2's stability (particularly useful when features are correlated, where pure L1 can behave erratically).

---

## 5. Dropout

### 5.1 Core idea
**Prevent complex co-adaptations** between neurons by randomly "dropping" (zeroing) a fraction `p` of units on every training pass — forcing the network to learn **redundant, robust representations** rather than relying on any single neuron or fragile combination of neurons.

### 5.2 Training-phase algorithm (per layer `l` with dropout)
```
for each training sample:
    for each layer l with dropout:
        m^(l) ~ Bernoulli(1-p)            # random binary mask
        h̃^(l) = h^(l) ⊙ m^(l)             # apply mask (zero out dropped units)
        h^(l+1) = f(W^(l+1) h̃^(l) + b^(l+1))   # forward pass as usual
```

### 5.3 Inference phase — "inverted dropout" scaling
At test time we want the **expected** output to match training-time behaviour. Since during training each unit's expected contribution was scaled by `(1-p)` (kept with probability `1-p`), at inference we must either:
- **(a)** scale activations by `(1-p)` at test time: `h^((l+1)) = f(W^((l+1))(1-p)h^((l))+b^((l+1)))`, **or**
- **(b, "inverted dropout", standard in modern frameworks)** scale by `1/(1-p)` **during training** instead, so **no change is needed at test time** — this is the version used in TensorFlow/PyTorch's `Dropout` layers.

### 5.4 Why dropout works (intuition)
- **Ensemble interpretation:** training with dropout approximates training (and then implicitly *averaging*) an exponential number of "thinned" sub-networks that share weights.
- **Co-adaptation prevention:** a neuron cannot rely on any specific other neuron always being present, forcing it to learn features that are useful **in combination with many different random subsets** of the other neurons — i.e., more generally useful, robust features.

### 5.5 Dropout variants for different architectures
| Variant | Mechanism | Best for |
|---|---|---|
| Standard Dropout | Random individual neurons | Fully connected layers (rate 0.5–0.8 hidden, 0.1–0.2 input) |
| DropConnect | Random individual **connections/weights** | General |
| DropBlock | Random contiguous spatial blocks (e.g., 7×7) | CNNs — dropping individual pixels is less effective since neighboring pixels are highly correlated |
| Attention Dropout | Dropout on attention weights | Transformers (usually lower rate 0.1–0.3) — prevents over-reliance on specific attention patterns |

---

## 6. Batch Normalization & Layer Normalization

### 6.1 Batch Normalization — motivation & algorithm
**Problem it solves:** without BN, each layer's input distribution keeps shifting during training (internal covariate shift, §2.3) — forcing every layer to constantly re-adapt.

**Training-phase algorithm** (mini-batch `mathcal B={x₁,…,x_m}`, learnable `γ,β`):

```
μ_(mathcal B) = (1)/(m)Σ(i=1..m) x_i σ_(mathcal B)² = (1)/(m)Σ(i=1..m) (x_i-μ_(mathcal B))²
```

> **`x̂_i = fracx_i-μ_(mathcal B)√σ_(mathcal B)²+ε y_i = gammax̂_i+β`**

Also maintain running statistics for inference: `μ_(running)≤ftarrow(1-α)μ_(running)+αμ_(mathcal B)` (similarly for `σ²_(running)`).

**Inference phase** (uses the accumulated running stats, not batch stats):

> **`x̂ = γ fracx-μ_(running)√σ²_(running)+ε+β`**

**Why `γ,β` are needed:** pure normalization forces every layer's input to have mean 0, variance 1 — but this may not be the *optimal* distribution for the next layer. The learnable scale `γ` and shift `β` let the network **undo** the normalization if that turns out to be better (in the limit, `γ=√σ_(mathcal B)²+ε, β=μ_(mathcal B)` recovers the original un-normalized `x`).

**Benefits:** allows much higher learning rates (10–100× faster training), less sensitive to initialization, acts as a mild regularizer (reduces need for dropout, because each sample's normalization depends on the other samples in its mini-batch — injecting a form of noise), enables training very deep networks.

#### Worked numerical example
Mini-batch, single feature: `x=[10,12,8,14]`.

> **`μ_(mathcal B) = (10+12+8+14)/(4)=11, σ²_(mathcal B) = (1+1+9+9)/(4)=5`**

Normalize (`ε=0.01`, `√(5.01)≈2.24`):

```
x̂₁=(10-11)/(2.24)=-0.45,\ x̂₂=(12-11)/(2.24)=0.45,\ x̂₃=(8-11)/(2.24)=-1.34,\ x̂₄=(14-11)/(2.24)=1.34
```

Scale & shift with `γ=2,β=3`:

```
y₁=2(-0.45)+3=2.1,\ y₂=2(0.45)+3=3.9,\ y₃=2(-1.34)+3=0.32,\ y₄=2(1.34)+3=5.68
```

**Verification:** normalized `x̂` has `μ=0,σ²=1`; after `γ,β` the network can restore whatever distribution is optimal.

### 6.2 Layer Normalization

**Why not always Batch Norm?** BN requires large, stable batch sizes; is unstable at batch size 1; behaves *differently* in train vs. test (running stats vs. batch stats); performs poorly for RNNs with variable-length sequences.

**Layer Norm solution:** normalize **across features**, for **each sample independently** (not across the batch) — so it works with any batch size and behaves identically at train/test.

For a sample `x∈mathbb R^d`:

```
μ = (1)/(d)Σ(i=1..d) x_i, σ²=(1)/(d)Σ(i=1..d)(x_i-μ)², x̂_i = (x_i-μ)/(√(σ²+ε)), y_i=γ_ix̂_i+β_i
```

No running statistics are needed — the same computation is used identically for both training and inference.

**Where used:** Transformers (essential — `LayerNorm(x + Attention(x))`, `LayerNorm(x + FFN(x))`; used in BERT/GPT/T5/all modern LLMs), RNNs/LSTMs/GRUs (handles variable sequence length naturally), and any small-batch/RL/online-learning scenario.

#### Worked numerical example
Single sample, 4 features: `x=[2,8,4,10]`.

> **`μ=(2+8+4+10)/(4)=6, σ² = (16+4+4+16)/(4)=10`**

Normalize (`ε=0.01`, `√(10.01)≈3.16`):

```
x̂₁=(2-6)/(3.16)=-1.27,\ x̂₂=(8-6)/(3.16)=0.63,\ x̂₃=(4-6)/(3.16)=-0.63,\ x̂₄=(10-6)/(3.16)=1.27
```

Scale & shift with `γ=[1,2,1,2],β=[0,1,0,1]`:

```
y₁=1(-1.27)+0=-1.27,\ y₂=2(0.63)+1=2.26,\ y₃=1(-0.63)+0=-0.63,\ y₄=2(1.27)+1=3.54
```

### 6.3 Batch Norm vs. Layer Norm

| Property | Batch Norm | Layer Norm |
|---|---|---|
| Normalization axis | Across the batch | Across features (per sample) |
| Depends on batch size? | Yes | No |
| Needs running statistics? | Yes (for inference) | No |
| Train vs. test computation | Different | Identical |
| Works at batch size 1? | Unstable | Works fine |
| Variable sequence length | Problematic | Perfect |
| Best for | CNNs | RNNs, Transformers |

---

## 7. Early Stopping & Data Augmentation

### 7.1 Early Stopping
**Concept:** monitor validation error during training; stop when it starts increasing (an **implicit** regularization technique — no explicit penalty term is added to the loss). Key parameters: **patience** (epochs to wait for improvement, typically 5–20), **min delta** (minimum improvement to count as progress, 0.001–0.01), **restore best** (use the best checkpoint, not the final one).

**Algorithm:**
```
Initialize best_val_error = ∞, patience_counter = 0, best_model = None
for epoch in 1..max_epochs:
    train_model_for_one_epoch()
    val_error = compute_validation_error()
    if val_error < best_val_error - min_delta:
        best_val_error = val_error
        patience_counter = 0
        best_model = save_model()          # new best checkpoint
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            break                          # early stopping triggered
restore_best_model()
```

### 7.2 Data Augmentation
**Concept:** artificially increase training data diversity by applying label-preserving transformations, reducing overfitting and improving robustness.
- **Images:** rotation, flipping, scaling, cropping, translation, color jitter/brightness, Cutout, Gaussian noise, **Mixup** (`tilde x=λ x_i+(1-λ)x_j`, `tilde y=λ y_i+(1-λ)y_j`), **CutMix** (paste a patch from one image onto another, mix labels proportionally to patch area).
- **Text:** synonym replacement, back-translation, random insertion/deletion.
- **Practical impact:** well-designed augmentation can improve generalization by **5–15%** with modest compute overhead; recommended to apply 2–3 augmentations per sample with ~50% probability each, and to increase augmentation strength progressively (baseline → basic → advanced → modern/Mixup-style).

---

## 8. Practical Guidance

### 8.1 Regularization selection decision tree (informal)
- Small/medium **tabular** data → L2 + Early Stopping + Cross-Validation (+ conservative data augmentation).
- **CNN** on images → Full suite: BatchNorm + Dropout + strong Data Augmentation.
- **NLP/Transformer**, limited resources → Dropout + LayerNorm.
- Very large models / high resources → Advanced techniques (label smoothing, stochastic depth, etc.); lightweight budget-constrained setups → Weight Decay + Early Stopping only.

### 8.2 Hyperparameter tuning priority order
1. **Learning rate** (most important!) — search range `[10⁻⁴,10⁻¹]`.
2. **L2 regularization strength** `λ∈[10⁻⁵,10⁻¹]`.
3. **Dropout rates** (different per layer) `∈[0.1,0.8]`.
4. **Batch size** (interacts with batch norm) `∈[16,512]` (powers of 2).
5. Data augmentation intensity.

### 8.3 Monitoring & troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| High training loss, slow learning | Underfitting | Reduce regularization, increase capacity |
| Large train–val gap (>15%) | Overfitting | Increase regularization, add data augmentation |
| Loss spikes / NaNs | Training instability | Gradient clipping, reduce LR |

Alert thresholds: train–val gap `>15%`; gradient norm `>5` or `<0.1`; validation loss trending up for 3+ epochs; weight magnitude `>1` or `<0.01`.

### 8.4 Regularization interactions

| Synergy | Effect |
|---|---|
| Batch Norm + Weight Decay | Complementary |
| Data Augmentation + Dropout | Different noise types, compound benefit |
| L2 + Early Stopping | Double protection against overfitting |

| Conflict | Resolution |
|---|---|
| Batch Norm + Dropout | Apply dropout **after** batch norm, not before (interference with normalization statistics) |
| High L1 + High Dropout | Excessive sparsity — use lower dropout with strong L1 |
| Strong Data Aug + Small dataset | Information loss — reduce augmentation intensity |

### 8.5 Common implementation mistakes
1. **Dropout during inference** — Wrong: keep dropout active at test time. Right: disable dropout (or use inverted-dropout scaling, §5.3) so predictions are deterministic.
2. **Batch Norm placement** — Wrong: BN → Conv → Activation. Right: **Conv → BN → Activation** (normalize the pre-activation, then apply the nonlinearity).
3. **Learning rate with added regularization** — regularization changes the effective loss landscape; often need to **reduce** LR slightly when adding strong regularization.

### 8.6 Architecture-specific regularization summary
- **CNNs:** DropBlock/SpatialDropout/Cutout (spatial-aware dropout), heavy data augmentation, BatchNorm after conv, Global Average Pooling instead of large FC layers, residual connections (implicit regularization).
- **RNNs:** Input/Output/Recurrent/Variational dropout variants, gradient clipping, Layer Normalization (more stable than BatchNorm for variable-length sequences).
- **Transformers:** Attention Dropout, Layer Dropout (skip whole layers), Label Smoothing, gradient accumulation, warmup+decay LR scheduling.

---

## 9. Code Implementation

### 9.1 L1/L2 regularized loss and gradient
```python
# Topic: L1, L2, and Elastic Net regularization — loss & gradient
# Purpose: Reproduce the formulas and worked numericals of §4 and §9 (practice problems)
import numpy as np

def l2_penalty(weights, lam):
    return lam * np.sum(weights ** 2)

def l1_penalty(weights, lam):
    return lam * np.sum(np.abs(weights))

def elastic_net_penalty(weights, lam, alpha):
    return lam * (alpha * np.sum(np.abs(weights)) + (1 - alpha) * np.sum(weights ** 2))

def l2_grad(weights, grad_loss, lam):
    return grad_loss + 2 * lam * weights          # weight-decay form

def l1_grad(weights, grad_loss, lam):
    return grad_loss + lam * np.sign(weights)

# Example: reproduce Q1/Q2 from the practice numericals
w = np.array([2.0, -3.0])
print("L1 total loss:", 10 + l1_penalty(w, 0.5))   # -> 12.5
print("L2 total loss:", 5 + l2_penalty(w, 0.1))    # -> 6.3
```

### 9.2 Dropout — forward & backward (inverted dropout)
```python
# Topic: Inverted Dropout layer
# Purpose: Standard training-time-only scaling so inference needs no change
import numpy as np

def dropout_forward(h, p, training=True):
    if not training:
        return h, None                      # no-op at inference (inverted dropout)
    mask = (np.random.rand(*h.shape) > p).astype(float) / (1 - p)   # scale by 1/(1-p)
    return h * mask, mask

def dropout_backward(dout, mask):
    return dout * mask                       # gradient only flows through kept units

# Example
h = np.array([1.0, 2.0, 3.0, 4.0])
out, mask = dropout_forward(h, p=0.5, training=True)
print("dropped output:", out)                # roughly half zeroed, rest scaled by 2
```

### 9.3 Batch Normalization — forward & backward
```python
# Topic: Batch Normalization forward/backward pass
# Purpose: Reproduce the §6.1 worked numerical example
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-2):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    cache = (x, x_hat, mu, var, gamma, eps)
    return out, cache

def batchnorm_backward(dout, cache):
    x, x_hat, mu, var, gamma, eps = cache
    m = x.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * std_inv**3, axis=0)
    dmu = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    dx = dx_hat * std_inv + dvar * 2 * (x - mu) / m + dmu / m
    return dx, dgamma, dbeta

# Reproduce §6.1 worked example
x = np.array([[10.0], [12.0], [8.0], [14.0]])
gamma, beta = np.array([2.0]), np.array([3.0])
out, _ = batchnorm_forward(x, gamma, beta, eps=0.01)
print(out.ravel())   # -> [2.1, 3.9, 0.32, 5.68]  (matches hand derivation)
```

### 9.4 Layer Normalization
```python
# Topic: Layer Normalization forward pass
# Purpose: Reproduce the §6.2 worked numerical example
import numpy as np

def layernorm_forward(x, gamma, beta, eps=1e-2):
    mu = x.mean()
    var = x.var()
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta

x = np.array([2.0, 8.0, 4.0, 10.0])
gamma = np.array([1.0, 2.0, 1.0, 2.0])
beta = np.array([0.0, 1.0, 0.0, 1.0])
y = layernorm_forward(x, gamma, beta, eps=0.01)
print(y)   # -> [-1.27, 2.26, -0.63, 3.54]  (matches hand derivation)
```

### 9.5 Xavier / He initialization
```python
import numpy as np

def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out))

def he_init(n_in, n_out):
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, size=(n_in, n_out))
```

---

## 10. 15 Full Worked Numerical Practice Problems
*(Adapted verbatim from `Regularization practice numbericals.pdf`, with full step-by-step solutions from `regularization_15_solutions_line_by_line.pdf`)*

### Part A: Regularization Penalties (Q1–Q10)

**Q1. L1 Regularization.** Given `w₁=2,\ w₂=-3,\ λ=0.5`, original loss `=10`.

> **`Σ|w_i| = |2|+|-3| = 5, penalty = 0.5×5=2.5, L_(total)=10+2.5=[12.5]`**

**Q2. L2 Regularization.** Given `w₁=2,\ w₂=-3,\ λ=0.1`, original loss `=5`.

> **`Σ w_i² = 4+9=13, penalty=0.1×13=1.3, L_(total)=5+1.3=[6.3]`**

**Q3. L1 vs L2.** Given `w₁=3,w₂=-4,w₃=2,λ=0.2`, loss `=8`.

> **`L1 penalty=0.2(3+4+2)=0.2×9=1.8 ⇒ L1_(total)=8+1.8=[9.8]`**

> **`L2 penalty=0.2(9+16+4)=0.2×29=5.8 ⇒ L2_(total)=8+5.8=[13.8]`**

**Q4. Ridge Regression.** Given `w₁=4,w₂=3`, MSE`=6`, `λ=0.5`.

> **`Σ w_i²=16+9=25, penalty=0.5×25=12.5, L_(Ridge)=6+12.5=[18.5]`**

**Q5. Lasso Regression.** Given `w₁=5,w₂=-2,w₃=1`, loss`=8`, `λ=0.4`.

> **`Σ|w_i|=5+2+1=8, penalty=0.4×8=3.2, L_(Lasso)=8+3.2=[11.2]`**

**Q6. L2 Weight Update.** Given `w=5,η=0.1,∂ L/∂ w=2,λ=0.5`. Using `w_(new)=w-η(∂ L/∂ w+2λ w)`:

> **`2λ w = 2(0.5)(5)=5, total grad=2+5=7, w_(new)=5-0.1(7)=5-0.7=[4.3]`**

**Q7. L1 Weight Update.** Given `w=4,η=0.1,∂ L/∂ w=3,λ=0.5` (`w>0`, so `sign(w)=1`):

> **`w_(new)=4-0.1(3+0.5)=4-0.1(3.5)=4-0.35=[3.65]`**

**Q8. Ridge with Multiple Weights.** Given `w₁=2,w₂=-5,w₃=3`, MSE`=12`, `λ=0.2`.

> **`Σ w_i²=4+25+9=38, penalty=0.2×38=7.6, L_(Ridge)=12+7.6=[19.6]`**

**Q9. Lasso with Multiple Weights.** Given `w₁=-4,w₂=6,w₃=-2`, MSE`=10`, `λ=0.3`.

> **`Σ|w_i|=4+6+2=12, penalty=0.3×12=3.6, L_(Lasso)=10+3.6=[13.6]`**

**Q10. Elastic Net.** Given `w₁=2,w₂=-3`, loss`=7`, `λ=0.5,α=0.6`.

```
Σ|w_i|=5, Σ w_i²=13, penalty=0.5[0.6(5)+0.4(13)]=0.5[3+5.2]=0.5×8.2=4.1
```

> **`L_(total)=7+4.1=[11.1]`**

### Part B: Regularized Cost Functions (Q1–Q5)

**Q1. L2 Regularized Cost.** `m=5`, errors `=2,-1,3,-2,1`; `θ₁=2,θ₂=-3`; `λ=0.4`.

> **`SSE=4+1+9+4+1=19, J(θ)=(19)/(2×5)=1.9`**

```
sumθ_j²=4+9=13, L2 penalty=(0.4)/(10)×13=0.52, J_(total)=1.9+0.52=[2.42]
```

**Q2. L1 Regularized Cost.** `m=4`, errors `=1,-2,2,-1`; `θ₁=3,θ₂=-4`; `λ=0.5`.

> **`SSE=1+4+4+1=10, J(θ)=(10)/(8)=1.25`**

```
Σ|θ_j|=3+4=7, L1 penalty=(0.5)/(8)×7=0.4375, J_(total)=1.25+0.4375=[1.6875]
```

**Q3. Compare Regularized vs Unregularized.** `m=5`, squared errors `=4,1,9,4,1`; `θ₁=2,θ₂=-2,θ₃=1`; `λ=0.5`.

> **`SSE=19, J(θ)=(19)/(10)=1.9\ (unregularized)`**

```
sumθ_j²=4+4+1=9, L2 penalty=(0.5)/(10)×9=0.45, J_(total)=1.9+0.45=[2.35]
```

> **`Increase due to regularization=2.35-1.9=[0.45]`**

**Q4. Ridge Cost Function.** `m=10`, SSE`=80`; `θ₁=3,θ₂=-4,θ₃=2`; `λ=0.2`.

> **`base cost=(80)/(20)=4, sumθ_j²=9+16+4=29, penalty=(0.2)/(20)×29=0.29`**

> **`J_(Ridge)=4+0.29=[4.29]`**

**Q5. Elastic Net Cost Function.** `m=6`, SSE`=48`; `θ₁=2,θ₂=-3`; `λ=0.6,α=0.7`.

> **`base cost=(48)/(12)=4, Σ|θ_j|=5,\ sumθ_j²=13`**

> **`bracket=0.7(5)+0.3(13)=3.5+3.9=7.4, penalty=(0.6)/(12)×7.4=0.37`**

> **`J_(EN)=4+0.37=[4.37]`**

---

## 11. Connections to Other Topics
- **Module 2 (Feedforward Networks):** vanishing gradients derived formally here (§2.1) reuse the exact chain-rule product structure from backpropagation.
- **Module 3 (Optimization):** weight decay (L2) interacts non-trivially with adaptive optimizers, motivating **AdamW**; gradient clipping addresses exploding gradients from both modules' perspectives.
- **Module 5 (CNNs):** DropBlock, spatial dropout, and BatchNorm-after-conv are CNN-specific instantiations of the general techniques here.
- **Module 6 (RNNs):** vanishing/exploding gradients are especially severe in RNNs (backprop-through-time); Layer Norm and recurrent dropout variants are the standard remedies.

---

## 12. Key Takeaways
1. Regularization trades a small increase in **bias** for a larger decrease in **variance**, improving generalization — necessary because modern over-parameterized networks can otherwise memorize training data.
2. **L2** shrinks weights proportionally (weight decay) and is smooth everywhere; **L1** subtracts a constant push toward zero and induces **sparsity** (automatic feature selection) — Elastic Net blends both.
3. **Dropout** is a stochastic, structural regularizer approximating an ensemble of thinned sub-networks; always use **inverted dropout** so no test-time rescaling is needed.
4. **Batch Norm** (across-batch) and **Layer Norm** (across-features, per-sample) both combat internal covariate shift, but have very different batch-size/sequence-length trade-offs — BN for CNNs, LN for RNNs/Transformers.
5. **Xavier/He initialization**, gradient clipping, and normalization layers are the primary tools against vanishing/exploding gradients; **early stopping** and **data augmentation** are cheap, broadly effective, implicit regularizers.

## 13. Common Mistakes & Misconceptions
- **Mistake:** applying standard dropout at test time. **Correction:** disable it (or use inverted dropout so no change is needed) — leaving it active makes predictions stochastic and typically worse.
- **Mistake:** placing BatchNorm before the convolution/linear layer. **Correction:** the standard order is **Linear/Conv → BatchNorm → Activation**.
- **Mistake:** assuming L2 regularization and weight decay are always identical. **Correction:** they are mathematically equivalent **only for plain SGD**; for Adam/RMSprop they diverge, which is exactly why **AdamW** exists (Module 3, §6.6).
- **Mistake:** using Batch Norm with very small or variable batch sizes (e.g., in RNNs or reinforcement learning with batch size 1). **Correction:** prefer **Layer Normalization** in these settings.
- **Mistake:** believing more regularization is always better. **Correction:** over-regularizing pushes the model from overfitting into **underfitting** — always tune strength via validation performance, not blind maximization.

## 14. Practice Problems (additional, for self-check)
1. Compute the L2-regularized loss for `w=[1,-2,3]`, `λ=0.3`, original loss `=4`.
   *Solution:* `Σ w²=1+4+9=14`; penalty`=0.3×14=4.2`; total`=4+4.2=8.2`.
2. Compute one L1 weight-update step for `w=-3` (negative weight!), `η=0.05`, `∂ L/∂ w=1`, `λ=0.4`.
   *Solution:* `sign(w)=-1`; update `=w-η(∂ L/∂ w+λ sign(w)) = -3-0.05(1-0.4)=-3-0.03=-3.03`.
3. For a BatchNorm layer with batch `x=[5,7,9,11]`, `γ=1,β=0,ε≈0`, compute the normalized outputs.
   *Solution:* `μ=8,σ²=5`; `x̂=[-1.34,-0.45,0.45,1.34]` (using `sqrt5≈2.24`); since `γ=1,β=0`, outputs equal `x̂`.
4. Explain, using the variance-preservation argument in §3.2, why using **Xavier** initialization (designed for symmetric activations) with a **ReLU** network would tend to shrink activation variance layer by layer.
   *Solution:* Xavier assumes the activation preserves variance symmetrically; ReLU zeros ~half its inputs, halving the forward variance at each layer relative to what Xavier assumes — compounding across depth causes activations (and gradients) to shrink toward zero, motivating He's compensating factor of 2.
5. A model shows training loss `=0.05` but validation loss `=0.45` (large gap). List 3 regularization changes you would try, in priority order, and justify each using §8.2/§8.3.
   *Solution:* (1) Increase L2/dropout strength — directly targets the overfitting gap; (2) add/strengthen data augmentation — increases effective data diversity; (3) reduce model capacity or add early stopping with tighter patience — directly limits the memorization opportunity.
