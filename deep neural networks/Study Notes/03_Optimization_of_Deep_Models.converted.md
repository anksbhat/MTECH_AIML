# Optimization of Deep Models

**Course:** AMLSIZG511 – Deep Neural Networks (BITS Pilani WILP)
**Sources:** Module 10 slide deck (`DNN_CS04_Optimizers.pdf`), Course Handout (T1 Ch.11 — Zhang et al., *Dive into Deep Learning*)

> Note: this module deck consistently tracks **one toy problem** — `L(w₁,w₂) = w₁² + 4w₂²`, starting at `(w₁,w₂)=(4,2)` — through every optimizer, which makes direct numeric comparison possible. We preserve this thread throughout §5–8.

---

## Table of Contents (this module)
1. Why Optimization Matters
2. Optimization Challenges: Saddle Points, Plateaus, Non-Convexity
3. Gradient Descent Variants (Batch / SGD / Mini-batch)
4. The Learning-Rate Effect & the Toy Problem
5. Momentum Methods (Momentum, Nesterov)
6. Adaptive Methods (AdaGrad, RMSprop, Adam, AdamW)
7. Learning Rate Schedules
8. Practical Optimization (gradient clipping, batch size, stopping criteria)
9. Comparison Summary, Code, Practice Problems

---

## 1. Why Optimization Matters

> **`θ^* = argmin(θ) L(θ)`**

Deep networks have millions–billions of parameters `θ = {W₁,b₁,W₂,b₂,…}` and **no closed-form solution** exists for `θ^*` — we must use **iterative** optimization. Without it: random weights → random predictions → no learning at all.

### 1.1 One iteration of gradient descent, by hand
`f(x,y)=x²+y²`, `∇ f = [2x,2y]`, start `(x₀,y₀)=(3,4)`, `η=0.1`:
- Before: position `(3,4)`, loss `=9+16=25`, gradient `=[6,8]`.
- Update: `x₁ = 3-0.1(6)=2.4`, `y₁=4-0.1(8)=3.2`.
- After: loss `=f(2.4,3.2)=5.76+10.24=16` — a **36% reduction** in one step.

### 1.2 Loss functions used with optimizers (recap)
| Loss | Formula | Gradient | Use case |
|---|---|---|---|
| MSE | `(1)/(2)(ŷ-y)²` | `(ŷ-y)` | Regression |
| MAE | `|ŷ-y|` | `sign(ŷ-y)` | Regression w/ outliers |
| BCE | `-[ylogŷ+(1-y)log(1-ŷ)]` | `(ŷ-y)` | Binary classification |
| Cat. Cross-Entropy | `-Σ(k) y_klogŷ_k` | `(ŷ-y)` | Multi-class |

**Key observation:** almost all these gradients reduce to "(prediction − target)" — this simple, clean form is *why* gradient-based optimization of neural nets is tractable at scale.

---

## 2. Optimization Challenges

### 2.1 Saddle points
A point where `∇ L=0` but it is **not** a minimum — some directions curve up, others curve down. Increasingly **common** (not rare) in high-dimensional parameter spaces, because for a random critical point, the probability that *all* eigenvalues of the Hessian have the same sign shrinks quickly as dimensionality grows.

### 2.2 Plateaus
Flat regions where `‖∇ L‖≈0` but the point is far from a minimum. Learning stalls; it is hard to tell (from the loss curve alone) whether training is stuck or making extremely slow progress.

### 2.3 Non-convex landscapes
Deep network losses have **multiple local minima**; there's no guarantee of finding the global minimum, and different weight initializations lead to different final solutions. (In practice, most local minima found by SGD-family methods on over-parameterized networks tend to generalize similarly well — but this is an empirical/theoretical nuance, not a guarantee.)

### 2.4 How to identify these in practice
- Training loss curve flattens, but validation error is still high.
- Gradient norms approach zero (`‖∇ L‖→0`) while loss is not yet low.
- Training progress stagnates for many epochs.

```
Loss
 │╲
 │ ╲___Plateau___         ___Saddle___
 │            ╲__       _╱          ╲___
 │               ╲_____╱   Local Min     ╲___ Global Min
 └───────────────────────────────────────────► Parameters
```

---

## 3. Gradient Descent Variants

| Method | Gradient used | Updates / epoch | Memory | Convergence behaviour |
|---|---|---|---|---|
| **Batch GD** | `(1)/(N)Σ(i=1..N) ∇ L_i` (full dataset) | 1 | High | Smooth, slow |
| **SGD** (pure stochastic) | `∇ L_i` (one sample) | `N` | Low | Noisy, fast per-step |
| **Mini-batch SGD** | `(1)/(B)Σ(i∈ B)∇ L_i` | `N/B` | Medium | Balanced |

**Concrete example:** `N=10,000` samples, batch size `B=32`:
- Batch GD: **1** update per epoch.
- SGD: **10,000** updates per epoch.
- Mini-batch: `10,000/32≈` **313** updates per epoch.

**Practical choice:** mini-batch with `B∈{32,64,128,256}` — balances GPU efficiency (large matrix multiplies) against convergence stability (enough averaging to reduce noise, but frequent enough updates to make fast progress).

### 3.1 Algorithm: Mini-Batch Gradient Descent
```
Input: learning rate η, batch size B, initial parameters θ
while stopping criterion not met:
    shuffle training data
    for each mini-batch of size B:
        L(θ) = (1/B) Σ_{i∈batch} L_i(θ)
        g = ∇θ L(θ) = (1/B) Σ_{i∈batch} ∇θ L_i(θ)
        θ ← θ - η g
```

### 3.2 The running Toy Problem

```
L(w₁,w₂) = w₁² + 4w₂² (an "elongated bowl" — w₂ direction is 4× steeper)
```

> **`∇ L = [2w₁,\ 8w₂]`**

Start: `(w₁,w₂)=(4,2)`, Loss `=16+16=32`. Because the `w₂` direction is steeper, a **fixed learning rate** forces an awkward compromise: small enough to not overshoot in `w₂`, but then too slow in `w₁`. This single toy problem will be re-used for every method below to allow direct, apples-to-apples comparison.

### 3.3 Mini-batch GD — full numerical trace (`η=0.1`)

| Iter | Loss | `∇ w₁` | `∇ w₂` | `w₁≤ftarrow w₁-η∇ w₁` | `w₂≤ftarrow w₂-η∇ w₂` |
|---|---|---|---|---|---|
| 0 | 32.00 | 8.0 | 16.0 | `4-0.1(8)=3.20` | `2-0.1(16)=0.40` |
| 1 | 10.88 | 6.4 | 3.2 | `3.20-0.1(6.4)=2.56` | `0.40-0.1(3.2)=0.08` |
| 2 | 6.58 | 5.1 | 0.64 | `2.56-0.1(5.1)=2.05` | `0.08-0.1(0.64)=0.016` |
| 3 | 4.21 | 4.1 | 0.13 | `2.05-0.1(4.1)=1.64` | `0.016-0.1(0.13)=0.003` |

**Observation:** loss falls `32→10.88→6.58→4.21`. **Issue:** `w₂` moves `Δ w₂=-1.6` in the first step vs. `Δ w₁=-0.8` — the steeper direction dominates, creating unbalanced/oscillatory progress. This directly motivates Momentum and adaptive per-parameter rates.

### 3.4 Learning-rate effect
| `η` | Behaviour | Example trace |
|---|---|---|
| Too large (`η=1.0`) | Oscillates / diverges | `15.3→8.2→19.7` (diverging) |
| Too small (`η=0.0001`) | Barely improves | `22.5→21.8→20.1` (needs 1000s of iters) |
| Optimal (`η=0.01`) | Fast, smooth convergence | `22.5→15.2→3.1` |

---

## 4. Motivation for Adaptive Techniques

Fixed LR problem: **all parameters share one `η`**, but different directions in the loss landscape have very different curvature (steepness). From the toy-problem trace:

| Parameter | Actual gradients (iters 0–3) | Problem | Needs |
|---|---|---|---|
| `w₁` | `[8.0,6.4,5.1,4.1]` | moderate, decreasing slowly | moderate LR |
| `w₂` | `[16.0,3.2,0.64,0.13]` | large initially, drops fast | adaptive LR |

Also: **sparse features** (rare) need larger updates when they *do* appear; **dense features** (frequent) need smaller, careful updates — another argument for **per-parameter** adaptive rates rather than one global `η`.

---

## 5. Momentum Methods

### 5.1 Momentum — physical intuition & derivation
**Analogy:** a ball rolling downhill accumulates velocity in consistent directions, and the accumulated inertia **dampens oscillations** in directions that keep reversing sign (perpendicular to the true descent direction) — while also helping it roll through small bumps (shallow local minima).

> **`v_t = β v_(t-1) + ∇_θ L(θ_t), θ_(t+1) = θ_t - η v_t`**

- `v_t`: velocity (an exponentially-weighted moving average of past gradients — **not** the gradient directly).
- `β`: momentum coefficient, typically `0.9` (i.e., 90% of previous velocity retained).

**Algorithm: SGD with Momentum**
```
Initialize velocity v = 0
while not converged:
    for each mini-batch:
        g = (1/B) Σ ∇θ L_i(θ)
        v ← β v + g
        θ ← θ - η v
```

### 5.2 Momentum — numerical trace (`η=0.05`, `β=0.9`, reduced LR vs. plain GD)

| Iter | Loss | `[∇ w₁,∇ w₂]` | `v₁≤ftarrow β v₁+∇ w₁` | `v₂≤ftarrowβ v₂+∇ w₂` | `w₁` | `w₂` |
|---|---|---|---|---|---|---|
| 0 | 32.00 | [8, 16] | 8 | 16 | 3.6 | 1.2 |
| 1 | 18.72 | [7.2, 9.6] | `7.2+7.2=14.4` | `14.4+9.6=24` | 2.88 | 0 |
| 2 | 8.29 | [5.76, 0] | `13.0+5.76=18.7` | `21.6+0=21.6` | 1.94 | `-1.08` |
| 3 | 8.43 | [3.88, `-8.64`] | `16.9+3.88=20.7` | `19.4-8.64=10.8` | 0.91 | `-1.62` |

**Key lesson:** `w₂` still oscillates (`2→1.2→0→-1.08→-1.62`) — momentum **amplifies** steps (it's a moving average of gradients, which can overshoot), so it typically needs a **smaller** `η` than vanilla GD to remain stable. (Loss even ticks back up slightly at iter 3: `8.29→8.43`.)

### 5.3 Nesterov Accelerated Gradient (NAG)
**Standard Momentum:** compute gradient at *current* `θ_t`, add to velocity, then step.
**Nesterov Momentum:** "look ahead" — jump to the anticipated next position `θ_t-β v_(t-1)` **first**, compute the gradient *there*, then update velocity and step:

> **`v_t = β v_(t-1) + ∇ L(θ_t - β v_(t-1)), θ_(t+1) = θ_t - η v_t`**

**Benefit:** more responsive to changes in the gradient (it "corrects" itself before overshooting) — slightly faster, more stable convergence than plain momentum, at negligible extra cost (one extra gradient evaluation location, not an extra pass).

---

## 6. Adaptive Methods

### 6.1 AdaGrad
**Idea:** adapt the learning rate **per parameter** using the *history* of squared gradients — parameters with historically large gradients get a smaller effective LR, and vice versa.

> **`s_t = s_(t-1) + g_t⊙ g_t, θ_(t+1) = θ_t - (η)/(√(s_t)+ε)⊙ g_t`**

`ε=10⁻⁸` for numerical stability. **Problem:** `s_t` accumulates *without bound* → `η/√(s_t)→0` eventually → learning **stops** permanently, even if the true minimum hasn't been reached. **Use case:** sparse features (e.g., NLP with large vocabularies where most gradients are zero most of the time).

### 6.2 RMSprop — fixing AdaGrad
**Fix:** replace the ever-growing sum with an **exponential moving average** (EMA) of squared gradients, so old gradients are gradually "forgotten":

> **`s_t = γ s_(t-1) + (1-γ) g_t², θ_(t+1) = θ_t - (η)/(√(s_t)+ε)g_t`**

Typical hyperparameters: `γ=0.9`, `η=0.001`, `ε=10⁻⁸`.

**Algorithm:**
```
Initialize s = 0
while not converged:
    for each mini-batch:
        g = (1/B) Σ ∇θ L_i(θ)
        s ← γ s + (1-γ) g⊙g
        θ ← θ - η/√(s+ε) ⊙ g
```

**Numerical trace** (`η=0.05,γ=0.9,ε≈0`), iteration 0 in full detail:

> **`g₁=2w₁=8, g₂=8w₂=16`**

> **`s₁ = 0.9(0)+0.1(8²)=6.4, s₂=0.9(0)+0.1(16²)=25.6`**

```
eff η₁=(0.05)/(√(6.4))=(0.05)/(2.53)=0.020, eff η₂=(0.05)/(√(25.6))=(0.05)/(5.06)=0.010
```

> **`w₁≤ftarrow4-0.02(8)=3.84, w₂≤ftarrow2-0.01(16)=1.84`**

| Iter | Loss | `[∇ w₁,∇ w₂]` | `s₁` | `s₂` | eff `η₁` | eff `η₂` | `w₁` | `w₂` |
|---|---|---|---|---|---|---|---|---|
| 0 | 32.00 | [8, 16] | 6.4 | 25.6 | 0.020 | 0.010 | 3.84 | 1.84 |
| 1 | 28.30 | [7.68, 14.72] | 11.7 | 44.7 | 0.015 | 0.0075 | 3.72 | 1.73 |
| 2 | 25.80 | [7.44, 13.84] | 16.0 | 59.3 | 0.0125 | 0.0065 | 3.63 | 1.64 |
| 3 | 23.93 | [7.26, 13.12] | 19.7 | 70.6 | 0.011 | 0.0060 | 3.55 | 1.56 |

**Observation:** larger gradients (the `w₂` direction) → larger `s₂` → automatically **smaller** effective LR — exactly the desired per-parameter adaptation. Loss decreases steadily (`32→28.30→25.80→23.93`) but **slowly**, because RMSprop has **no velocity/momentum term**.

**Iter-3 loss comparison so far:** Mini-batch GD `=4.21`, Momentum `=8.43`, RMSprop `=23.93` (RMSprop is the slowest here — because on this toy problem the well-tuned fixed-LR GD is actually already quite effective; RMSprop's strength shows on problems with more heterogeneous, non-stationary gradient scales).

### 6.3 Adam: Adaptive Moment Estimation
**Idea:** combine **Momentum** (1st moment: EMA of gradients) with **RMSprop** (2nd moment: EMA of squared gradients).

> **`m_t = β₁ m_(t-1)+(1-β₁)g_t (1st moment: momentum-like)`**

> **`v_t = β₂ v_(t-1)+(1-β₂)g_t² (2nd moment: RMSprop-like)`**

> **`θ_(t+1) = θ_t - η(m_t)/(√(v_t)+ε)`**

Default hyperparameters: `β₁=0.9`, `β₂=0.999`, `η=0.001` or `0.0003`, `ε=10⁻⁸`.

**Bias correction:** Since `m₀=v₀=0`, early estimates `m_t,v_t` are **biased toward zero**. Correct by dividing out the "amount of decay accumulated so far":

```
m̂_t = (m_t)/(1-β₁^t), v̂_t = (v_t)/(1-β₂^t), θ_(t+1)=θ_t-η(m̂_t)/(√(v̂_t)+ε)
```

**Algorithm: Adam**
```
Initialize m=0, v=0, t=0
while not converged:
    for each mini-batch:
        t ← t+1
        g = (1/B) Σ ∇θ L_i(θ)
        m ← β1 m + (1-β1) g
        v ← β2 v + (1-β2) g²
        m̂ = m / (1 - β1^t)
        v̂ = v / (1 - β2^t)
        θ ← θ - η m̂/(√v̂ + ε)
```

**Numerical trace (Iteration 0 in full detail)**, `η=0.05,β₁=0.9,β₂=0.999`, `t=1`:

> **`m₁=0.9(0)+0.1(8)=0.8, m₂=0.9(0)+0.1(16)=1.6`**

> **`v₁=0.999(0)+0.001(64)=0.064, v₂=0.999(0)+0.001(256)=0.256`**

```
m̂₁=(0.8)/(1-0.9)=8, m̂₂=(1.6)/(0.1)=16, v̂₁=(0.064)/(1-0.999)=64, v̂₂=(0.256)/(0.001)=256
```

```
w₁≤ftarrow4-0.05·(8)/(√(64))=4-0.05(1)=3.95, w₂≤ftarrow2-0.05·(16)/(√(256))=2-0.05(1)=1.95
```

| Iter | Loss | `∇` | `[m₁,m₂]` | `[v₁,v₂]` | `[m̂₁,m̂₂]` | `[v̂₁,v̂₂]` | `w₁` | `w₂` |
|---|---|---|---|---|---|---|---|---|
| 0 | 32.00 | [8,16] | [0.8,1.6] | [0.064,0.256] | [8,16] | [64,256] | 3.95 | 1.95 |
| 1 | 30.81 | [7.9,15.6] | [1.51,3.0] | [0.126,0.499] | [7.95,15.79] | [63.2,249.6] | 3.90 | 1.90 |
| 2 | 29.65 | [7.8,15.2] | [2.14,4.22] | [0.187,0.730] | [7.89,15.57] | [62.4,243.5] | 3.85 | 1.85 |
| 3 | 28.51 | [7.7,14.8] | [2.70,5.28] | [0.246,0.948] | [7.84,15.35] | [61.6,237.3] | 3.80 | 1.80 |

**Notice:** each update is nearly **constant** (`≈0.05` per iteration) — a hallmark of Adam: because `m̂/√(v̂)` is roughly the *sign* of the gradient scaled to unit-ish magnitude, bias-correction + normalization together produce very steady step sizes early in training, regardless of raw gradient magnitude.

**Iter-3 loss comparison (all with matched `η=0.05`):** Mini-batch GD `=4.21`, Momentum `=8.43`, RMSprop `=23.93`, Adam `=28.51`. *(On this particular toy problem, plain mini-batch GD actually converges fastest — a useful reminder that adaptive methods are not universally "better," they are more **robust across a wide range of problems with less tuning**, which matters far more on real, messy, high-dimensional losses than on this clean quadratic bowl.)*

### 6.4 RMSprop vs. Momentum — what's missing?

| Feature | Momentum | RMSprop |
|---|---|---|
| Accumulates velocity? | ✓ | ✗ |
| Smooths oscillations? | ✓ | ✗ |
| Per-parameter learning rates? | ✗ | ✓ |
| Adapts to gradient history? | ✗ | ✓ |

**Adam = Momentum's 1st moment + RMSprop's 2nd moment** — the best of both.

### 6.5 Why Adam is so popular
- Bias correction ensures unbiased moment estimates, especially early in training.
- Combines momentum (fast, smooth convergence) with per-parameter adaptive rates.
- Works well **out-of-the-box** with minimal tuning; robust across architectures and problem types (default choice for most modern deep learning).
- **Prefer SGD+Momentum instead when:** production systems where careful tuning is affordable, computer-vision tasks (sometimes yields better *final* generalization), or when best possible generalization (not fastest convergence) is the goal.

### 6.6 AdamW — Decoupled Weight Decay
**Problem with Adam + L2 regularization:** adding `(λ)/(2)‖θ‖²` to the loss makes the *gradient* include `λθ`, which then gets **divided by** `√(v̂)+ε` along with the "real" gradient — the adaptive learning rate ends up scaling the regularization term in unintended, parameter-dependent ways.

**AdamW fix:** apply weight decay **directly** to the parameters, decoupled from the adaptive gradient update:

> **`θ_(t+1) = θ_t - η((m̂_t)/(√(v̂_t)+ε) + λθ_t)`**

| Aspect | Adam (+L2) | AdamW |
|---|---|---|
| L2 term | Added to loss: `L+fracλ2‖θ‖²` | Applied directly to weights |
| Gradient | Includes reg. term: `∇ L+λθ` | Only from loss: `∇ L` |
| Update | `θ-η(m̂)/(√(v̂)+ε)` | `θ-η((m̂)/(√(v̂)+ε)+λθ)` |
| Weight decay | Depends on adaptive rates (inconsistent) | Constant, decoupled |
| Typical use | General optimization | Whenever L2/weight-decay regularization is desired |

**Practical advice:** prefer AdamW over "Adam + L2" whenever weight decay is used — this is now the default in most modern training recipes (e.g., Transformers).

---

## 7. Learning Rate Schedules

**Motivation:** early in training, large gradients justify a large LR for fast exploration; late in training, near the optimum, we want small, fine-grained steps. A single fixed `η` cannot satisfy both.

| Schedule | Formula | Notes / worked example |
|---|---|---|
| **Step decay** | `η_t=η₀·γ^(⌊ t/step⌋)` | `η₀=0.1,γ=0.1,step=30`: epoch 10 → `0.1×0.1⁰=0.1`; epoch 40 → `0.1×0.1¹=0.01`. Simple, interpretable — common in CV. |
| **Exponential decay** | `η_t=η₀ e^(-λ t)` | `η₀=0.1,λ=0.05`: epoch 10 → `0.1e^(-0.5)=0.061`; epoch 20 → `0.1e^(-1.0)=0.037`. Smooth, single hyperparameter. |
| **Cosine annealing** | `η_t=η_(min)+(1)/(2)(η_(max)-η_(min))(1+cos(π t/T))` | `η_(max)=0.1,η_(min)=0.001,T=100`: epoch 0 → `0.1`; epoch 50 → `0.001` (midpoint of the cosine reaches the minimum). Very smooth; no decay-rate tuning; popular in modern DL. |
| **Warmup (linear)** | `η_t=η₀· t/T_(warmup)` for `t≤ T_(warmup)` | `η₀=0.1,T_(warmup)=5`: epoch 2 → `0.1×2/5=0.04`; epoch 5 → `0.1` (warmup complete). Then apply decay schedule (e.g., cosine) for epochs 6–100. Important for large batch sizes/models where a large initial LR causes instability. |

---

## 8. Practical Optimization

### 8.1 Gradient Clipping (for exploding gradients)
**Norm-based clipping** (preserves direction, most commonly used):

> **`g ≤ftarrow g if ‖g‖≤τ ; τ(g)/(‖g‖) if ‖g‖>τ (τ≈1.0–5.0)`**

**Value-based clipping** (per-component, may change direction):

> **`g_i ≤ftarrow clip(g_i,-τ,τ) (τ≈0.5–1.0)`**

**When to use:** RNNs, Transformers, very deep networks, or any unstable training exhibiting NaN/Inf losses.

### 8.2 Batch size effects

| Aspect | Small batch (16–32) | Large batch (256–1024) |
|---|---|---|
| Gradient noise | High | Low |
| Updates/epoch | Many (`N/B`) | Few |
| Convergence | Noisy, explores well | Smooth, may get stuck |
| Generalization | Often better | May need LR scaling |
| Training time | Slower (more updates) | Faster (GPU efficient) |
| Memory | Low | High |
| Learning rate | Can use larger | May need warmup |

**Linear Scaling Rule:** when doubling batch size, double the learning rate to roughly maintain convergence speed. **Practical recommendation:** start with `B∈[64,128]`; increase if GPU memory allows and training is slow; use warmup when scaling batch size up.

### 8.3 Debugging training issues

| Symptom | Likely cause / fix |
|---|---|
| Loss not decreasing | LR too small, or bad initialization |
| Loss oscillating | LR too large — reduce by 10× |
| Loss exploding | Clip gradients / reduce LR |
| Loss plateau | Try momentum or Adam |

**General checklist:** plot train + validation loss curves; monitor gradient norms for NaN/Inf; verify data loading/preprocessing; sanity-check by trying to overfit a tiny batch (if the model can't even overfit 10 examples, something is broken).

### 8.4 Stopping criteria

| Criterion | Description | Typical value |
|---|---|---|
| Max epochs | Predetermined limit | 100–200 |
| Early stopping | Stop when validation loss stops improving | patience 5–10 epochs |
| Gradient threshold | Stop when `‖∇ L‖` below threshold | `<10⁻⁶` |
| Loss threshold | Stop at target loss value | problem-specific |

**Early stopping** (most common): track best validation loss; if no improvement for `patience` epochs, stop and **restore** the model checkpoint from the best epoch (prevents overfitting — see Module 4). Practical strategy: combine a max-epoch upper bound with early stopping to end training sooner if converged.

---

## 9. Comparison Summary & Guidance

| Method | Convergence | Memory | Tuning effort | Stability |
|---|---|---|---|---|
| SGD | Slow | Low | Hard | High |
| SGD + Momentum | Medium | Low | Medium | High |
| AdaGrad | Good | Medium | Easy | Medium |
| RMSprop | Good | Medium | Easy | Medium |
| Adam | Fast | Medium | Very easy | High |
| AdamW | Fast | Medium | Very easy | High |

**Best use cases:**
- **Adam/AdamW:** default choice, quick experiments, NLP, most problems.
- **SGD+Momentum:** production computer vision, when tuning budget allows, best final generalization.
- **RMSprop:** RNNs, online/non-stationary learning problems.

---

## 10. Code Implementation

### 10.1 From-scratch optimizers reproducing the toy problem (`L=w₁²+4w₂²`)

```python
# Topic: Optimizer comparison from scratch on toy quadratic bowl
# Purpose: Reproduce the numerical traces of GD, Momentum, RMSprop, Adam
#          exactly as derived in the module slides.
# Dependencies: numpy

import numpy as np

def grad(w):
    w1, w2 = w
    return np.array([2 * w1, 8 * w2])

def loss(w):
    w1, w2 = w
    return w1**2 + 4 * w2**2

# ---- Plain (mini-batch) Gradient Descent ----
def gd(eta=0.1, iters=4):
    w = np.array([4.0, 2.0])
    for i in range(iters):
        print(f"iter {i}: loss={loss(w):.2f}, w={w}")
        w = w - eta * grad(w)
    return w

# ---- SGD with Momentum ----
def momentum(eta=0.05, beta=0.9, iters=4):
    w = np.array([4.0, 2.0])
    v = np.zeros(2)
    for i in range(iters):
        print(f"iter {i}: loss={loss(w):.2f}, w={w}")
        g = grad(w)
        v = beta * v + g
        w = w - eta * v
    return w

# ---- RMSprop ----
def rmsprop(eta=0.05, gamma=0.9, eps=1e-8, iters=4):
    w = np.array([4.0, 2.0])
    s = np.zeros(2)
    for i in range(iters):
        print(f"iter {i}: loss={loss(w):.2f}, w={w}")
        g = grad(w)
        s = gamma * s + (1 - gamma) * g**2
        w = w - (eta / (np.sqrt(s) + eps)) * g
    return w

# ---- Adam (with bias correction) ----
def adam(eta=0.05, beta1=0.9, beta2=0.999, eps=1e-8, iters=4):
    w = np.array([4.0, 2.0])
    m = np.zeros(2); v = np.zeros(2); t = 0
    for i in range(iters):
        print(f"iter {i}: loss={loss(w):.2f}, w={w}")
        t += 1
        g = grad(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - eta * m_hat / (np.sqrt(v_hat) + eps)
    return w

print("=== Mini-batch GD ==="); gd()
print("=== Momentum ==="); momentum()
print("=== RMSprop ==="); rmsprop()
print("=== Adam ==="); adam()
# Expected: numbers match the hand-worked tables in §3.3, §5.2, §6.2, §6.3
```

### 10.2 AdamW (decoupled weight decay) from scratch
```python
# Topic: AdamW optimizer — decoupled weight decay
# Purpose: Show how weight decay is applied outside the adaptive-rate term
import numpy as np

def adamw_step(w, g, m, v, t, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    t += 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    w = w - eta * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)   # decoupled decay
    return w, m, v, t
```

### 10.3 Gradient clipping
```python
# Topic: Norm-based gradient clipping
import numpy as np

def clip_grad_norm(g, max_norm=5.0):
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g = g * (max_norm / norm)   # rescale, preserving direction
    return g
```

### 10.4 Learning-rate schedules
```python
import numpy as np

def step_decay(epoch, eta0=0.1, gamma=0.1, step_size=30):
    return eta0 * gamma ** (epoch // step_size)

def exponential_decay(epoch, eta0=0.1, lam=0.05):
    return eta0 * np.exp(-lam * epoch)

def cosine_annealing(epoch, eta_max=0.1, eta_min=0.001, T=100):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T))

def linear_warmup(epoch, eta0=0.1, T_warmup=5):
    return eta0 * min(1.0, epoch / T_warmup)
```

---

## 11. Connections to Other Topics
- **Module 2 (Feedforward Networks):** the gradients `∂ L/∂ W^[l]` derived via backpropagation are exactly the `g` consumed by every optimizer here.
- **Module 4 (Regularization):** AdamW's decoupled weight decay directly connects optimization to L2 regularization; gradient clipping is one of the standard remedies for exploding gradients (alongside better initialization / batch norm).
- **Module 5/6 (CNNs/RNNs):** gradient clipping and adaptive optimizers (especially Adam) are near-universal defaults for training these architectures, particularly RNNs where exploding gradients are common (Module 6).

---

## 12. Key Takeaways
1. Deep network loss surfaces are **non-convex**, riddled with saddle points and plateaus — increasingly so with more parameters/dimensions.
2. **Mini-batch SGD** is the practical default over pure batch or pure stochastic GD — it balances GPU efficiency and gradient-noise-driven exploration.
3. **Momentum** smooths oscillations by accumulating a velocity term but can amplify overshoot; **Nesterov** looks ahead for a course-correction.
4. **AdaGrad → RMSprop → Adam** is a natural progression: per-parameter adaptive rates, fix the ever-shrinking LR problem via EMA, then combine with momentum.
5. **Adam/AdamW** are the default modern choice for ease-of-use and robustness; **SGD+Momentum** can still yield better final generalization on vision tasks given tuning budget. Always pair careful **LR scheduling**, **gradient clipping**, and **early stopping** with whichever optimizer is chosen.

## 13. Common Mistakes & Misconceptions
- **Mistake:** "Adam always converges faster/better than SGD." **Correction:** on the toy quadratic problem here, plain GD actually reached a lower loss fastest by iter 3 — Adam's real advantage is **robustness with minimal tuning** on messy, high-dimensional, non-stationary real losses, not universal superiority on every problem.
- **Mistake:** using Adam with naive L2 regularization added to the loss. **Correction:** use **AdamW**, which decouples weight decay from the adaptive learning-rate scaling.
- **Mistake:** treating momentum's `η` as safely reusable from plain GD. **Correction:** momentum amplifies effective step sizes — typically requires a **smaller** `η` than vanilla GD to remain stable (seen numerically in §5.2, where oscillation persisted even at half the GD learning rate).
- **Mistake:** clipping gradients component-wise by default. **Correction:** prefer **norm-based** clipping, which preserves the gradient's direction; value-based clipping can distort it.

## 14. Practice Problems

1. **Compute one Adagrad step** for `g=[8,16]` starting `s=[0,0]`, `η=0.05`, `ε≈0`.
   *Solution:* `s≤ftarrow[0+64,0+256]=[64,256]`; update `=η g/√ s = 0.05×[8/8,16/16]=[0.05,0.05]`; new `w=[4-0.05,2-0.05]=[3.95,1.95]`.
2. **Given** `β=0.9`, previous velocity `v_(t-1)=[2.0,-1.0]`, new gradient `g_t=[1.0,3.0]`, compute `v_t` and the parameter update with `η=0.1`.
   *Solution:* `v_t = 0.9[2,-1]+[1,3]=[1.8+1,-0.9+3]=[2.8,2.1]`; update `=-η v_t=[-0.28,-0.21]`.
3. **Adam bias correction:** at `t=2`, `β₁=0.9`, raw `m₂=1.51`. Compute `m̂₂`.
   *Solution:* `m̂₂ = 1.51/(1-0.9²)=1.51/0.19=7.95` (matches table in §6.3).
4. **Cosine annealing:** compute `η₂₅` for `η_(max)=0.2,η_(min)=0.0,T=100`.
   *Solution:* `η₂₅=0+0.1(1+cos(π·25/100))=0.1(1+cos(45°))=0.1(1+0.707)=0.1707`.
5. **Gradient clipping:** given `g=[3,4]` (norm `=5`) and `τ=2`, compute the clipped gradient.
   *Solution:* since `‖g‖=5>τ=2`, `g≤ftarrow2·[3,4]/5=[1.2,1.6]`.
