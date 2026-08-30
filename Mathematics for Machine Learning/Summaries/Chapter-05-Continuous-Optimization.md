# Chapter 5 / Module 5 — Continuous Optimization

> **Course:** AMLSIZC416 — Mathematical Foundations for Machine Learning
> **Module 5:** Continuous Optimization
> **Reading mapped:** Textbook **T1** (Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*) **§7.1–7.3**
> **Going deeper (beyond the map):** learning‑rate intuition, momentum, exact line search, convexity as "why gradient descent even works," and constrained optimization via Lagrange multipliers.
> **Lab woven in:** **Lab 7 — Gradient descent in code.**

---

## 📖 How to read this chapter

> Every topic is taught in **three passes**:
> 1. **🧠 Plain English** — a mental picture, no symbols. Read this first.
> 2. **✍️ By hand** — the actual math, worked slowly with numbers.
> 3. **🐍 In Python / ML** — the same thing in code, with **real outputs** (every `# ->` line was executed by `_verify_ch5.py`, never guessed).
>
> New words are **bolded** and collected in the **Jargon Buster**. If a section feels heavy, read the 🧠 box and the diagram, then come back.
>
> **Prerequisite:** this chapter *uses* the gradient \(\nabla f\) and Hessian \(H\) from Chapter 4. If those feel shaky, skim Ch 4 §3 and §8 first.

---

## 🔤 Jargon Buster (read once, refer back forever)

| Word | In one plain sentence | Symbol |
|---|---|---|
| **Optimization** | Finding the input that makes a function as **small** (or large) as possible. | \(\min_x f(x)\) |
| **Objective / loss** | The number we're trying to minimize (in ML, "how wrong the model is"). | \(f\), \(L\) |
| **Minimum** | The lowest point — where we want to end up. | \(x^\*\) |
| **Gradient descent** | Repeatedly step **downhill** (opposite the gradient) to shrink \(f\). | \(x \leftarrow x-\eta\nabla f\) |
| **Learning rate / step size** | How big each downhill step is. Too small = slow; too big = overshoot/blow up. | \(\eta\) (eta) |
| **Convergence** | The steps settling down onto the minimum instead of wandering. | — |
| **Convex** | Bowl‑shaped: any line between two points stays **above** the curve — one global minimum, no traps. | — |
| **Momentum** | Give the ball "inertia" so it rolls through small bumps and speeds along. | \(\beta\) |
| **Line search** | Picking the *best* step size along the current downhill direction. | \(\eta^\*\) |
| **Constraint** | A rule the answer must obey (e.g., "must lie on this line"). | \(g(x)=0\) |
| **Lagrange multiplier** | A number that lets us fold a constraint into the objective and solve both at once. | \(\lambda\) (lambda) |
| **Stationary point** | Where the gradient is zero — a candidate minimum/maximum/saddle. | \(\nabla f=0\) |

---

## 🗺️ Mental map of this chapter

```
                    CONTINUOUS OPTIMIZATION
              "find the input that makes f smallest"
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
  HOW we descend        WILL it work?         WITH RULES?
       │                     │                     │
 GRADIENT DESCENT        CONVEXITY           CONSTRAINED opt
 x ← x − η·∇f          bowl vs. saddle      "obey g(x)=0"
       │                     │                     │
 ┌─────┴─────┐          convex ⇒ one          LAGRANGE
 │           │          global min,           MULTIPLIERS
 step size   momentum   GD is safe            ∇f = λ·∇g
 η (tuning)  (inertia)        │                     │
 │           │                └──────────┬──────────┘
 line search  roll through              │
 η* (optimal) small bumps               ▼
       └───────────┬───────────────  TRAIN THE MODEL
                   ▼                (minimize the loss L)
            Lab 7: code it up
```

**The big idea in one line:** *Learning = optimization.* Chapter 4 gave us the **direction** to move (the gradient). This chapter is about **how to actually move** — how big a step, how to speed up, how to know we'll arrive, and what to do when the answer must obey rules.

---

## Table of contents

1. [What "optimization" means in ML](#1)
2. [Gradient descent — the core loop (§7.1)](#2)
3. [The learning rate — the single most important knob](#3)
4. [Descent in many dimensions & momentum (§7.1)](#4)
5. [Convexity — why gradient descent works (§7.2/§7.3)](#5)
6. [Step size done right — exact line search](#6)
7. [Constrained optimization & Lagrange multipliers (§7.2)](#7)
8. [Putting it together — training linear regression by GD](#8)
9. [🧩 Problem‑Solving Workshop](#9)
10. [Synthesis, cheat‑sheet & mini‑glossary](#10)

---

<a name="1"></a>
## 1. What "optimization" means in ML

> 🧠 **Plain English.** Optimization is just "find the best setting." In ML the "setting" is the model's **weights**, and "best" means "smallest **loss**" (least wrong). Picture a landscape where height = loss and your map position = the weights. You're a hiker in fog who can only feel the **slope under your feet** (the gradient). You can't see the whole valley, but if you always step downhill, you'll reach a low point. That downhill‑stepping *is* training.

```
   loss f
     ▲   ●  start (bad weights)
     │  / \
     │ /   \_ step downhill…
     │/      \__
     │          \___ …again…
     │              \_____ …until we settle
     │                    ●  x*  (good weights, low loss)
     └───────────────────────────────► weight x
```

**Two questions this whole chapter answers:**
1. *Which way and how far do I step?* → gradient descent + learning rate + momentum (§2–4, §6).
2. *Will I actually reach the bottom, and what if there are rules?* → convexity + Lagrange multipliers (§5, §7).

---

<a name="2"></a>
## 2. Gradient descent — the core loop (§7.1)

> 🧠 **Plain English.** The gradient points **uphill** (steepest ascent). We want to go **down**, so we step in the *opposite* direction. Take a step, recompute the slope where you land, step again. Repeat. Each step multiplies the gradient by a small number \(\eta\) (the **learning rate**) so we don't leap too far. That's the entire algorithm — three symbols: \(x \leftarrow x-\eta\nabla f\).

```
   f(x)=x²
     ▲
   25┤●  x0=5, slope f'=2x=10 (steep, points RIGHT/up)
     │ \      step LEFT (downhill) by η·10
     │  \●  x1
     │   \●  x2
     │    \●___
     │        ●●●●●  → 0  (slope shrinks as we near the bottom)
     └────────────────────► x
        step = −η·f'(x): big when steep, tiny near the min
```

### (a) Concept — the update rule

\[
\boxed{\,x_{k+1} = x_k - \eta\,\nabla f(x_k)\,}
\]
- \(\nabla f(x_k)\): direction of steepest **ascent** at the current point.
- minus sign: turn it into steepest **descent**.
- \(\eta>0\): the step size. Notice the step **auto‑shrinks** near a minimum because \(\nabla f\to 0\) there.

### (b) ✍️ By hand — minimize \(f(x)=x^2\), start \(x_0=5\), \(\eta=0.1\)

The slope is \(f'(x)=2x\). Each step: \(x\leftarrow x-0.1(2x)=x-0.2x=0.8x\).
\[
5 \to 4 \to 3.2 \to 2.56 \to 2.048 \to 1.6384 \to \dots
\]
Every step multiplies by \(0.8\), so \(x_k=5\cdot(0.8)^k\to 0\). After 5 steps \(x=1.6384\); after 50 steps it's essentially 0.

### (c) 🐍 In Python

```python
x = 5.0
for k in range(5):
    g = 2*x            # gradient f'(x)=2x
    x = x - 0.1*g      # gradient-descent step
print(round(x,4))      # -> 1.6384   (matches hand calc)

x = 5.0
for k in range(50): x = x - 0.1*(2*x)
print(round(x,6))      # -> 7.1e-05  (essentially the minimum 0)
```

---

<a name="3"></a>
## 3. The learning rate — the single most important knob

> 🧠 **Plain English.** The learning rate \(\eta\) sets your step size, and it's a Goldilocks problem. **Too small** → you crawl and training takes forever. **Just right** → you glide to the bottom. **Too big** → you overshoot the valley and bounce around. **Way too big** → each step lands *higher* than the last and you fly off to infinity (diverge). Choosing \(\eta\) is the #1 practical skill in training models.

```
  too small (η=0.01)     just right (η=0.5)      too big (η=1.01)
     ●                        ●                    ●        ●
      ●                        \                   │\      /│
       ●                        ●  → min            │ \    / │
        ●   …crawls…           (one clean slide)    │  \  /  │  …bounces
         ●                                          ●   \/   ●   OUTWARD
   slow but safe            fast + stable          DIVERGES to ∞
```

### (a) ✍️ By hand — why \(\eta\ge 1\) explodes for \(f=x^2\)

The update is \(x\leftarrow(1-2\eta)x\). Convergence needs the multiplier \(|1-2\eta|<1\), i.e. \(0<\eta<1\).
- \(\eta=0.5\): multiplier \(=0\) → reaches the min in **one** step.
- \(\eta=1.01\): multiplier \(=1-2.02=-1.02\), \(|{-}1.02|>1\) → magnitude **grows** each step → diverges.

### (b) 🐍 In Python — four learning rates, 10 steps from \(x_0=5\)

```python
for eta in [0.01, 0.1, 0.5, 1.01]:
    x = 5.0
    for k in range(10): x = x - eta*(2*x)
    print(eta, round(x,4))
# -> 0.01  4.0854   (barely moved — too small)
# -> 0.1   0.5369   (steady progress)
# -> 0.5   0.0000   (perfect — one-shot for this problem)
# -> 1.01  6.0950   (grew past the start — DIVERGING)
```

> 🧠 **Takeaway.** There's a *stable range* of learning rates. Inside it, bigger = faster. Cross the ceiling and training blows up. Real practice: start small, increase until it's fast but still stable.

---

<a name="4"></a>
## 4. Descent in many dimensions & momentum (§7.1)

> 🧠 **Plain English.** Real losses have many weights, so the landscape is a high‑dimensional valley. Often the bowl is **stretched** — gentle in one direction, steep in another (an "ill‑conditioned" valley). Plain gradient descent then **zig‑zags**: it bounces across the steep, narrow direction while barely crawling along the gentle, long one. **Momentum** fixes this by giving the ball **inertia** — it accumulates a running "velocity," damping the zig‑zag and building speed along the consistent downhill direction. Like a heavy ball rolling versus a cautious hiker.

```
  Elongated bowl f = x² + 10y²  (steep in y, gentle in x)

   plain GD zig-zags:            momentum smooths + speeds:
        ╲                              ╲
      ╱  ╲  ╱╲                          ╲___
     ╱    ╲╱  ╲   slow along x            ╲__ rolls steadily
    •──────────► x                    •──────────► x
    bounces across y                  inertia cancels the bounce
```

### (a) Concept — the two updates

**Plain GD:** \(\ x_{k+1}=x_k-\eta\nabla f(x_k)\).
**GD + momentum:** keep a velocity \(v\) that remembers past steps:
\[
v_{k+1}=\beta v_k-\eta\nabla f(x_k),\qquad x_{k+1}=x_k+v_{k+1}.
\]
\(\beta\in[0,1)\) is how much past velocity you keep (typically 0.9). \(\beta=0\) is plain GD.

### (b) 🐍 In Python — same bowl, same 30 steps

```python
def gradf(v): return np.array([2*v[0], 20*v[1]])   # ∇(x²+10y²)

# plain GD
x = np.array([5.0,5.0]); eta = 0.05
for k in range(30): x = x - eta*gradf(x)
print(np.round(x,4))          # -> [0.212 0.   ]   still creeping in x

# GD + momentum
x = np.array([5.0,5.0]); v = np.zeros(2); beta = 0.9; eta = 0.02
for k in range(30):
    v = beta*v - eta*gradf(x); x = x + v
print(np.round(x,4))          # -> [0.9256 0.41  ]
```

> 🧠 **Honest read of the numbers.** The steep \(y\)‑direction is tamed fast by **both**. In the gentle \(x\)‑direction, plain GD barely moves (0.212 after 30 steps, still far from 0), while momentum has built speed and is sweeping \(x\) down — but with \(\beta=0.9\) it can also **overshoot/oscillate** before settling. That's the real lesson: momentum accelerates the slow directions but adds its own tuning (\(\beta\) and \(\eta\) together). This tuning tension is exactly why optimizers like **Adam** exist — they adapt the step per‑direction automatically.

---

<a name="5"></a>
## 5. Convexity — why gradient descent works (§7.2/§7.3)

> 🧠 **Plain English.** Gradient descent only *feels* the local slope — so how do we know it won't get stuck in a fake dip and miss the real bottom? The magic word is **convex**. A convex function is a single clean **bowl**: it has exactly **one** lowest point, no local traps, no saddles. On a convex loss, "always walk downhill" is *guaranteed* to reach the global minimum. Many classic ML losses (linear/ridge regression, logistic regression, SVM) are convex — that's why they train reliably. Deep nets are **non**‑convex (bumpy), which is why they need momentum, good initialization, and luck.

```
   CONVEX (bowl)                 NON-CONVEX (bumpy)
      \        /                   \    /\      /
       \      /                     \  /  \    /  ← local min TRAP
        \____/                       \/    \__/
   one global min,              many dips; GD can get
   GD always arrives ✔          stuck in the wrong one ✖
```

### (a) Concept — how to *test* convexity

- **1‑D:** \(f\) is convex if \(f''(x)\ge 0\) everywhere (curves upward).
- **Many‑D:** \(f\) is convex if its **Hessian** \(H\) is **positive semidefinite** everywhere — i.e., all eigenvalues \(\ge 0\). (Recall Ch 4 §8: the Hessian is the curvature matrix.)

Also equivalent (the picture definition): the straight line between any two points on the graph never dips **below** the curve.

### (b) ✍️ By hand + 🐍 checks

- \(f(x)=x^2\): \(f''=2>0\) → **convex**. ✔
- \(f(x,y)=x^2+10y^2\): \(H=\begin{bmatrix}2&0\\0&20\end{bmatrix}\), eigenvalues \(2,20>0\) → **convex** (an elongated bowl). ✔
- \(f(x,y)=x^2-y^2\): \(H=\begin{bmatrix}2&0\\0&-2\end{bmatrix}\), eigenvalues \(2,-2\) (mixed) → **not convex** — it's a **saddle**. ✖

```python
print(np.linalg.eigvalsh(np.array([[2,0],[0,20]],float)))  # -> [ 2. 20.]  all>0  => convex
print(np.linalg.eigvalsh(np.array([[2,0],[0,-2]],float)))  # -> [-2.  2.]  mixed  => NOT convex
```

> 🧠 **Why you care.** Before trusting gradient descent, ask "is my loss convex?" If yes, one run finds the true optimum. If no (deep learning), you accept a *good* local minimum and lean on the tricks in §4.

---

<a name="6"></a>
## 6. Step size done right — exact line search

> 🧠 **Plain English.** Instead of guessing \(\eta\), sometimes we can compute the **best possible** step along the current downhill direction — the one that lowers \(f\) the most before it would start rising again. That's **line search**: freeze the direction, and minimize \(f\) as a function of the single number \(\eta\). For a simple quadratic bowl there's a clean formula, and it can reach the bottom in **one** perfect step.

```
   pick direction −∇f, then slide along it:

   f along the ray
      \                     ● minimum of the 1-D slice
       \                   /   → that η is η* (best step)
        \_______________ /
         └──────────────────► η
              η*  = the sweet spot
```

### (a) ✍️ By hand — \(f(x)=x^2\) from \(x=5\)

Step in direction \(-f'(x)=-2x\). New point: \(x-\eta\,2x\). Minimize \((x-2\eta x)^2\) over \(\eta\): the inside hits 0 when \(2\eta=1\), i.e. \(\eta^\*=0.5\). One step: \(5-0.5(10)=0\) — straight to the minimum.

### (b) 🐍 In Python

```python
x = 5.0; g = 2*x; eta_star = 0.5
print(x - eta_star*g)   # -> 0.0   (reaches the min in ONE step)
```

> 🧠 **Reality check.** Exact line search is cheap only for special (quadratic) problems. In big ML we usually use a fixed or *scheduled* \(\eta\) (decaying over time) instead — but the intuition "there's an optimal step" is what schedules approximate.

---

<a name="7"></a>
## 7. Constrained optimization & Lagrange multipliers (§7.2)

> 🧠 **Plain English.** Sometimes the answer must obey a **rule** — "minimize cost, **but** the parts must add up to 1," or "find the closest point **on this line**." You can't just roll to the unconstrained bottom because it might break the rule. **Lagrange multipliers** are a trick: at the best allowed point, the objective's downhill direction points **straight along** the constraint's normal — there's no way to slide along the constraint and improve. We encode that with a new variable \(\lambda\) and solve a slightly bigger system. It turns "minimize with a rule" into plain calculus.

```
   contours of f (cost)          the KEY condition at the optimum:
   (   ((  •  ))   )   ∇f  ─────────────►  ∇f  is PARALLEL to ∇g
    ───────────────── constraint g(x)=0    (∇f = λ ∇g)
        ↑ best allowed point sits where a         → you can't move
          cost contour just KISSES the line          along g and do better
```

### (a) Concept — the recipe

To solve \(\min f(x)\) subject to \(g(x)=0\):
1. Build the **Lagrangian** \(\ \mathcal L(x,\lambda)=f(x)-\lambda\,g(x)\).
2. Set all partial derivatives to zero: \(\nabla_x\mathcal L=0\) (gives \(\nabla f=\lambda\nabla g\)) **and** \(\partial\mathcal L/\partial\lambda=0\) (this just re‑states the constraint \(g=0\)).
3. Solve the resulting equations for \(x\) and \(\lambda\).

### (b) ✍️ By hand — \(\min x^2+y^2\) subject to \(x+y=1\)

*(Geometrically: the point on the line \(x+y=1\) closest to the origin.)*

Lagrangian: \(\mathcal L = x^2+y^2-\lambda(x+y-1)\). Set derivatives to zero:
\[
\frac{\partial\mathcal L}{\partial x}=2x-\lambda=0,\qquad
\frac{\partial\mathcal L}{\partial y}=2y-\lambda=0,\qquad
\frac{\partial\mathcal L}{\partial\lambda}=-(x+y-1)=0.
\]
First two give \(x=\lambda/2\) and \(y=\lambda/2\), so \(x=y\). Plug into \(x+y=1\): \(2x=1\Rightarrow x=y=\tfrac12\), and \(\lambda=1\). Minimum value \(=\left(\tfrac12\right)^2+\left(\tfrac12\right)^2=0.5\).

### (c) 🐍 In Python — brute‑force check along the constraint

```python
best = None
for t in np.linspace(-1, 2, 300001):
    xx, yy = t, 1-t                 # every point satisfies x+y=1
    val = xx*xx + yy*yy
    if best is None or val < best[0]: best = (val, xx, yy)
print(np.round(best,4))   # -> [0.5 0.5 0.5]   (min value 0.5 at x=y=0.5)  ✓
```

> 🧠 **Where this shows up in ML.** SVMs (max‑margin classifiers) and PCA (best low‑dimensional directions) are *constrained* optimizations solved with exactly this Lagrange machinery — you'll meet both in Module 7.

---

<a name="8"></a>
## 8. Putting it together — training linear regression by GD

> 🧠 **Plain English.** Time to use everything at once. Linear regression has a **convex** loss (§5), so gradient descent (§2) with a sensible learning rate (§3) is *guaranteed* to reach the one true minimum — which we can also get instantly from the closed‑form **normal equation**. Watching GD's answer converge to the exact formula is the satisfying "it all works" moment.

### (a) Setup

Fit \(\hat b = w_0 + w_1\cdot(\text{feature})\) to
\(A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\ \mathbf b=(1,2,2)\). Loss \(L(w)=\lVert Aw-\mathbf b\rVert^2\), gradient \(\nabla L=2A^\top(Aw-\mathbf b)\) (from Ch 4 §5).

- **Closed form (normal equation):** \(w^\*=(A^\top A)^{-1}A^\top\mathbf b\).
- **Iterative (gradient descent):** \(w\leftarrow w-\eta\,\nabla L\), repeated.

### (b) 🐍 In Python — GD converges to the exact solution

```python
A = np.array([[1,1],[1,2],[1,3]], float); b = np.array([1,2,2.])

w = np.zeros(2); eta = 0.05
for k in range(2000):
    g = 2*A.T@(A@w - b); w = w - eta*g
print(np.round(w,4))                              # -> [0.6667 0.5   ]  (gradient descent)

w_closed = np.linalg.solve(A.T@A, A.T@b)
print(np.round(w_closed,4))                       # -> [0.6667 0.5   ]  (normal equation) ✓
```

### (c) The loss really does fall every step

```python
w = np.zeros(2)
for k in range(6):
    L = np.sum((A@w - b)**2); print(round(L,4))
    w = w - 0.05*(2*A.T@(A@w - b))
# -> 9.0, 4.09, 1.9235, 0.9665, 0.5429, 0.3545   (monotonically decreasing)
```

> 🧠 **The whole chapter in one experiment:** convex loss ⇒ safe to descend; good \(\eta\) ⇒ steady monotone drop; enough steps ⇒ GD's answer equals the exact optimum. That's model training.

---

<a name="9"></a>
## 9. 🧩 Problem‑Solving Workshop

> **Flow per scenario:** ① story → ② translate to math → ③ solve by hand → ④ ML lens → ⑤ verify in code. Build the reflex: *which way, how far, and will I arrive?*

### Scenario A — Pick a safe learning rate 🎚️

**① Story.** You're minimizing \(f(x)=x^2\) and training keeps exploding. What range of \(\eta\) is safe, and which is fastest?

**② Translate.** The update is \(x\leftarrow(1-2\eta)x\); it converges iff \(|1-2\eta|<1\).

**③ By hand.** \(|1-2\eta|<1 \Rightarrow 0<\eta<1\). Fastest is where the multiplier is 0: \(\eta=0.5\) (one‑shot). At \(\eta=1.01\) the multiplier is \(-1.02\) → diverges.

**④ ML lens.** Every real training run lives or dies by staying under this stability ceiling; schedules decay \(\eta\) to stay safe as the landscape steepens.

**⑤ Verify.**
```python
for eta in [0.5, 1.01]:
    x = 5.0
    for k in range(10): x = x - eta*(2*x)
    print(eta, round(x,4))     # -> 0.5 0.0   |   1.01 6.095 (diverging)
```

### Scenario B — Is my loss safe to descend? 🥣

**① Story.** Before training, you want to know whether gradient descent can get stuck.

**② Translate.** Check convexity via the Hessian's eigenvalues for \(f(x,y)=x^2+10y^2\) vs. the saddle \(x^2-y^2\).

**③ By hand.** Bowl: \(H=\mathrm{diag}(2,20)\), eigenvalues \(2,20>0\) → **convex**, one global min, GD is safe. Saddle: eigenvalues \(2,-2\) → **not convex**, GD can stall at the saddle.

**④ ML lens.** Linear/logistic regression are convex (train once, done); deep nets aren't (need momentum, restarts, good init).

**⑤ Verify.**
```python
print(np.linalg.eigvalsh(np.array([[2,0],[0,20]],float)))  # -> [ 2. 20.]  convex ✓
print(np.linalg.eigvalsh(np.array([[2,0],[0,-2]],float)))  # -> [-2.  2.]  saddle ✗
```

### Scenario C — Optimize under a rule (Lagrange) 🎯

**① Story.** Split a budget between two options, \(x+y=1\), while keeping \(x^2+y^2\) (a "spread" penalty) as small as possible. What's the best split?

**② Translate.** \(\min x^2+y^2\) s.t. \(x+y=1\). Lagrangian \(\mathcal L=x^2+y^2-\lambda(x+y-1)\).

**③ By hand.** \(2x=\lambda,\ 2y=\lambda \Rightarrow x=y\); with \(x+y=1\Rightarrow x=y=0.5\), \(\lambda=1\), min value \(0.5\). The "balanced" split wins — intuitive for a spread penalty.

**④ ML lens.** This is the seed of ridge‑regularization and max‑margin (SVM) thinking: optimize an objective while a constraint pins the solution.

**⑤ Verify.**
```python
best = None
for t in np.linspace(-1,2,300001):
    xx, yy = t, 1-t; val = xx*xx + yy*yy
    if best is None or val < best[0]: best = (val, xx, yy)
print(np.round(best,4))     # -> [0.5 0.5 0.5]   (min 0.5 at x=y=0.5) ✓
```

---

<a name="10"></a>
## 10. Synthesis, cheat‑sheet & mini‑glossary

### The one‑picture summary

```
   GOAL:  min f(x)   (make the loss small)
   ───────────────────────────────────────────────
   DIRECTION   −∇f            downhill (from Ch 4)
   STEP        x ← x − η∇f    η = learning rate
     too small → slow │ good → glide │ too big → diverge
   SPEED-UP    momentum: v ← βv − η∇f ; x ← x + v
   OPTIMAL η   line search  (η* minimizes f along −∇f)
   ───────────────────────────────────────────────
   WILL IT WORK?  convex (H ⪰ 0) ⇒ one global min ⇒ GD arrives
                  non-convex ⇒ local traps ⇒ need tricks
   WITH RULES?    Lagrange:  ∇f = λ∇g ,  g(x)=0
   ───────────────────────────────────────────────
   RESULT: GD on a convex loss = the exact optimum (normal eq.)
```

### Formula cheat‑sheet

| Idea | Formula | Verified value |
|---|---|---|
| Gradient descent | \(x_{k+1}=x_k-\eta\nabla f\) | \(x^2\): \(5\to0\) in ~50 steps |
| Stability (for \(x^2\)) | converges iff \(0<\eta<1\) | \(\eta{=}0.5\to0\); \(\eta{=}1.01\to\) diverges |
| Momentum | \(v\leftarrow\beta v-\eta\nabla f;\ x\leftarrow x+v\) | tames steep dir; needs tuning |
| Convex test | Hessian eigenvalues \(\ge 0\) | \([2,20]\) convex; \([-2,2]\) not |
| Exact line search | \(\eta^\*\) minimizes \(f(x-\eta\nabla f)\) | \(\eta^\*{=}0.5\), one step to 0 |
| Lagrangian | \(\mathcal L=f-\lambda g;\ \nabla f=\lambda\nabla g\) | \(x{=}y{=}0.5,\ \lambda{=}1\) |
| GD = closed form | \(w^\*=(A^\top A)^{-1}A^\top b\) | GD & normal eq: \([0.6667,0.5]\) |

### Mini‑glossary recap (say each in your own words)

- **Gradient descent** — repeatedly step opposite the gradient to shrink \(f\).
- **Learning rate \(\eta\)** — step size; Goldilocks knob (small=slow, big=diverge).
- **Momentum** — velocity/inertia that smooths zig‑zags and speeds slow directions.
- **Convex** — bowl‑shaped; one global minimum; GD is guaranteed to arrive.
- **Line search** — compute the best step size along the current direction.
- **Constraint** — a rule the solution must satisfy, e.g. \(g(x)=0\).
- **Lagrange multiplier \(\lambda\)** — folds a constraint into the objective so \(\nabla f=\lambda\nabla g\) at the optimum.

---

> ✅ **Reproduce all outputs.** Every `# ->` value above was produced by **`_verify_ch5.py`** (1‑D and multi‑D gradient descent, learning‑rate sweep, momentum, convexity via Hessian eigenvalues, exact line search, Lagrange brute‑force check, and GD‑vs‑normal‑equation for linear regression). Run it with `python _verify_ch5.py` — it prints each result and ends with **`ALL OK`**.

---

*Next up:* **Chapter 6 / Module 6 — Nonlinear Optimization** (Textbook **T2**, Aggarwal — *Linear Algebra and Optimization for Machine Learning* §4.4, 4.5, 5.2, 5.3): Newton's method & second‑order steps, gradient descent variants, and optimization for ML models beyond the convex quadratic.
