# Chapter 4 / Module 4 — Vector Calculus

> **Course:** AMLSIZC416 — Mathematical Foundations for Machine Learning
> **Module 4:** Vector Calculus
> **Reading mapped:** Textbook **T1** (Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*) **§5.1–5.8**
> **Going deeper (beyond the map):** backpropagation as *repeated chain rule* (the engine inside every neural network), computational graphs, and second‑order (curvature) information.
> **Lab woven in:** **Lab 6 — Taylor series & the Hessian.**

---

## 📖 How to read this chapter

> Every topic is taught in **three passes** so it sticks:
> 1. **🧠 Plain English** — a one‑paragraph mental picture, no symbols. Read this first.
> 2. **✍️ By hand** — the actual math, worked slowly, numbers included.
> 3. **🐍 In Python / ML** — the same thing in code, with **real outputs** (every `# ->` line was executed by `_verify_ch4.py`, never guessed).
>
> New words are **bolded** and collected in the **Jargon Buster** below. If a section feels heavy, just read the 🧠 box and the diagram, then come back.

---

## 🔤 Jargon Buster (read this once, refer back forever)

| Word | In one plain sentence | Symbol |
|---|---|---|
| **Derivative** | How fast a function changes as you nudge its input — the *slope*. | \(f'(x)\), \(\frac{df}{dx}\) |
| **Slope / rate of change** | "If I step right by a tiny bit, how much do I go up?" | — |
| **Partial derivative** | The slope in **one** input direction, holding the others frozen. | \(\frac{\partial f}{\partial x}\) |
| **Gradient** | A **vector** of all the partial derivatives — points in the steepest‑uphill direction. | \(\nabla f\) |
| **Jacobian** | The gradient idea for a function that outputs a **vector** — a matrix of all slopes. | \(J\) |
| **Hessian** | The matrix of **second** derivatives — measures *curvature* (how bowl‑shaped). | \(H\) |
| **Chain rule** | To differentiate a function‑inside‑a‑function, **multiply** the slopes layer by layer. | — |
| **Backpropagation** | The chain rule applied backwards through a network to get every weight's slope. | — |
| **Taylor series** | Approximate a curvy function near a point using its derivatives (a local "stunt double"). | — |
| **Loss function** | A number that says "how wrong is my model?" — we want it small. | \(L\) |
| **Gradient descent** | Take a step **downhill** (opposite the gradient) to reduce the loss. | \(x \leftarrow x - \eta\nabla f\) |
| **Learning rate** | The size of that downhill step. | \(\eta\) |
| **Stationary point** | Where the gradient is zero — a flat spot (min, max, or saddle). | \(\nabla f = 0\) |

---

## 🗺️ Mental map of this chapter

```
                          VECTOR CALCULUS
                       "how does the output change
                        when I nudge the input?"
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   1 INPUT                  MANY INPUTS              MANY IN → MANY OUT
   1 OUTPUT                 1 OUTPUT                 (vector-valued)
        │                        │                        │
   derivative  f'(x)        gradient  ∇f            Jacobian  J
   (a number)               (a vector)              (a matrix)
        │                        │                        │
        └───────────┬────────────┴───────────┬────────────┘
                    │                         │
             CHAIN RULE                 SECOND DERIVATIVES
        (compose functions →         gradient of the gradient
         multiply the slopes)               → HESSIAN H
                    │                    (curvature / convexity)
                    ▼                         │
            BACKPROPAGATION                   ▼
        (chain rule, backwards,        TAYLOR SERIES
         through every layer)      (local polynomial approx;
                    │               Lab 6, linearization)
                    ▼
             TRAIN THE MODEL
        gradient descent: step downhill
           x ← x − η·∇f   until loss is small
```

**The big idea in one line:** *Calculus is how a model learns.* Training = "measure how wrong you are (**loss**), find which direction reduces the wrongness (**gradient**), and step that way (**gradient descent**)." Everything below builds the tools to compute that direction.

---

## Table of contents

1. [Why calculus is the heart of ML](#1)
2. [The derivative — slope of one‑input functions (§5.1)](#2)
3. [Partial derivatives & the gradient (§5.2)](#3)
4. [The Jacobian — gradients of vector‑valued functions (§5.3)](#4)
5. [Gradients of quadratic forms & least squares (§5.4–5.5)](#5)
6. [The chain rule — differentiating compositions](#6)
7. [Backpropagation — the chain rule, backwards (§5.6)](#7)
8. [The Hessian — second derivatives & curvature (§5.7)](#8)
9. [Taylor series & linearization (§5.8, Lab 6)](#9)
10. [🧩 Problem‑Solving Workshop](#10)
11. [Synthesis, cheat‑sheet & mini‑glossary](#11)

---

<a name="1"></a>
## 1. Why calculus is the heart of ML

> 🧠 **Plain English.** A machine‑learning model has **knobs** (called *weights* or *parameters*). At the start the knobs are set randomly, so the model is bad. We measure "how bad" with a single number — the **loss**. Now the whole game is: *which way should I turn each knob to make the loss smaller?* Calculus answers exactly that question. The **derivative** tells you the direction and steepness of "downhill" in the loss landscape. Turn the knobs a little in the downhill direction, repeat a few thousand times, and the model gets good. That loop is called **training**.

```
   loss (how wrong)
     ▲
     │● start (random knobs, high loss)
     │ \
     │  \   each arrow = one gradient-descent step
     │   ●
     │    \
     │     ●___
     │         ●___   ← we slide downhill
     │             ●________●  ✔ trained (low loss)
     └───────────────────────────────►  knob value (a weight)
```

**Where each tool plugs in:**

| ML task | Calculus tool |
|---|---|
| "Which way lowers the loss?" | **gradient** \(\nabla L\) |
| "How big a step is safe?" | **Hessian** (curvature) & **Taylor** |
| "How do I get slopes through 20 layers?" | **chain rule → backpropagation** |
| "Is this flat spot a real minimum?" | **gradient = 0** + **Hessian** positive |

---

<a name="2"></a>
## 2. The derivative — slope of one‑input functions (§5.1)

> 🧠 **Plain English.** The **derivative** of \(f\) at a point is the **slope** of the curve there: *if I nudge \(x\) to the right by a tiny amount, how much does \(f\) go up (or down)?* Steep uphill → big positive number. Steep downhill → big negative. Flat → zero.

```
   f(x)
     │            ___----  ← steep here: derivative BIG
     │        _--
     │     _-        ____
     │   _-      _---     ---___    ← flat top: derivative ≈ 0
     │ _-    _--                --_
     │-  _--                       --_  ← going down: derivative NEGATIVE
     └──────────────────────────────────► x
```

### (a) Concept — the limit definition

The derivative is the slope of the line through two points on the curve, as those points squeeze together:

\[
f'(x) \;=\; \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
\]

Read it as **rise over run** when the run \(h\) becomes infinitesimally small.

### (b) ✍️ By hand — \(f(x)=x^3\)

Use the rule \(\frac{d}{dx}x^n = n\,x^{n-1}\):

\[
f'(x) = 3x^{2}, \qquad f'(2) = 3\cdot 2^{2} = 3\cdot 4 = \boxed{12}.
\]

Sanity‑check with the limit for a tiny \(h=0.001\):
\[
\frac{(2.001)^3-(2)^3}{0.001}=\frac{8.012006-8}{0.001}=12.006\ldots\approx 12.\ ✓
\]

**Handy derivative table (memorize these five):**

| \(f(x)\) | \(f'(x)\) |
|---|---|
| \(x^n\) | \(n x^{n-1}\) |
| \(e^{x}\) | \(e^{x}\) |
| \(\ln x\) | \(1/x\) |
| \(\sin x\) | \(\cos x\) |
| \(\cos x\) | \(-\sin x\) |

### (c) 🐍 In Python — finite differences confirm the hand answer

We approximate the slope numerically (a **central difference**) and compare to the exact \(3x^2\):

```python
import numpy as np
def _s(v): return np.asarray(v,float).ravel()[0]
def grad(f, x, h=1e-6):                 # numeric slope, central difference
    x=np.asarray(x,float); g=np.zeros_like(x)
    for i in range(x.size):
        e=np.zeros_like(x); e[i]=h
        g[i]=(_s(f(x+e))-_s(f(x-e)))/(2*h)
    return g

f = lambda x: x**3
print(round(grad(f,[2.])[0],4))   # -> 12.0     (matches exact 3*2^2 = 12)
```

> 🧠 **Why finite differences?** They're our *lie detector*. We derive a slope by hand, then let the computer nudge the input and measure the real change. If the two numbers match, our calculus is right. We'll use this trick to verify **every** gradient in this chapter.

---

<a name="3"></a>
## 3. Partial derivatives & the gradient (§5.2)

> 🧠 **Plain English.** Most ML functions have **many** inputs (a loss depends on millions of weights). A **partial derivative** asks a simpler question: *freeze every input except one, then take the ordinary slope in that one direction.* Do that for every input and stack the answers into a vector — that stack is the **gradient**. The gradient is an arrow that points in the direction where the function climbs **fastest**. To go **down** (what we want in training), step in the **opposite** direction.

```
  Think of a hill (f = height) over a 2-D map (x, y):

     y
     ▲        ∇f points straight UPHILL (steepest ascent)
     │            ↗
     │          ↗   contour lines = equal height
     │   (   ( ( •→ ∇f )  )   )
     │          the gradient is ⟂ to the contour it sits on
     └──────────────────────────────► x

  Gradient descent walks the OPPOSITE way:  −∇f  (steepest downhill).
```

### (a) Concept

For \(f(x,y)\), the two partials and the gradient are:
\[
\frac{\partial f}{\partial x}\ \text{(vary }x\text{, freeze }y),\quad
\frac{\partial f}{\partial y}\ \text{(vary }y\text{, freeze }x),\qquad
\nabla f=\begin{bmatrix}\partial f/\partial x\\[2pt]\partial f/\partial y\end{bmatrix}.
\]

### (b) ✍️ By hand — \(f(x,y)=x^2+3xy+y^2\)

Differentiate treating the *other* variable as a constant:
\[
\frac{\partial f}{\partial x}=2x+3y,\qquad
\frac{\partial f}{\partial y}=3x+2y .
\]
At the point \((x,y)=(1,2)\):
\[
\nabla f(1,2)=\begin{bmatrix}2(1)+3(2)\\ 3(1)+2(2)\end{bmatrix}
=\begin{bmatrix}8\\ 7\end{bmatrix}.
\]
So near \((1,2)\), the function rises fastest heading in direction \([8,7]\); to *decrease* it, head in \([-8,-7]\).

### (c) 🐍 In Python

```python
f = lambda v: v[0]**2 + 3*v[0]*v[1] + v[1]**2
print(np.round(grad(f,[1,2]),4))    # -> [8. 7.]   (matches hand-derived [8, 7])
```

---

<a name="4"></a>
## 4. The Jacobian — gradients of vector‑valued functions (§5.3)

> 🧠 **Plain English.** So far the output was a single number. But sometimes a function eats a vector **and spits out a vector** (e.g., a layer that turns 3 inputs into 2 outputs). Now "the slope" is a whole **table**: for **each output**, how does it change with **each input**? That table is the **Jacobian** — one row per output, one column per input. It's just "all the gradients, stacked."

```
   f: (x, y)  ─────►  ( f1 , f2 )     2 inputs → 2 outputs

              ∂f1/∂x   ∂f1/∂y        row 1 = gradient of output f1
      J  =  [                 ]
              ∂f2/∂x   ∂f2/∂y        row 2 = gradient of output f2
                ▲        ▲
             column =  how ALL outputs react to one input
```

### (a) Concept

For \(\mathbf f(\mathbf x)=\big(f_1(\mathbf x),\dots,f_m(\mathbf x)\big)\) with \(\mathbf x\in\mathbb R^n\), the Jacobian is the \(m\times n\) matrix
\[
J_{ij}=\frac{\partial f_i}{\partial x_j}.
\]
(A gradient is just a Jacobian with a **single row** — the case \(m=1\).)

### (b) ✍️ By hand — \(\mathbf f(x,y)=\big(x^2y,\; x+y\big)\)

Row 1 (\(f_1=x^2y\)): \(\ \partial f_1/\partial x = 2xy,\quad \partial f_1/\partial y = x^2.\)
Row 2 (\(f_2=x+y\)): \(\ \partial f_2/\partial x = 1,\quad \partial f_2/\partial y = 1.\)

\[
J(x,y)=\begin{bmatrix}2xy & x^2\\ 1 & 1\end{bmatrix}
\quad\Longrightarrow\quad
J(1,2)=\begin{bmatrix}2(1)(2) & 1^2\\ 1 & 1\end{bmatrix}
=\begin{bmatrix}4 & 1\\ 1 & 1\end{bmatrix}.
\]

### (c) 🐍 In Python

```python
def jacobian(f, x, h=1e-6):
    x=np.asarray(x,float); f0=np.asarray(f(x),float); J=np.zeros((f0.size,x.size))
    for i in range(x.size):
        e=np.zeros_like(x); e[i]=h
        J[:,i]=(np.asarray(f(x+e))-np.asarray(f(x-e)))/(2*h)
    return J

F = lambda v: np.array([v[0]**2*v[1], v[0]+v[1]])
print(np.round(jacobian(F,[1,2]),4))
# -> [[4. 1.]
#     [1. 1.]]     (matches hand-derived [[4,1],[1,1]])
```

---

<a name="5"></a>
## 5. Gradients of quadratic forms & least squares (§5.4–5.5)

> 🧠 **Plain English.** Two shapes appear *everywhere* in ML: the **quadratic form** \(\mathbf x^\top A\,\mathbf x\) (a bowl) and the **least‑squares error** \(\lVert A\mathbf x-\mathbf b\rVert^2\) (how badly a linear model fits data). If you memorize their gradient formulas, you can differentiate huge chunks of ML **without** grinding through partials each time. Think of these as "calculus shortcuts."

### (a) Two identities worth memorizing

| Function | Gradient | Why you care |
|---|---|---|
| \(\ f(\mathbf x)=\mathbf x^\top A\,\mathbf x\ \) (A symmetric) | \(\ \nabla f = 2A\mathbf x\ \) | curvature bowls, regularizers |
| \(\ f(\mathbf x)=\lVert A\mathbf x-\mathbf b\rVert^2\ \) | \(\ \nabla f = 2A^\top(A\mathbf x-\mathbf b)\ \) | **linear regression loss** |

> 🧠 **Analogy.** In 1‑D, \(\frac{d}{dx}(a x^2)=2ax\). The matrix versions are the *exact same pattern* — the "2" and the "\(x\)" survive, and \(A\) plays the role of \(a\).

### (b) ✍️ By hand

**Quadratic form.** \(A=\begin{bmatrix}2&1\\1&3\end{bmatrix}\) (symmetric), at \(\mathbf x=(1,1)\):
\[
\nabla f = 2A\mathbf x = 2\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}
=2\begin{bmatrix}3\\4\end{bmatrix}=\begin{bmatrix}6\\8\end{bmatrix}.
\]

**Least squares.** Take
\(A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\ \mathbf b=\begin{bmatrix}1\\2\\2\end{bmatrix}\), evaluated at \(\mathbf x=(0,0)\).
Since \(A\mathbf x-\mathbf b=-\mathbf b\), the gradient is
\[
\nabla f = 2A^\top(A\mathbf x-\mathbf b) = -2A^\top\mathbf b
= -2\begin{bmatrix}1&1&1\\1&2&3\end{bmatrix}\begin{bmatrix}1\\2\\2\end{bmatrix}
= -2\begin{bmatrix}5\\11\end{bmatrix}=\begin{bmatrix}-10\\-22\end{bmatrix}.
\]
This negative gradient \([10,22]\) is *exactly the first downhill step* linear‑regression training would take from the origin. 🎯

### (c) 🐍 In Python

```python
A = np.array([[2,1],[1,3]], float)
f = lambda v: v @ A @ v
print(np.round(grad(f,[1,1]),4))          # -> [6. 8.]   (= 2 A x)

Am = np.array([[1,1],[1,2],[1,3]], float); b = np.array([1,2,2.])
f = lambda x: np.sum((Am@x - b)**2)
print(np.round(grad(f,[0,0]),4))          # -> [-10. -22.]   (= -2 Aᵀ b)
```

---

<a name="6"></a>
## 6. The chain rule — differentiating compositions

> 🧠 **Plain English.** Real models are functions **inside** functions inside functions (a "pipeline"). The **chain rule** says: to get the slope of the whole pipeline, find the slope of **each stage** and **multiply** them together. Like gears meshing — if gear A spins gear B twice as fast, and B spins C three times as fast, then A spins C **2×3 = 6** times as fast.

```
   x ──►[ inner g ]──► u ──►[ outer f ]──► y

   dy/dx  =  dy/du   ×   du/dx
             (outer)     (inner)
              slope       slope       ← MULTIPLY the local slopes
```

### (a) Concept

If \(y=f(u)\) and \(u=g(x)\), then
\[
\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}=f'(g(x))\cdot g'(x).
\]

### (b) ✍️ By hand — \(h(x)=\sin(x^2+1)\)

Split into stages: inner \(u=g(x)=x^2+1\) (so \(g'(x)=2x\)); outer \(f(u)=\sin u\) (so \(f'(u)=\cos u\)). Multiply:
\[
h'(x)=\cos(x^2+1)\cdot 2x.
\]
At \(x=1\): \(\ h'(1)=\cos(2)\cdot 2 = 2\cos 2 \approx 2(-0.41615)=\boxed{-0.8323}.\)

### (c) 🐍 In Python

```python
h = lambda x: np.sin(x[0]**2 + 1)
print(round(grad(h,[1.])[0],4))     # -> -0.8323
print(round(2*np.cos(2),4))         # -> -0.8323   (exact 2·cos 2)
```

---

<a name="7"></a>
## 7. Backpropagation — the chain rule, backwards (§5.6)

> 🧠 **Plain English.** A neural network is a long chain of simple steps. To train it we need the slope of the **loss** with respect to **every weight**. Computing each one separately would repeat a ton of work. **Backpropagation** is the clever bookkeeping: do **one forward pass** to compute the output and loss, then **walk backwards** multiplying local slopes (chain rule) — reusing shared pieces — so you get *all* the weight‑slopes in one sweep. It's not new calculus; it's the chain rule organized efficiently.

### (a) One neuron, end to end

We use the smallest possible network: **one neuron** with a **sigmoid** activation and a **squared‑error** loss.

```
  FORWARD  ───────────────────────────────────────►
     x, w, b
        │   z = w·x + b     (linear)
        ▼
        z ──► a = σ(z)      (sigmoid squashes to (0,1))
              │
              ▼
              L = (a − y)²   (how wrong vs. the target y)

  BACKWARD ◄───────────────────────────────────────
     dL/da = 2(a − y)                 ← slope of loss w.r.t. output
     da/dz = a(1 − a)                 ← slope of sigmoid
     dL/dz = dL/da · da/dz            ← chain rule (MULTIPLY)
     then split dz to each input:
        dL/dw = dL/dz · x   (since ∂z/∂w = x)
        dL/db = dL/dz · 1   (since ∂z/∂b = 1)
        dL/dx = dL/dz · w   (since ∂z/∂x = w)
```

Two facts we lean on: \(\sigma(z)=\dfrac{1}{1+e^{-z}}\) and its neat derivative \(\sigma'(z)=\sigma(z)\,(1-\sigma(z))=a(1-a)\).

### (b) ✍️ By hand — \(x=1,\ w=0.5,\ b=0,\ y=0\)

**Forward pass:**
\[
z = w x + b = 0.5(1)+0 = 0.5,\qquad
a=\sigma(0.5)=\frac{1}{1+e^{-0.5}} = 0.62246,\qquad
L=(a-y)^2 = 0.62246^2 = 0.38746.
\]

**Backward pass:**
\[
\frac{dL}{da}=2(a-y)=2(0.62246)=1.24492,\qquad
\frac{da}{dz}=a(1-a)=0.62246\,(0.37754)=0.23500,
\]
\[
\frac{dL}{dz}=\frac{dL}{da}\cdot\frac{da}{dz}=1.24492\times0.23500=0.29256.
\]
Now hand \(dL/dz\) to each input:
\[
\frac{dL}{dw}=\frac{dL}{dz}\cdot x = 0.29256,\qquad
\frac{dL}{db}=\frac{dL}{dz}\cdot 1 = 0.29256,\qquad
\frac{dL}{dx}=\frac{dL}{dz}\cdot w = 0.29256\times0.5 = 0.14628.
\]
So to reduce the loss we'd nudge \(w\) and \(b\) **down** by a step proportional to \(0.29256\). That's one training step. 🎉

### (c) 🐍 In Python — and a finite‑difference cross‑check

```python
def sigmoid(z): return 1/(1+np.exp(-z))
x, w, b, y = 1.0, 0.5, 0.0, 0.0

z = w*x + b; a = sigmoid(z); L = (a - y)**2
print("forward: z=%.4f a=%.5f L=%.5f" % (z, a, L))
# -> forward: z=0.5000 a=0.62246 L=0.38746

dL_da = 2*(a - y)
da_dz = a*(1 - a)
dL_dz = dL_da*da_dz
dL_dw = dL_dz*x; dL_db = dL_dz*1; dL_dx = dL_dz*w
print("backward: dL/dw=%.5f dL/db=%.5f dL/dx=%.5f" % (dL_dw, dL_db, dL_dx))
# -> backward: dL/dw=0.29256 dL/db=0.29256 dL/dx=0.14628

# Cross-check dL/dw and dL/db numerically (nudge the parameters):
Lf = lambda p: (sigmoid(p[0]*x + p[1]) - y)**2      # p = (w, b)
print(np.round(grad(Lf,[w,b]),5))    # -> [0.29256 0.29256]  ✓ matches by-hand
```

> 🧠 **Why this is the whole game.** A deep network is just *this neuron pattern repeated thousands of times*. Backprop chains these local `dL/d(·)` multiplications from the loss back to every weight. Libraries (PyTorch, TensorFlow) do it automatically (**autodiff**) — but it is *exactly* the chain rule you just did by hand.

---

<a name="8"></a>
## 8. The Hessian — second derivatives & curvature (§5.7)

> 🧠 **Plain English.** The gradient tells you the **slope** (which way is downhill). The **Hessian** tells you the **curvature** — is the surface a *bowl* (curving up, so there's a bottom to fall into), a *dome* (curving down), or a *saddle* (up one way, down another)? Curvature answers "*how big a step is safe?*" and "*is this flat spot actually a minimum?*" It's the matrix of all **second** partial derivatives.

```
  bowl (min)        dome (max)          saddle
   \     /            _     _         \         /
    \   /            / \   / \         \  ___  /
     \_/            /   \_/   \         \/   \/
  H "positive"    H "negative"     H mixed signs
  → true minimum  → true maximum   → not a min or max
```

### (a) Concept

For \(f(x,y)\), the Hessian collects the four second partials:
\[
H=\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x\,\partial y}\\[8pt]
\dfrac{\partial^2 f}{\partial y\,\partial x} & \dfrac{\partial^2 f}{\partial y^2}
\end{bmatrix}.
\]
For nice (smooth) functions the mixed partials are equal, so \(H\) is **symmetric**.

**The second‑derivative test at a stationary point** (\(\nabla f = 0\)):
- \(H\) **positive definite** (all eigenvalues \(>0\)) → **minimum** (bowl). ✔ what training wants.
- \(H\) **negative definite** → **maximum**.
- mixed‑sign eigenvalues → **saddle point**.

### (b) ✍️ By hand — \(f(x,y)=x^2y\)

First partials: \(\ \partial f/\partial x = 2xy,\quad \partial f/\partial y = x^2.\)
Now differentiate again:
\[
\frac{\partial^2 f}{\partial x^2}=2y,\qquad
\frac{\partial^2 f}{\partial x\partial y}=2x,\qquad
\frac{\partial^2 f}{\partial y\partial x}=2x,\qquad
\frac{\partial^2 f}{\partial y^2}=0.
\]
So
\[
H(x,y)=\begin{bmatrix}2y & 2x\\ 2x & 0\end{bmatrix}
\quad\Longrightarrow\quad
H(1,2)=\begin{bmatrix}4 & 2\\ 2 & 0\end{bmatrix}.
\]
(Notice \(H\) is symmetric, as promised.)

### (c) 🐍 In Python

```python
def hessian(f, x, h=1e-4):
    x=np.asarray(x,float); n=x.size; H=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ei=np.zeros(n); ei[i]=h; ej=np.zeros(n); ej[j]=h
            H[i,j]=(_s(f(x+ei+ej))-_s(f(x+ei-ej))
                    -_s(f(x-ei+ej))+_s(f(x-ei-ej)))/(4*h*h)
    return H

f = lambda v: v[0]**2 * v[1]
print(np.round(hessian(f,[1,2]),3))
# -> [[4. 2.]
#     [2. 0.]]     (matches hand-derived [[4,2],[2,0]])
```

---

<a name="9"></a>
## 9. Taylor series & linearization (§5.8, Lab 6)

> 🧠 **Plain English.** A curvy function is hard to work with. Near any single point, though, we can replace it with a simple polynomial "stunt double" that behaves almost identically **nearby** — this is a **Taylor series**. Keep just the first slope term and you get a straight‑line approximation (**linearization**); add the curvature term (Hessian) and you get a parabola that hugs the curve even better. This is how optimizers *predict* what a step will do before taking it.

```
   f(x)
     │        real curve  ~~~~~
     │      ~~       ______ tangent line (1st-order Taylor)
     │    ~~   __---   good ONLY near the point a
     │  ~~ _--          ..... parabola (2nd-order) hugs longer
     │ ~-•───────────────────────► x
        a  (expansion point)
```

### (a) Concept

Around a point \(a\), a one‑variable function expands as
\[
f(a+\delta)\approx f(a) + f'(a)\,\delta + \tfrac12 f''(a)\,\delta^2 + \cdots
\]
The **multivariate** version (vector \(\mathbf x\), step \(\boldsymbol\delta\)) uses the gradient and Hessian:
\[
f(\mathbf a+\boldsymbol\delta)\approx f(\mathbf a) + \nabla f(\mathbf a)^\top\boldsymbol\delta + \tfrac12\,\boldsymbol\delta^\top H(\mathbf a)\,\boldsymbol\delta.
\]
**Key fact (Lab 6):** for a **quadratic** function the 2nd‑order Taylor expansion is not an approximation — it is **exact**.

### (b) ✍️ By hand

**One variable — approximate \(e^{0.1}\).** Expand \(e^x\) around \(a=0\) (where \(f=f'=f''=1\)) to 2nd order:
\[
e^{0.1}\approx 1 + 0.1 + \tfrac12(0.1)^2 = 1 + 0.1 + 0.005 = 1.105.
\]
True value \(e^{0.1}=1.105171\dots\), so the error is only \(\approx 0.00017\). Two terms already nail 4 decimals. ✓

**Multivariate exactness.** Take the pure quadratic \(f(x,y)=x^2+3xy+y^2\), expand around \(\mathbf a=(0,0)\) (where \(f(\mathbf a)=0\) and \(\nabla f(\mathbf a)=0\)) with \(H=\begin{bmatrix}2&3\\3&2\end{bmatrix}\) and step \(\boldsymbol\delta=(1,2)\):
\[
f(\boldsymbol\delta)\approx \tfrac12\boldsymbol\delta^\top H\boldsymbol\delta
=\tfrac12\begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}2&3\\3&2\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}
=\tfrac12\begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}8\\7\end{bmatrix}
=\tfrac12(8+14)=11.
\]
And the true \(f(1,2)=1+6+4=11\). **Exact**, as promised. 🎯

### (c) 🐍 In Python

```python
x = 0.1
approx = 1 + x + x**2/2
print(approx, round(np.exp(0.1),6), round(abs(np.exp(0.1)-approx),6))
# -> 1.105 1.105171 0.000171

f = lambda v: v[0]**2 + 3*v[0]*v[1] + v[1]**2
H = np.array([[2,3],[3,2]], float); d = np.array([1,2.])
print(0.5*d@H@d, f([1,2]))      # -> 11.0 11   (2nd-order Taylor is exact for a quadratic)
```

---

<a name="10"></a>
## 10. 🧩 Problem‑Solving Workshop

> **How each scenario flows:** ① a short story → ② translate it to math → ③ solve by hand → ④ the ML lens → ⑤ verify in code. Build the *reflex*: "what changes, and which way should I step?"

### Scenario A — One gradient‑descent step on linear regression 🏠

**① Story.** You're fitting a line \( \hat b = x_0 + x_1\cdot(\text{feature})\) to three data points. Your weights start at \(\mathbf x=(0,0)\) (a blank model). What's the very first learning step?

**② Translate.** Data \(A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}\), targets \(\mathbf b=(1,2,2)\), loss \(L=\lVert A\mathbf x-\mathbf b\rVert^2\). One step of gradient descent is \(\ \mathbf x \leftarrow \mathbf x - \eta\,\nabla L\).

**③ By hand.** From §5 the gradient at the origin is \(\nabla L=-2A^\top\mathbf b=[-10,-22]\). With a small learning rate \(\eta=0.01\):
\[
\mathbf x_{\text{new}} = (0,0) - 0.01\,[-10,-22] = (0.10,\ 0.22).
\]
The weights moved **up** because the gradient pointed down — we always step *against* the gradient.

**④ ML lens.** This is literally what `model.fit()` does under the hood, thousands of times, each step shrinking the loss.

**⑤ Verify.**
```python
Am = np.array([[1,1],[1,2],[1,3]], float); b = np.array([1,2,2.])
L  = lambda x: np.sum((Am@x - b)**2)
g  = grad(L,[0,0]); print(np.round(g,4))          # -> [-10. -22.]
x_new = np.array([0,0.]) - 0.01*g; print(x_new)   # -> [0.1  0.22]
print(L([0,0]), ">", L(x_new))                    # loss went DOWN ✓
```

### Scenario B — Is this flat spot really a minimum? 🥣

**① Story.** An optimizer stopped where the gradient is zero. Did it find a real bottom, or is it stuck on a saddle?

**② Translate.** Test the loss \(f(x,y)=x^2+y^2\) at the stationary point \((0,0)\): check \(\nabla f=0\), then read the **Hessian**.

**③ By hand.** \(\nabla f=(2x,2y)=(0,0)\) ✓ at the origin. Hessian \(H=\begin{bmatrix}2&0\\0&2\end{bmatrix}\); eigenvalues \(2,2>0\) → **positive definite** → genuine **minimum** (a bowl). If instead one eigenvalue were negative, it'd be a **saddle** and training should keep moving.

**④ ML lens.** Curvature (Hessian eigenvalues) distinguishes "we're done" from "we're stuck on a saddle" — a real issue in deep nets.

**⑤ Verify.**
```python
f = lambda v: v[0]**2 + v[1]**2
print(np.round(grad(f,[0,0]),6))            # -> [0. 0.]   (stationary)
H = hessian(f,[0,0]); print(np.round(H,3))  # -> [[2. 0.] [0. 2.]]
print(np.round(np.linalg.eigvalsh(H),3))    # -> [2. 2.]   all > 0 => minimum ✓
```

### Scenario C — Predicting a step with a Taylor model 🔮

**① Story.** Before committing to a step \(\boldsymbol\delta\), you want to *predict* how much the loss will drop — cheaply, without re‑evaluating the whole model.

**② Translate.** Use the 2nd‑order Taylor model \(f(\mathbf a+\boldsymbol\delta)\approx f(\mathbf a)+\nabla f^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta\) for \(f(x,y)=x^2+3xy+y^2\) at \(\mathbf a=(1,2)\).

**③ By hand.** \(f(1,2)=11\); \(\nabla f(1,2)=[8,7]\) (from §3); \(H=\begin{bmatrix}2&3\\3&2\end{bmatrix}\). For a step \(\boldsymbol\delta=(0.1,-0.1)\):
\[
\Delta f \approx \nabla f^\top\boldsymbol\delta + \tfrac12\boldsymbol\delta^\top H\boldsymbol\delta
= (8(0.1)+7(-0.1)) + \tfrac12\big[\ldots\big] = 0.1 + \tfrac12(0.01\cdot2 + 2(0.1)(-0.1)3 + 0.01\cdot2).
\]
The quadratic part \(=\tfrac12(0.02 -0.06 +0.02)=\tfrac12(-0.02)=-0.01\); so \(\Delta f\approx 0.1-0.01=0.09\), predicting \(f\approx 11.09\).

**④ ML lens.** Second‑order optimizers (Newton's method) use exactly this model to choose smart steps instead of tiny blind ones.

**⑤ Verify.**
```python
f = lambda v: v[0]**2 + 3*v[0]*v[1] + v[1]**2
a = np.array([1,2.]); d = np.array([0.1,-0.1])
g = grad(f,a); H = hessian(f,a)
pred = f(a) + g@d + 0.5*d@H@d
print(round(pred,4), round(f(a+d),4))   # -> 11.09 11.09   (Taylor model nails it; exact for quadratics)
```

---

<a name="11"></a>
## 11. Synthesis, cheat‑sheet & mini‑glossary

### The one‑picture summary

```
   INPUT nudged  ──►  how does OUTPUT react?
   ────────────────────────────────────────────
   1→1   derivative  f'(x)          a number   (slope)
   n→1   gradient    ∇f             a vector    (steepest uphill)
   n→m   Jacobian    J              a matrix    (all slopes)
   n→1   Hessian     H              a matrix    (curvature)
   ────────────────────────────────────────────
   compose functions  →  CHAIN RULE  (multiply local slopes)
   many layers        →  BACKPROP    (chain rule, backwards, reuse work)
   approximate locally→  TAYLOR      (slope + curvature = local model)
   train the model    →  x ← x − η·∇L  (step downhill, repeat)
```

### Formula cheat‑sheet

| Idea | Formula | Verified value |
|---|---|---|
| Power rule | \(\frac{d}{dx}x^n=nx^{n-1}\) | \(f'(2)=12\) for \(x^3\) |
| Gradient | \(\nabla f=[\partial f/\partial x_i]\) | \(\nabla(x^2{+}3xy{+}y^2)|_{(1,2)}=[8,7]\) |
| Quadratic form | \(\nabla(\mathbf x^\top A\mathbf x)=2A\mathbf x\) | \([6,8]\) |
| Least squares | \(\nabla\lVert A\mathbf x-\mathbf b\rVert^2=2A^\top(A\mathbf x-\mathbf b)\) | \([-10,-22]\) |
| Jacobian | \(J_{ij}=\partial f_i/\partial x_j\) | \([[4,1],[1,1]]\) |
| Chain rule | \((f\circ g)'=f'(g)\,g'\) | \(h'(1)=2\cos2=-0.8323\) |
| Backprop | \(\frac{dL}{dw}=\frac{dL}{dz}\frac{dz}{dw}\) | \(dL/dw=0.29256\) |
| Hessian | \(H_{ij}=\partial^2 f/\partial x_i\partial x_j\) | \([[4,2],[2,0]]\) |
| Taylor (2nd) | \(f(\mathbf a{+}\boldsymbol\delta)\approx f{+}\nabla f^\top\boldsymbol\delta{+}\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta\) | \(e^{0.1}\approx1.105\); quadratic exact = 11 |
| Gradient descent | \(\mathbf x\leftarrow\mathbf x-\eta\,\nabla f\) | step \((0,0)\to(0.1,0.22)\) |

### Mini‑glossary recap (say each in your own words)

- **Derivative** — slope of a 1‑input function.
- **Gradient** — vector of partials; points steepest‑uphill; step *against* it to learn.
- **Jacobian** — matrix of slopes for a vector‑in/vector‑out function.
- **Hessian** — matrix of second derivatives; tells bowl vs. saddle (curvature).
- **Chain rule** — multiply the slopes of nested functions.
- **Backpropagation** — chain rule run backwards through a network to get every weight's slope, reusing work.
- **Taylor series** — local polynomial stand‑in built from derivatives; linearization keeps just the slope term.
- **Gradient descent** — repeatedly step downhill by \(\eta\nabla f\) to shrink the loss.

---

> ✅ **Reproduce all outputs.** Every `# ->` value above was produced by **`_verify_ch4.py`** (finite‑difference `grad` / `jacobian` / `hessian` helpers cross‑check each hand‑derived result). Run it with `python _verify_ch4.py` — it prints each numeric‑vs‑exact comparison and ends with **`ALL OK`**.

---

*Next up:* **Chapter 5 / Module 5 — Continuous Optimization** (T1 §7.1–7.3): gradient descent with momentum, step‑size/learning‑rate choices, constrained optimization & Lagrange multipliers, convexity — plus **Lab 7 (gradient descent in code)**.
