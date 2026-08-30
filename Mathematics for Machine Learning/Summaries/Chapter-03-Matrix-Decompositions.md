# Chapter 3 — Matrix Decompositions (Beginner-Friendly, In Depth)

**Course:** AMLSIZC416 — Mathematical Foundations for Machine Learning
**Module 3:** *Determinant & trace, eigenvalues & eigenvectors, Cholesky, eigendecomposition & diagonalization, SVD, matrix approximation.*
**Reading (mapped):** **T1** §4.1–4.6.
**Going deeper (this doc adds):** the **power method** for finding a dominant eigenvector, and the **Eckart–Young** low-rank story — the math behind PCA, recommender systems, and compression. **Labs woven in:** **Lab 3** (matrix powers via eigendecomposition), **Lab 4** (power method), **Lab 5** (SVD & low-rank approximation).

> ### 📖 How to read this document (beginner note)
> Layers per topic: **🧠 Plain English** (idea + picture) → **(a) Concept & intuition** → **(b) Worked-by-hand math** → **(c) Python / ML**. Keep the **Jargon Buster** open. All Python outputs are real (verified with NumPy 2.4 — run `_verify_ch3.py`).
>
> **Where we are:** Chapter 1 = *solve* systems. Chapter 2 = *geometry* (spaces, length, angle, projection). Chapter 3 = **"take a matrix apart"** — factor it into simpler pieces (like factoring `12 = 2·2·3`) that expose *what the matrix really does*. This is the single most useful chapter for ML: **PCA, SVD, recommender systems, and Gaussian sampling all live here.**

---

## 🔤 Jargon Buster (keep this open while reading)

| Term | Plain meaning |
|---|---|
| **decomposition / factorization** | rewriting a matrix as a product of simpler matrices (e.g. `A = PDP⁻¹`). Like factoring a number. |
| **determinant `det(A)`** | one number telling how much `A` scales area/volume; `det=0` ⇒ `A` squashes space flat (no inverse). |
| **trace `tr(A)`** | the sum of the diagonal entries; also equals the **sum of eigenvalues**. |
| **eigenvector `v`** | a special direction that `A` only **stretches**, never rotates: `Av = λv`. |
| **eigenvalue `λ`** | the stretch factor along its eigenvector (`λ=2` doubles, `λ=−1` flips, `λ=0` collapses). |
| **characteristic polynomial** | the equation `det(A − λI) = 0` whose roots are the eigenvalues. |
| **diagonal matrix** | zeros everywhere except the main diagonal — the simplest kind of matrix. |
| **diagonalizable** | can be written `A = PDP⁻¹` with `D` diagonal (has enough independent eigenvectors). |
| **eigendecomposition** | `A = PDP⁻¹`: eigenvectors in `P`, eigenvalues in `D`. |
| **symmetric matrix** | `A = Aᵀ`; always has real eigenvalues and **orthogonal** eigenvectors. |
| **SPD** | symmetric positive-definite: symmetric with all eigenvalues `> 0` (covariance/kernel matrices). |
| **Cholesky `A = LLᵀ`** | factor an SPD matrix into a lower-triangular `L` times its transpose — the "square root" of a matrix. |
| **singular values `σ`** | the stretch factors of *any* matrix (even non-square); `σ = √(eigenvalues of AᵀA)`, always `≥ 0`. |
| **SVD `A = UΣVᵀ`** | every matrix = rotate (`Vᵀ`) → stretch (`Σ`) → rotate (`U`). The universal decomposition. |
| **rank-`k` approximation** | the best possible `A` you can build using only `k` "layers" — keep the top `k` singular values. |
| **orthogonal matrix `Q`** | `QᵀQ = I`; a pure rotation/reflection (from Ch. 2). |

> 🧠 **One-line mantra:** *A matrix decomposition reveals the hidden "stretch directions" of a transformation. Find those, and hard problems (powers, inverses, compression, PCA) become easy.*

---

## 🗺️ Mental map of this chapter

```
                     A MATRIX  A  (a transformation)
                                │
         ┌───────── SUMMARIES (one number) ─────────┐
         ▼                                           ▼
   determinant det(A)                          trace tr(A)
   = area/volume scaling                       = sum of diagonal
   det=0 ⇒ singular, collapses                 = sum of eigenvalues
                                │
         ┌────────── THE CORE: special directions ──────────┐
         ▼                                                   ▼
   eigenvectors v & eigenvalues λ  (Av = λv)         singular values σ
   "directions only stretched, not turned"           "stretch factors of ANY matrix"
                    │                                          │
                    ▼                                          ▼
   EIGENDECOMPOSITION  A = P D P⁻¹              SVD  A = U Σ Vᵀ
   (square, enough eigenvectors)                (ALWAYS works, any shape)
        │                    │                            │
   fast powers Aᵏ        Cholesky A=LLᵀ         LOW-RANK APPROX (keep top-k σ)
   (Lab 3)               (SPD "sqrt", sampling) │      = PCA, compression,
   Markov chains         Gaussian, GP           │        recommenders (Lab 5)
        │                                        │
   power method (Lab 4) ── finds the biggest λ ──┘
   PageRank, dominant mode
```

Each section is one box. Keep glancing back.

---

## Table of Contents
1. The big idea — why factor a matrix?
2. Determinant (§4.1)
3. Trace (§4.1)
4. Eigenvalues & eigenvectors (§4.2)
5. Eigendecomposition, diagonalization & matrix powers (§4.4, Lab 3)
6. The power method — finding the dominant eigenvector (Lab 4)
7. Cholesky decomposition (§4.3)
8. Singular Value Decomposition — SVD (§4.5)
9. Low-rank matrix approximation (§4.6, Lab 5)
10. 🧩 Problem-Solving Workshop (data & ML scenarios)
11. Synthesis + cheat-sheet

---

## 1. The big idea — why factor a matrix?

🧠 **Plain English:** With numbers, factoring `12 = 2 × 2 × 3` instantly tells you things (it's even, divisible by 3, not prime). **Matrix decomposition does the same for transformations:** it rewrites a confusing matrix as a product of *simple, meaningful* pieces — usually **rotations** and **pure stretches**. Once you see the stretches, everything else (inverting, raising to powers, compressing, finding the "main directions" of data) becomes easy.

```
   HARD to read directly            EASY once factored
   ┌ 2  1 ┐                          rotate      stretch      rotate back
   │ 1  2 │        =        (change basis) → (×3 and ×1) → (change basis back)
   └      ┘                          this is what A "really does"
```

Three factorizations to master, in order of generality:
- **Eigendecomposition** `A = PDP⁻¹` — for square matrices with enough eigenvectors.
- **Cholesky** `A = LLᵀ` — for symmetric positive-definite matrices (a fast "square root").
- **SVD** `A = UΣVᵀ` — for **any** matrix, any shape. The crown jewel.

---

## 2. Determinant (§4.1)

### 🧠 Plain English
The determinant is **one number that measures how much a matrix stretches or shrinks space**. In 2-D it's an **area** scaling factor; in 3-D a **volume** factor. If `det = 0`, the matrix **flattens** space into a lower dimension (a plane becomes a line, a line becomes a point) — and flattening can't be undone, so the matrix has **no inverse**.

```
   Unit square (area 1)      After A=[[2,0],[0,3]]        After a det=0 matrix
   ┌──┐                      ┌──────┐                      unit square → a line
   │  │  area 1              │      │  area 6 = det        ───────────  area 0
   └──┘                      │      │  (stretched 2×3)     (space collapsed!)
                             └──────┘
```

### (a) Concept & intuition
For a square `A`, `det(A)` is a single scalar. Key facts:
- `|det(A)|` = factor by which `A` scales area/volume. **Sign** tells orientation (negative = a flip/reflection).
- `det(A) = 0` ⟺ columns are **linearly dependent** ⟺ `A` is **singular** (no inverse) ⟺ `Ax=0` has non-trivial solutions.
- `det(AB) = det(A)det(B)`, `det(Aᵀ) = det(A)`, `det(A⁻¹) = 1/det(A)`.
- For a **triangular** (or diagonal) matrix, `det` = **product of the diagonal** — a huge shortcut.

### (b) Worked-by-hand math

**2×2 (the base case):** `det[[a,b],[c,d]] = ad − bc`.
`det[[1,2],[3,4]] = 1·4 − 2·3 = 4 − 6 = −2.` (Nonzero ⇒ invertible; the `−` sign ⇒ it also flips orientation.)

**3×3 by Laplace (cofactor) expansion along the top row:** expand, alternating `+ − +`:
```
A = [1 2 3]
    [4 5 6]
    [7 8 10]

det = 1·det[5 6]  − 2·det[4 6]  + 3·det[4 5]
          [8 10]        [7 10]        [7 8]
    = 1·(5·10 − 6·8) − 2·(4·10 − 6·7) + 3·(4·8 − 5·7)
    = 1·(50 − 48)    − 2·(40 − 42)    + 3·(32 − 35)
    = 2 − 2·(−2) + 3·(−3) = 2 + 4 − 9 = −3.
```

**Triangular shortcut:** `det[[2,7,1],[0,3,5],[0,0,4]] = 2·3·4 = 24` (ignore everything off the diagonal).

### (c) Python / ML
```python
print(round(np.linalg.det(np.array([[1,2],[3,4]], float))))         # -> -2
A3 = np.array([[1,2,3],[4,5,6],[7,8,10]], float)
print(round(np.linalg.det(A3)))                                     # -> -3
print(round(np.linalg.det(np.array([[2,7,1],[0,3,5],[0,0,4]],float))))  # -> 24
```
> **ML lens.** (1) `det=0` is the exact signal that a feature matrix is **redundant/singular** (Ch. 1 multicollinearity). (2) In probability, when you change variables in a density you multiply by `|det(Jacobian)|` — this is how **normalizing flows** and the change-of-variables formula work. (3) The **log-determinant** of a covariance matrix appears in the Gaussian log-likelihood. *(In code we rarely compute `det` directly for large matrices — it under/overflows; we use `slogdet` or decompositions instead.)*

---

## 3. Trace (§4.1)

### 🧠 Plain English
The **trace** is the easiest matrix summary: just **add up the diagonal**. Its superpower: it secretly equals the **sum of the eigenvalues** (the total "stretch") — so you can sanity-check eigenvalues without computing eigenvectors.

### (a) Concept + (b) worked math
`tr(A) = Σ aᵢᵢ`. Example: `tr[[1,2],[3,4]] = 1 + 4 = 5`.
Useful properties:
- `tr(A) = Σ λᵢ` (sum of eigenvalues) and `det(A) = Π λᵢ` (product) — two quick checks.
- **Cyclic:** `tr(ABC) = tr(BCA) = tr(CAB)` (rotate the product; do **not** arbitrarily reorder).
- `tr(A+B) = tr(A) + tr(B)`, `tr(Aᵀ) = tr(A)`.

### (c) Python / ML
```python
B = np.array([[4,1],[2,3]], float); wB,_ = np.linalg.eig(B)
print(np.trace(B), round(wB.sum()))     # 7.0 7   (trace = sum of eigenvalues 2+5)
print(round(np.linalg.det(B)), round(wB.prod()))  # 10 10 (det = product 2*5)
```
> **ML lens.** The **trace of the covariance matrix = total variance** of your data (PCA reports "variance explained" as ratios of these). Trace also shows up in the "trace trick" for expectations (`E[xᵀAx] = tr(A·Cov)`), and in regularizers like the **nuclear norm** (sum of singular values) used for low-rank learning.

---

## 4. Eigenvalues & eigenvectors (§4.2)

### 🧠 Plain English
Most vectors, when hit by a matrix `A`, get **turned AND stretched**. But a few special directions get **only stretched, not turned** — they come out pointing the same way (or exactly backwards). Those magic directions are **eigenvectors**, and how much they stretch is the **eigenvalue** `λ`. They are the matrix's "grain," like the grain of wood.

```
   generic vector: rotates          eigenvector: stays on its line
        ↑ x            ↗ Ax               →  v        →→  Av = λv  (here λ=2, just longer)
        │            ↗                     ●──────►    ●───────────►
        └───►                              same direction, only scaled
```

### (a) Concept & intuition
An **eigenvector** `v ≠ 0` and its **eigenvalue** `λ` satisfy
```
A v = λ v          ("A acting on v = just scaling v")
```
- `λ > 1` stretches, `0<λ<1` shrinks, `λ < 0` flips to the opposite side, `λ = 0` collapses that direction (⇒ `A` singular).
- Eigenvectors give the **natural axes** of the transformation. Along them, a complicated matrix acts like simple multiplication.
- **Symmetric matrices** (`A=Aᵀ`) are especially nice: real eigenvalues, and eigenvectors are mutually **orthogonal** (perpendicular) — this is why PCA axes are perpendicular.

### (b) Worked-by-hand math — the full recipe
**Find eigenvalues:** solve the **characteristic equation** `det(A − λI) = 0`.
**Find eigenvectors:** for each `λ`, solve `(A − λI)v = 0`.

**Example 1 (symmetric):** `A = [[2,1],[1,2]]`.
```
det(A − λI) = det[2−λ   1  ] = (2−λ)² − 1 = λ² − 4λ + 3 = (λ−1)(λ−3) = 0
                 [ 1   2−λ ]
⇒ λ = 3 and λ = 1
```
- For `λ=3`: `(A−3I)v = [[−1,1],[1,−1]]v = 0 ⇒ −v₁+v₂=0 ⇒ v = (1, 1)`.
- For `λ=1`: `(A−1I)v = [[1,1],[1,1]]v = 0 ⇒ v₁+v₂=0 ⇒ v = (1, −1)`.
Notice `(1,1) ⟂ (1,−1)` — **orthogonal**, because `A` is symmetric. ✅

**Example 2 (non-symmetric):** `B = [[4,1],[2,3]]`.
```
det(B − λI) = (4−λ)(3−λ) − 1·2 = λ² − 7λ + 10 = (λ−2)(λ−5) = 0  ⇒ λ = 2, 5
```
- `λ=5`: `[[−1,1],[2,−2]]v=0 ⇒ v=(1,1)`.
- `λ=2`: `[[2,1],[2,1]]v=0 ⇒ 2v₁+v₂=0 ⇒ v=(1,−2)`.
Quick check: `λ₁+λ₂ = 7 = tr(B)` ✓ and `λ₁·λ₂ = 10 = det(B)` ✓.

### (c) Python / ML
```python
A = np.array([[2,1],[1,2]], float)
w, v = np.linalg.eig(A)
print(np.round(w))          # -> [3. 1.]  (eigenvalues)
print(np.round(v,4))        # columns are unit eigenvectors ±(0.7071,0.7071), ±(−0.7071,0.7071)
```
*(NumPy returns **unit-length** eigenvectors and may flip a sign — direction is what matters, not sign/scale.)*

> **ML lens — this is the seed of PCA.** The **principal components** of data are the eigenvectors of its **covariance matrix**; the eigenvalues tell you **how much variance** lies along each. **PageRank** is the dominant eigenvector of the web's link matrix. **Spectral clustering** uses eigenvectors of a graph matrix. Eigenvalues also tell you **stability** (of dynamical systems, RNNs: `|λ|>1` explodes, `|λ|<1` vanishes — the vanishing/exploding gradient story).

---

## 5. Eigendecomposition, diagonalization & matrix powers (§4.4, Lab 3)

### 🧠 Plain English
If a matrix has enough eigenvectors, we can rewrite it as **"change to eigen-axes → stretch → change back."** In eigen-coordinates the matrix is just a **diagonal** (a list of stretches) — the simplest object in linear algebra. This makes brutal computations (like `A¹⁰⁰`) trivial.

```
     A = P D P⁻¹
          │  │  └ change back to normal axes
          │  └── stretch by eigenvalues (diagonal, easy!)
          └───── change into eigenvector axes
```

### (a) Concept & intuition
Stack the eigenvectors as **columns** of `P`, and the eigenvalues on the diagonal of `D`. Then
```
A = P D P⁻¹        (eigendecomposition / diagonalization)
```
This works when `A` has `n` **linearly independent** eigenvectors (then it's **diagonalizable**). For **symmetric** `A`, `P` can be chosen **orthogonal** (`P⁻¹ = Pᵀ`) → `A = QΛQᵀ` (the **spectral theorem**).

**Why it's powerful — matrix powers (Lab 3):**
```
Aᵏ = (PDP⁻¹)(PDP⁻¹)…(PDP⁻¹) = P Dᵏ P⁻¹      (the P⁻¹P pairs cancel!)
```
and `Dᵏ` is just each diagonal entry raised to `k` — no repeated matrix multiplication.

### (b) Worked-by-hand math
Take `A = [[2,1],[1,2]]` (eigenpairs from §4: `λ=3→(1,1)`, `λ=1→(1,−1)`).
```
P = [1  1]     D = [3 0]      P⁻¹ = 1/2 [ 1  1]
    [1 −1]         [0 1]                [ 1 −1]     (since det P = −2)
```
Check `A = PDP⁻¹`:
```
PD = [3  1]        (PD)P⁻¹ = 1/2 [3+1  3−1] = 1/2 [4 2] = [2 1] = A ✓
     [3 −1]                      [3−1  3+1]        [2 4]   [1 2]
```
Now `A¹⁰ = P·diag(3¹⁰, 1¹⁰)·P⁻¹`. Since `3¹⁰ = 59049`, this is one multiply instead of ten. The `[0,0]` entry works out to `(59049+1)/2 = 29525`.

### (c) Python / ML
```python
P = np.array([[1,1],[1,-1]], float); D = np.diag([3.,1.])
print(P@D@np.linalg.inv(P))          # -> [[2,1],[1,2]]  reconstructs A
A10 = P@np.diag([3.**10, 1.])@np.linalg.inv(P)
print(round(A10[0,0]))               # -> 29525
print(np.allclose(A10, np.linalg.matrix_power(np.array([[2,1],[1,2]]),10)))  # -> True
```
> **ML lens.** (1) **Markov chains / random walks:** the state after `k` steps is `P^k x₀`; eigen-decomposition gives the long-run behaviour instantly (the **stationary distribution** is the `λ=1` eigenvector). (2) **Graph diffusion & PageRank** iterate a matrix — powers again. (3) **Whitening/decorrelation** uses `A^{-1/2} = P D^{-1/2} P⁻¹`, trivial once diagonalized.

---

## 6. The power method — finding the dominant eigenvector (Lab 4)

### 🧠 Plain English
For huge matrices you can't solve `det(A−λI)=0` by hand. Trick: **pick any vector and keep multiplying by `A`, renormalizing each time.** The vector automatically swings toward the eigenvector with the **biggest** eigenvalue — because that direction gets amplified most each step. That's the **power method**, and it's literally how early **PageRank** was computed.

```
  start random →  A· →  A· →  A· → …  ⟶  locks onto the dominant eigenvector
       ↘           ↘      ↘                 (biggest-λ direction wins)
        (small-λ parts shrink away relative to the biggest one)
```

### (a) Concept + (b) worked idea
Any start vector `x = c₁v₁ + c₂v₂ + …` (mix of eigenvectors). Applying `A` `k` times:
`Aᵏx = c₁λ₁ᵏv₁ + c₂λ₂ᵏv₂ + …`. If `|λ₁|` is the largest, the `λ₁ᵏ` term **dominates**, so `Aᵏx` aligns with `v₁`. We renormalize each step to avoid overflow. The eigenvalue estimate is the **Rayleigh quotient** `λ ≈ xᵀAx` (for a unit `x`).

### (c) Python / ML (Lab 4, verified)
```python
A = np.array([[2,1],[1,2]], float)
x = np.array([1., 0.])
for _ in range(20):
    x = A @ x
    x = x / np.linalg.norm(x)        # renormalize so it can't blow up
print(np.round(x, 4))                # -> [0.7071 0.7071]  = dominant eigenvector (1,1)/√2
print(round(x @ A @ x, 4))           # -> 3.0  = dominant eigenvalue λ=3
```
> **ML lens.** The power method (and its cousins) computes **PageRank**, the **top principal component** without forming the whole covariance, and the largest singular value for SVD. Its cousin, checking `|λ_max|`, diagnoses **exploding/vanishing** behaviour in recurrent nets.

---

## 7. Cholesky decomposition (§4.3)

### 🧠 Plain English
For a special, very common kind of matrix — **symmetric positive-definite (SPD)**, like every covariance matrix — there's a "**square root**": a lower-triangular `L` with `A = LLᵀ`. Think of `L` as `√A`. It's cheap to compute and makes solving systems and **sampling random data** fast.

```
   A (SPD)     =        L        ·        Lᵀ
   ┌ 4  2 ┐          ┌ 2  0 ┐          ┌ 2  1 ┐
   │ 2  5 │    =     │ 1  2 │    ·     │ 0  2 │
   └      ┘          └      ┘          └      ┘
   symmetric         lower-triangular   upper-triangular
```

### (a) Concept & intuition
> **Definition — positive-definite:** symmetric `A` with `xᵀAx > 0` for all `x ≠ 0` (equivalently, all eigenvalues `> 0`). Covariance matrices and kernel/Gram matrices are (semi-)SPD.

Every SPD matrix factors **uniquely** as `A = LLᵀ` with `L` lower-triangular and positive diagonal. It's essentially "half" the work of other factorizations, and numerically very stable.

### (b) Worked-by-hand math (2×2)
Write `L = [[l₁₁,0],[l₂₁,l₂₂]]` and match `A = LLᵀ = [[4,2],[2,5]]`:
```
l₁₁²          = 4    ⇒ l₁₁ = 2
l₂₁·l₁₁       = 2    ⇒ l₂₁ = 2/2 = 1
l₂₁² + l₂₂²   = 5    ⇒ 1 + l₂₂² = 5 ⇒ l₂₂ = 2
⇒  L = [[2,0],[1,2]]
```
(You go top-left to bottom-right, solving one unknown at a time — a mini Gaussian elimination.)

### (c) Python / ML
```python
S2 = np.array([[4,2],[2,5]], float)
L2 = np.linalg.cholesky(S2)
print(L2)                       # -> [[2,0],[1,2]]
print(np.allclose(L2@L2.T, S2)) # -> True

S3 = np.array([[4,2,2],[2,5,3],[2,3,6]], float)
print(np.linalg.cholesky(S3))   # -> [[2,0,0],[1,2,0],[1,1,2]]
```
> **ML lens.** (1) **Sampling a multivariate Gaussian** `x ~ N(μ, Σ)`: draw standard-normal `z`, then `x = μ + Lz` where `Σ = LLᵀ` — Cholesky "colours" white noise with the right correlations. (2) **Gaussian Processes** solve SPD systems `Kα = y` via Cholesky (faster & more stable than `inv`). (3) It's the go-to way to test "is this matrix SPD?" (Cholesky fails if not).

---

## 8. Singular Value Decomposition — SVD (§4.5)

### 🧠 Plain English
Eigendecomposition only works for *some square* matrices. **SVD works for EVERY matrix** — any shape, rank, rectangular. It says: **any transformation = rotate, then stretch along perpendicular axes, then rotate again.** A unit circle always maps to an ellipse; the ellipse's axis lengths are the **singular values**.

```
        Vᵀ (rotate)        Σ (stretch axes)        U (rotate)
   ●───────────────►   ●───────────────►   ●───────────────►
  unit circle          aligned to axes      stretched ellipse    final ellipse
                                             (by σ₁, σ₂)
   A = U Σ Vᵀ   :   the σ's are the ellipse's semi-axis lengths (always ≥ 0)
```

### (a) Concept & intuition
For **any** `A ∈ ℝ^{m×n}`:
```
A = U Σ Vᵀ
  U  (m×m) orthogonal  — output rotation; columns = "left singular vectors"
  Σ  (m×n) diagonal    — singular values σ₁ ≥ σ₂ ≥ … ≥ 0 (the stretches)
  Vᵀ (n×n) orthogonal  — input rotation; rows = "right singular vectors"
```
Connections that make it concrete:
- `σᵢ = √(eigenvalues of AᵀA)` (and of `AAᵀ`) — singular values are the "eigenvalues" of the symmetric matrices you can build from `A`.
- **Number of nonzero σ's = rank of `A`.**
- For a **symmetric PSD** matrix, SVD = eigendecomposition (`σ = λ`).

### (b) Worked-by-hand math
Take `M = [[1,0,1],[0,1,1]]` (a 2×3 matrix — not square, so eigendecomposition can't touch it, but SVD can).
Build the small symmetric matrix `MMᵀ` (2×2):
```
MMᵀ = [1·1+0·0+1·1   1·0+0·1+1·1] = [2  1]
      [0·1+1·0+1·1   0·0+1·1+1·1]   [1  2]
```
That's our friend `[[2,1],[1,2]]` with eigenvalues `3` and `1`. So the **singular values** are
```
σ₁ = √3 ≈ 1.732,   σ₂ = √1 = 1.
```
Two nonzero singular values ⇒ `rank(M) = 2`. The left singular vectors are the eigenvectors of `MMᵀ`: `(1,1)/√2` and `(1,−1)/√2`.

### (c) Python / ML (Lab 5)
```python
M = np.array([[1,0,1],[0,1,1]], float)
U, S, Vt = np.linalg.svd(M)
print(np.round(S,4))                       # -> [1.7321 1.]  = (√3, 1)
print(np.round(S**2,4))                     # -> [3. 1.]  = eigenvalues of M Mᵀ
print(np.allclose((U[:,:2]*S)@Vt[:2], M))   # -> True  (reconstructs M)
```
> **ML lens.** SVD is arguably the **most important matrix tool in ML**: it powers **PCA** (components = right singular vectors of centered data), **Latent Semantic Analysis** for text, the **pseudoinverse** `A⁺ = VΣ⁺Uᵀ` (Ch. 1 least squares!), **recommender systems**, **image compression**, and **denoising** — all via the low-rank idea next.

---

## 9. Low-rank matrix approximation (§4.6, Lab 5)

### 🧠 Plain English
SVD writes a matrix as a **sum of simple "layers,"** each weighted by a singular value, biggest first:
`A = σ₁·(layer 1) + σ₂·(layer 2) + …`. Since the first layers carry the most, you can **throw away the small ones** and keep a great approximation using far less data. That's **compression, PCA, and recommender systems** in one idea.

```
   A   ≈   σ₁ u₁v₁ᵀ  +  σ₂ u₂v₂ᵀ  +  … +  σ_k u_k v_kᵀ    (keep only the top k)
           ████          ██                  ▁
           biggest       smaller             tiny (drop these → little error lost)
```

### (a) Concept & intuition
Each **rank-1 layer** is `σᵢ·uᵢvᵢᵀ` (an outer product — a full grid built from two vectors). Keeping the top `k` gives `A_k`, the **rank-`k` approximation**.
> **Eckart–Young theorem:** `A_k` (top-`k` SVD) is the **best possible** rank-`k` approximation of `A`; the leftover error (in spectral norm) is exactly the **next singular value** `σ_{k+1}`.

So the singular values are a **menu of "how much you lose"** if you truncate there. A big gap after `σ_k` means "rank `k` is basically enough."

### (b) Worked-by-hand math
Consider a tidy "ratings" matrix (4 users × 4 items) with two clear blocks:
```
R = [5 5 0 0]     users 1–2 love items 1–2,
    [5 5 0 0]     users 3–4 love items 3–4.
    [0 0 4 4]     Two independent "taste groups" ⇒ rank 2.
    [0 0 4 4]
```
Its singular values are `σ = (10, 8, 0, 0)`. The **rank-1** approximation keeps only `σ₁=10`:
it perfectly reconstructs the top-left `5`-block and zeros the rest — capturing the *strongest* pattern. The error of this rank-1 fit equals the next singular value, `σ₂ = 8` (Eckart–Young).

### (c) Python / ML (verified)
```python
R = np.array([[5,5,0,0],[5,5,0,0],[0,0,4,4],[0,0,4,4]], float)
U,S,Vt = np.linalg.svd(R)
print(np.round(S,4))                              # -> [10. 8. 0. 0.]  (rank 2)
rank1 = S[0]*np.outer(U[:,0], Vt[0])              # keep the single biggest layer
print(np.round(rank1,4))                          # -> reconstructs the 5-block, rest ~0
print("%.4f == %.4f" % (np.linalg.norm(R-rank1,2), S[1]))  # -> 8.0000 == 8.0000
```
> **ML lens.** (1) **Recommender systems (Netflix Prize):** users×items ratings are low-rank; SVD fills in the blanks. (2) **PCA = low-rank approximation of centered data** — keep the top-`k` directions of variance. (3) **Image compression:** an image is a matrix; keeping the top singular values shrinks file size with little visible loss. (4) **Noise reduction / embeddings:** small singular values are often noise — dropping them denoises and yields compact **latent features**.

---

## 10. 🧩 Problem-Solving Workshop (data & ML scenarios)

> Same loop: **① read the story → ② turn it into a matrix → ③ decompose/solve by hand → ④ ML interpretation → ⑤ verify in code.**

### Problem 1 — Is this feature matrix usable? (determinant as a redundancy alarm)
**Story.** You have 3 features and this "feature-relationship" matrix. Before fitting a model you want to know: are the features independent, or is one redundant (which would break the fit, Ch. 1)?
```
F = [1 2 3]
    [4 5 6]
    [7 8 10]
```
**② → ③ Solve.** Compute `det(F)`. From §2: `det(F) = −3 ≠ 0`. Nonzero ⇒ **full rank, invertible ⇒ features independent ⇒ safe to fit.** (Had we used the row `[7,8,9]` instead of `[7,8,10]`, the third row would be `2·row2 − row1`, giving `det = 0` — a redundancy alarm.)

**④ ML lens.** `det ≠ 0` is a one-number **go/no-go** check for multicollinearity in a square feature/Gram matrix. In practice for tall data we check the **rank** or **condition number** of `XᵀX` (same spirit).

**⑤ Python.**
```python
F = np.array([[1,2,3],[4,5,6],[7,8,10]], float)
print(round(np.linalg.det(F)))                 # -> -3  (nonzero => independent)
print(round(np.linalg.det(np.array([[1,2,3],[4,5,6],[7,8,9]],float))))  # -> 0 (redundant!)
```

---

### Problem 2 — PCA by hand: find the main direction of a cloud of points
**Story.** Three data points: `(2,0), (0,2), (2,2)`. In which single direction does the data vary the most? (That's the **1st principal component**.)

**② → ③ Solve.** PCA = eigen-analysis of the **covariance** (here we use the symmetric matrix `MMᵀ`-style scatter). We already know from §8 that projecting onto `u = (1,1)/√2` gave the spread. Concretely, the top eigenvector of the data's scatter matrix points along `(1,1)` — the diagonal — because the points trend from bottom-left to top-right. Projecting each point onto `u`:
```
(2,0)·u = 2/√2 = 1.414     (0,2)·u = 1.414     (2,2)·u = 4/√2 = 2.828
```
These 1-D coordinates are the **PCA scores** — a 2-D dataset squeezed to 1-D along its main axis.

**④ ML lens.** This is **dimensionality reduction**: replace two correlated features by one "combined index" (like turning length & width into a single "size"). The eigenvalue tells you what fraction of variance you kept.

**⑤ Python (verified).**
```python
u = np.array([1,1], float)/np.sqrt(2)
pts = np.array([[2,0],[0,2],[2,2]], float)
print(np.round(pts@u, 4))            # -> [1.4142 1.4142 2.8284]  (PCA scores along main axis)
```

---

### Problem 3 — Compress a ratings table with SVD (recommenders)
**Story.** A streaming service has this users×items ratings block. Storage is expensive — can you capture the pattern with fewer numbers, and even *predict* missing tastes?
```
R = [5 5 0 0]     (two taste groups: sci-fi lovers vs rom-com lovers)
    [5 5 0 0]
    [0 0 4 4]
    [0 0 4 4]
```
**② → ③ Solve.** SVD gives `σ = (10, 8, 0, 0)` → only **rank 2** really matters. The top-2 layers reconstruct `R` *exactly* using vectors instead of the full grid. The two big singular vectors literally *are* the two taste groups.

**④ ML lens.** Real ratings matrices are huge and mostly empty, but **low-rank** — a handful of latent "taste factors" explain most preferences. SVD/low-rank factorization (**matrix completion**) both **compresses** and **fills in missing ratings** = the core of recommender systems.

**⑤ Python (verified).**
```python
R = np.array([[5,5,0,0],[5,5,0,0],[0,0,4,4],[0,0,4,4]], float)
U,S,Vt = np.linalg.svd(R)
print(np.round(S,4))                              # -> [10. 8. 0. 0.]  (only 2 real patterns)
approx2 = (U[:,:2]*S[:2])@Vt[:2]
print(np.allclose(approx2, R))                    # -> True  (rank-2 captures everything)
```
> 🧠 **Intuition thread for the whole chapter:** *find the special stretch-directions (eigen/singular), sort them big-to-small, and keep the ones that matter.* That single habit is PCA, compression, recommenders, and stability analysis all at once.

---

## 11. Synthesis + cheat-sheet

| Tool | What it gives you | Where it powers ML |
|---|---|---|
| **determinant** | area/volume scale; `0 ⇒ singular` | redundancy/invertibility check; change-of-variables (flows); Gaussian likelihood |
| **trace** | sum of diagonal = sum of eigenvalues | total variance (PCA); nuclear-norm regularizers; trace trick |
| **eigenvectors/values** | stretch directions & factors | **PCA**, PageRank, spectral clustering, stability (exploding/vanishing) |
| **eigendecomposition `PDP⁻¹`** | diagonalize; fast powers `Aᵏ` | Markov chains, graph diffusion, whitening |
| **power method** | dominant eigenvector cheaply | PageRank, top principal component |
| **Cholesky `LLᵀ`** | "square root" of SPD matrix | Gaussian sampling, GP solves, SPD test |
| **SVD `UΣVᵀ`** | universal rotate-stretch-rotate | PCA, LSA, pseudoinverse, recommenders, compression |
| **low-rank (top-k SVD)** | best rank-`k` approximation | compression, denoising, matrix completion, embeddings |

### Cheat-sheet (exam-ready)
- **det:** `2×2 = ad−bc`; 3×3 by cofactor expansion; **triangular = product of diagonal**. `det=0 ⇔ singular`. `det(AB)=det(A)det(B)`.
- **trace = Σ diagonal = Σ eigenvalues**; `det = Π eigenvalues`. Cyclic: `tr(ABC)=tr(BCA)`.
- **Eigen:** solve `det(A−λI)=0` for `λ`, then `(A−λI)v=0` for `v`. Symmetric ⇒ real λ, orthogonal eigenvectors.
- **Diagonalize:** `A = PDP⁻¹` (eigenvectors in `P`, eigenvalues in `D`). **Powers:** `Aᵏ = PDᵏP⁻¹`.
- **Power method:** repeatedly `x ← Ax/‖Ax‖` → dominant eigenvector; `λ ≈ xᵀAx`.
- **Cholesky:** SPD only, `A=LLᵀ`, `L` lower-triangular; the matrix "square root."
- **SVD:** `A=UΣVᵀ`, works for **any** matrix; `σ=√eig(AᵀA)`, `#σ>0 = rank`. Unit circle → ellipse.
- **Low-rank (Eckart–Young):** best rank-`k` = top-`k` SVD; truncation error = `σ_{k+1}`.

**Mini-glossary recap**
- **determinant / trace** — volume-scale (0 ⇒ collapse) / diagonal-sum (= Σλ).
- **eigenvector / eigenvalue** — direction only stretched / its stretch factor (`Av=λv`).
- **diagonalizable / eigendecomposition** — has enough eigenvectors / `A=PDP⁻¹`.
- **SPD / Cholesky** — symmetric, all λ>0 / its `LLᵀ` square root.
- **singular values / SVD** — universal stretch factors (`≥0`) / `A=UΣVᵀ` for any matrix.
- **rank-k approximation** — best low-rank fit (keep top-k σ); basis of PCA & compression.

---

### ✅ Reproduce all outputs
Run `python _verify_ch3.py` in this `Summaries` folder. Every numeric result above was generated and verified (NumPy 2.4).
