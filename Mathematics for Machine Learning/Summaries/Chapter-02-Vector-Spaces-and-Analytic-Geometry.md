# Chapter 2 — Vector Spaces & Analytic Geometry (Detailed)

**Course:** AMLSIZC416 — Mathematical Foundations for Machine Learning
**Module 2 / Sessions 2–3:** *Vector spaces, linear independence, basis & rank, linear mappings, affine spaces; norms, inner products, lengths & distances, angles & orthogonality, orthonormal basis.*
**Reading (mapped):** **T1** §2.4–2.8 and §3.1–3.5.
**Going deeper (this doc adds):** §2.7.3 image/kernel & the **Rank–Nullity theorem**, §3.6 orthogonal complement, §3.8 **orthogonal projections** (the engine behind PCA, least-squares & SVMs), and the **Gram–Schmidt process** (Handout **Lab 2**).

> ### 📖 How to read this document (beginner note)
> I assume you're **new to the math and to ML**. Every topic comes in layers: **🧠 Plain English** (the idea in everyday words + a picture) → **(a) Concept & intuition** → **(b) Worked-by-hand math** → **(c) Python / ML**. Keep the **Jargon Buster** below open. All Python outputs are real (verified with NumPy 2.4 — run `_verify_ch2.py`).
>
> **Where we are:** Chapter 1 asked *"how do I solve `Ax=b`?"* Chapter 2 zooms out and asks *"what is the **space** these vectors live in, and how do we measure **length, angle, and closeness** inside it?"* That second question — geometry — is the foundation of PCA, similarity search, SVMs, and least squares.

---

## 🔤 Jargon Buster (keep this open while reading)

| Term | Plain meaning |
|---|---|
| **vector space** | a "playground" of vectors where adding and scaling never throws you out of the playground. `ℝⁿ` is the one we use. |
| **subspace** | a smaller playground *inside* a bigger one that still contains the origin and is closed under `+` and scaling. A flat sheet through `0`. |
| **closure** | "you can't escape": add/scale members and you're still inside the set. |
| **linear combination** | mix vectors by scaling and adding: `λ₁x₁ + λ₂x₂ + …`. |
| **span** | *all* the points you can reach by linear combinations of some vectors. |
| **linearly independent** | no vector is redundant — none can be built from the others. |
| **linearly dependent** | at least one vector is redundant (a combo of the rest). |
| **basis** | a *minimal* set of independent vectors that spans the whole space — the "axes." |
| **dimension** | how many vectors are in a basis = number of independent directions. |
| **rank** | number of independent columns (= independent rows) of a matrix = "true size" of what it does. |
| **coordinates** | the recipe `(α₁,…,αₙ)` telling how much of each basis vector builds a point. |
| **linear map / transformation** | a function `Φ(x)=Ax` that respects `+` and scaling — every matrix is one. |
| **image / range** | all outputs a map can produce (= column space). |
| **kernel / null space** | all inputs squashed to `0`. |
| **affine space** | a subspace **shifted off the origin** — a line/plane not through `0`. |
| **norm `‖x‖`** | the **length** of a vector. |
| **unit vector** | a vector of length 1. |
| **inner product `⟨x,y⟩`** | the machine that creates length & angle; the **dot product** `xᵀy` is the usual one. |
| **orthogonal** | perpendicular: `⟨x,y⟩ = 0`. |
| **orthonormal** | perpendicular **and** each of length 1. |
| **ONB** | an orthonormal basis — the "nicest" set of axes. |
| **orthogonal matrix** | a square matrix `Q` with `QᵀQ=I` — a pure rotation/reflection; `Q⁻¹=Qᵀ`. |
| **projection** | drop a vector straight down onto a subspace to get its closest point there. |
| **SPD matrix** | symmetric & "positive" (`xᵀAx>0`); think covariance / kernel matrices. |

> 🧠 **One-line mantra:** *Chapter 1 was algebra (solve for numbers); Chapter 2 is **geometry** (spaces, lengths, angles, and shadows) — and geometry is where PCA, cosine-similarity, and SVMs come from.*

---

## 🗺️ Mental map of this chapter

```
                         VECTORS LIVE IN A SPACE (ℝⁿ)
                                    │
        ┌───────────── STRUCTURE (no ruler yet) ─────────────┐
        ▼                          ▼                         ▼
   vector space /            span, independence,        linear map = matrix
   subspace (§1)             basis, rank (§2,§3)         Φ(x)=Ax (§4)
        │                          │                         │
        │                          │              image (outputs) & kernel (§5)
        │                          │                  rank + nullity = n
        │                     affine = subspace shifted off 0 (§6)
        │
        └──────── ADD A RULER: inner product ⟨·,·⟩ (§8) ────────┐
                                    │                            │
              ┌─────────────────────┼───────────────┬───────────┘
              ▼                     ▼               ▼
        length ‖x‖ (§7,§9)    angle & cosine (§10)  orthogonality (§10,§11,§12)
                                    │
                                    ▼
                    PROJECTION = closest point in a subspace (§13)
                                    │
                 ┌──────────────────┴───────────────────┐
                 ▼                                       ▼
       LEAST SQUARES / regression            Gram–Schmidt → orthonormal axes
       (project y onto columns of X)         (Lab 2; basis of QR, PCA)
```

Each section below is one box in this map. Glance back whenever you feel lost.

---

## Table of Contents
1. Vector spaces & subspaces (§2.4)
2. Linear combinations & linear independence (§2.5)
3. Basis, span, dimension & rank (§2.6)
4. Linear mappings, transformation matrices, coordinates (§2.7)
5. Image, kernel & the Rank–Nullity theorem (§2.7.3)
6. Affine spaces (§2.8)
7. Norms — measuring length (§3.1)
8. Inner products & SPD matrices (§3.2)
9. Lengths, distances & metrics (§3.3)
10. Angles & orthogonality (§3.4)
11. Orthonormal basis & orthogonal matrices (§3.5)
12. Orthogonal complement (§3.6)
13. Orthogonal projections — the ML workhorse (§3.8)
14. Gram–Schmidt process (Lab 2)
15. 🧩 Problem-Solving Workshop (data & ML scenarios)
16. Big-picture ML synthesis & cheat-sheet

---

## 1. Vector spaces & subspaces (§2.4)

### 🧠 Plain English
A **vector space** is just a "playground" where two moves — **adding** vectors and **scaling** them — always keep you inside the playground. A **subspace** is a smaller flat playground *inside* it that still passes through the origin `0`. Picture a flat sheet of paper through the center of a room: stay on the sheet no matter how you add/stretch arrows on it.
```
   ℝ³ (the whole room)
        ┌───────────────┐
        │      ╱         │     ← U: a plane through 0 (a 2-D SUBSPACE)
        │     ╱  U       │        every combo of arrows in U stays in U
        │    ╱ • 0       │
        └───────────────┘
```

### (a) Concept
Chapter 1 solved `Ax=b`. Now we ask: *what is the arena these vectors live in?* A **vector space** `V = (𝒱, +, ·)` is a set `𝒱` with two operations — **addition** (`+`) and **scalar multiplication** (`·`) — obeying 8 axioms (associativity, commutativity of `+`, a zero vector `0`, additive inverses, distributivity, `1·x = x`, etc.). The point of the axioms: you can **add** and **scale** and never leave the set (**closure**).

The space we use in ML is **`ℝⁿ`**.

A **subspace** `U ⊆ V` is a subset that is *itself* a vector space under the same operations. The practical test — `U` is a subspace iff:
1. `0 ∈ U` (contains the origin),
2. **closed under addition** (`u, w ∈ U ⇒ u+w ∈ U`),
3. **closed under scaling** (`u ∈ U, λ∈ℝ ⇒ λu ∈ U`).

**Deep fact (bridge to Chapter 1):** *Every* subspace of `ℝⁿ` is the **solution set of some homogeneous system `Ax = 0`.* So "subspace" and "null space of a matrix" are two names for the same geometric object — a flat through the origin.

### (b) Worked math
Is `U = {(a, b, 0) : a,b ∈ ℝ}` (the xy-plane in `ℝ³`) a subspace? `0=(0,0,0)∈U` ✓; `(a,b,0)+(c,d,0)=(a+c,b+d,0)∈U` ✓; `λ(a,b,0)=(λa,λb,0)∈U` ✓ → **yes**.
Is `W = {(a,b,1)}` (plane shifted up by 1) a subspace? `0∉W` → **no** (it's an *affine* space, §6).

### (c) ML lens
Feature space, weight space, the span of your data's columns, and the set of predictions a linear model can produce are all **subspaces**. When we later say "PCA finds the best `k`-dimensional subspace," this is the exact object.

---

## 2. Linear combinations & linear independence (§2.5)

### 🧠 Plain English
A **linear combination** is any "mix" of vectors made by scaling and adding them. Vectors are **independent** if none of them is a *leftover* you could rebuild from the others — every one adds a genuinely new direction. They're **dependent** if at least one is redundant.
```
 INDEPENDENT (2 real directions)     DEPENDENT (3rd is redundant)
        ↑ y                                ↑
        │                                  │   ↗  (this one = mix of the
        │                                  │  ↗    other two → adds nothing)
        └────► x                           └────►
```

### (a) Concept
A **linear combination** of `x₁,…,x_k` is `v = λ₁x₁ + … + λ_k x_k`. The vectors are **linearly independent** if the *only* way to get `0` is the trivial `λ₁=…=λ_k=0`. If some **non-trivial** combination gives `0`, they're **dependent** — meaning at least one vector is redundant (expressible via the others).

**Intuition (the Nairobi→Kigali example):** "506 km NW" and "374 km SW" are independent (neither is a scaling of the other). Adding "751 km W" makes the set dependent because West is already reachable by combining NW and SW — it carries *no new direction*.

### (b) Worked math — the Gaussian-elimination test
Put the vectors as **columns** of a matrix, reduce to row-echelon form:
- **pivot columns** → independent vectors;
- **non-pivot columns** → dependent (each = a combination of pivots to its left).
All independent ⟺ every column is a pivot.

**Example 2.14** — `x₁=(1,2,3,4)ᵀ, x₂=(−1,1,0,2)ᵀ, x₃=(1,2,−1,1)ᵀ` reduce to three pivots → **independent**.
**Example 2.15** — four vectors expressed via an independent set `b₁..b₄`; RREF reveals `x₄ = −7x₁ − 15x₂ − 18x₃`, so they are **dependent**.

Useful shortcuts: any set containing `0` is dependent; if `m > k`, then `m` combinations of `k` vectors are always dependent (you can't have more independent directions than you started with).

### (c) Python / ML
```python
X = np.array([[1,-1,1],[2,1,2],[3,0,-1],[-4,2,1]], float)  # Example 2.14 as columns
print(np.linalg.matrix_rank(X), X.shape[1])   # -> 3 3  => all independent

A = np.array([[1,-4,2,17],[-2,-2,3,-10],[1,0,-1,11],[-1,4,-3,1]], float)  # Example 2.15
print(np.linalg.matrix_rank(A), A.shape[1])   # -> 3 4  => dependent
print(np.allclose(A[:,3], -7*A[:,0]-15*A[:,1]-18*A[:,2]))   # -> True (the exact relation)
```
> **ML lens — multicollinearity.** Linearly dependent feature columns are *multicollinearity*. They make `XᵀX` singular, blow up regression weights, and mean some features are redundant. `rank(X) < #features` is the precise diagnosis; the fix is dropping/combining features or regularizing.

---

## 3. Basis, span, dimension & rank (§2.6)

### 🧠 Plain English
- **Span** = every place you can reach by mixing some vectors.
- A **basis** is the "just right" set of vectors: **enough** to reach everything (spanning), with **none wasted** (independent). These are your **axes**.
- **Dimension** = how many axes you need (= size of a basis).
- **Rank** = how many *truly different* columns a matrix has = the real "size" of the transformation it does.
```
   too few → can't reach all      just right (BASIS)      too many → redundant
   (only a line)                   (fills the plane)       (one vector wasted)
      →                              ↑  ↗                    ↑ ↗ →
                                     └──►                     └──►
```

### (a) Concept
- **Span** of a set = *all* linear combinations of it. A **generating set** of `V` spans all of `V`.
- A **basis** = a *minimal* generating set = a *maximal* independent set. It's the "just right" number of vectors: enough to reach everything, none wasted.
- **Key uniqueness:** with a basis, **every** vector has *exactly one* coordinate representation.
- **Dimension** `dim(V)` = number of basis vectors = number of independent directions. (`ℝⁿ` has dimension `n`; the standard/canonical basis is `e₁,…,eₙ`.)
- Bases are **not unique** (infinitely many), but **all have the same size**.

**Rank** `rk(A)` = number of independent columns = number of independent rows (**column rank = row rank**, a beautiful theorem). It measures the "true dimensionality" of what `A` does.

### (b) Worked math
**Finding a basis of a span:** stack spanning vectors as columns, row-reduce, keep the vectors at **pivot columns**.
**Example 2.17** — four vectors in `ℝ⁵` reduce so that `x₁,x₂,x₄` are pivots → `{x₁,x₂,x₄}` is a basis of `U`, `dim(U)=3`.
**Example 2.18** — `A=[[1,2,1],[−2,−3,1],[3,5,0]]` reduces to 2 pivots → `rk(A)=2`.

**Rank facts you must know:**
- `rk(A) = rk(Aᵀ)`.
- `A ∈ ℝⁿˣⁿ` invertible ⟺ `rk(A) = n` (**full rank**).
- `Ax=b` solvable ⟺ `rk(A) = rk([A|b])`.
- solution space of `Ax=0` has dimension `n − rk(A)`.
- **full rank** means `rk = min(m,n)`; otherwise **rank-deficient**.

### (c) Python / ML
```python
A = np.array([[1,2,1],[-2,-3,1],[3,5,0]], float)
print(np.linalg.matrix_rank(A))   # -> 2   (rank-deficient 3x3 => singular)
```
> **ML lens.** `rank` = intrinsic dimensionality. A data matrix with `rank ≪ #features` is *low-rank* — exactly what PCA, matrix completion (recommenders), and embeddings exploit to compress. Full-rank `X` ⟺ features not redundant ⟺ unique least-squares fit.

---

## 4. Linear mappings, transformation matrices, coordinates (§2.7)

### 🧠 Plain English
A **linear map** is a "well-behaved" transformation: it can rotate, stretch, or flip space, but it keeps grid-lines straight and evenly spaced and keeps the origin fixed. The big idea: **every linear map is just a matrix**, and multiplying `Ax` *applies* that transformation to `x`.
```
     input x            matrix A = a machine          output Ax
     ┌───┐   ────────►  [ rotate / stretch / flip ]  ────────►  ┌───┐
     grid                                                        tilted/scaled grid
```

### (a) Concept
A **linear mapping** (a.k.a. linear transformation / homomorphism) `Φ: V→W` preserves the vector-space structure:
```
Φ(λx + ψy) = λΦ(x) + ψΦ(y)
```
i.e. it commutes with addition and scaling. **Every linear map between finite-dimensional spaces is a matrix**, and every matrix is a linear map — this is *the* central identification of the chapter.

Special maps: **injective** (one-to-one), **surjective** (onto), **bijective** (both). A bijective linear map is an **isomorphism** — and two finite-dimensional spaces are isomorphic **iff they have the same dimension** (so `ℝ^{m×n} ≅ ℝ^{mn}`; a matrix and its flattened vector are "the same").

**Coordinates.** Fix an *ordered basis* `B=(b₁,…,bₙ)`. Any `x` is `x = α₁b₁+…+αₙbₙ`; the vector `α=(α₁,…,αₙ)ᵀ` is the **coordinate representation** of `x` in basis `B`. **A basis defines a coordinate system.** The same arrow `x` has different coordinates in different bases.

**Transformation matrix.** For `Φ: V→W` with bases `B` (of `V`) and `C` (of `W`), write each `Φ(bⱼ)` in the `C`-basis; those coordinate columns form `A_Φ`. Then coordinates transform by
```
ŷ = A_Φ x̂.
```

### (b) Worked math
**Example 2.20 (coordinates).** `x` with standard coords `(2,3)ᵀ` means `x=2e₁+3e₂`. In basis `b₁=(1,−1)ᵀ, b₂=(1,1)ᵀ`, the same `x` has coords `½(−1,5)ᵀ` because `x = −½b₁ + 5⁄2 b₂`.

**Example 2.22 (transforms of a picture).** Applied to 400 points forming a square:
- `A₁ = [[cos45,−sin45],[sin45,cos45]]` → **rotation** by 45°,
- `A₂ = [[2,0],[0,1]]` → **stretch** x-axis by 2,
- `A₃ = ½[[3,−1],[1,−1]]` → reflection+rotation+stretch combined.

### (c) Python / ML
```python
th = np.pi/4
R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])  # rotation
S = np.array([[2,0],[0,1]])                                       # stretch
print(R @ np.array([1,0]))    # -> [0.7071 0.7071]  (e1 rotated 45°)
print(S @ np.array([1,1]))    # -> [2 1]
print(round(np.linalg.det(R),4))          # -> 1.0  (rotation preserves area)
```
> **ML lens — a neural net layer *is* a transformation matrix.** `h = Wx` maps coordinates from the input basis to a hidden basis; training *learns the matrix*. "Change of basis" is literally what an embedding/whitening/PCA-rotation layer does. Data augmentation (rotations, scalings) applies these very matrices.

---

## 5. Image, kernel & the Rank–Nullity theorem (§2.7.3)

### 🧠 Plain English
Feed all vectors through a matrix `A`. Two special sets appear: the **image** = everything that *comes out* (the reachable outputs), and the **kernel** = everything that gets *crushed to zero* on the way in. The **Rank–Nullity theorem** says every input dimension is either "kept" (shows up in the image) or "killed" (shows up in the kernel): `kept + killed = total`.
```
     INPUT space (n dims)            OUTPUT space
     ┌──────────────┐
     │  kernel  ───────► 0   (killed directions)
     │              │
     │  the rest ──────► image  (kept directions)
     └──────────────┘
            nullity   +   rank   =   n
```

### (a) Concept
For `Φ: V→W` (matrix `A`):
- **Kernel / null space** `ker(Φ) = {v : Φ(v)=0}` — everything crushed to zero. It's a subspace of the **domain**.
- **Image / range** `Im(Φ) = {Φ(v)}` — everything reachable. For a matrix it's the **column space** = `span(columns of A)`, a subspace of the **codomain**, with `dim(Im) = rk(A)`.
- `Φ` injective ⟺ `ker(Φ) = {0}`.

**Rank–Nullity theorem** (the *fundamental theorem of linear maps*):
```
dim(ker Φ) + dim(Im Φ) = dim(V)      i.e.   nullity + rank = n
```
Beautifully: every input dimension is either "kept" (contributes to the image) or "killed" (contributes to the kernel).

### (b) Worked math (Example 2.25)
`Φ:ℝ⁴→ℝ²`, `A=[[1,2,−1,0],[1,0,0,1]]`. `rk(A)=2` so `dim(Im)=2` (all of `ℝ²` — surjective). By Rank–Nullity, `dim(ker)=4−2=2`. Solving `Ax=0` (RREF + minus-1 trick from Ch.1) gives
`ker(Φ) = span{(0,−½,1,0)ᵀ? …}` — a 2-D subspace. Check: `2 + 2 = 4 = n`. ✓

### (c) Python / ML
```python
A = np.array([[1,2,-1,0],[1,0,0,1]], float)
r = np.linalg.matrix_rank(A)
u,s,vt = np.linalg.svd(A); ker = vt[np.sum(s>1e-10):].T
print(r, ker.shape[1], r+ker.shape[1], A.shape[1])  # -> 2 2 4 4  (Rank-Nullity!)
print(A @ ker)   # -> ~0  (kernel basis really maps to 0)
```
> **ML lens.** A non-trivial kernel of `X` = directions in weight space that **don't change predictions** → parameters unidentifiable → infinitely many equal-loss models → why we **regularize**. The image (column space) is exactly the set of predictions a linear model can produce; least squares picks the point in it closest to `y` (§13).

---

## 6. Affine spaces (§2.8)

### (a) Concept
An **affine subspace** is a subspace **shifted off the origin**: `L = x₀ + U` where `x₀` is a support point and `U` a subspace (the "direction space"). Lines and planes *not through the origin* are affine, not linear. Affine maps = linear map + translation: `x ↦ Ax + b`.

Affine spaces don't contain `0` (in general) and aren't closed under linear combos — but they **are** closed under *affine* combinations (`Σλᵢxᵢ` with `Σλᵢ=1`).

### (b)/(c) ML lens
The **bias term** makes ML models affine, not linear: `ŷ = Wx + b`. A classifier's **decision boundary** `wᵀx + b = 0` is an affine hyperplane — precisely the object an SVM (Module 7) positions. The common trick "absorb `b` by appending a 1 to `x`" turns an affine map into a linear one in one higher dimension.

---

## 7. Norms — measuring length (§3.1)

### (a) Concept
Chapters so far had no notion of *length* or *angle* — a bare vector space is "structureless." A **norm** `‖·‖: V→ℝ` assigns a length, satisfying:
1. **Absolutely homogeneous:** `‖λx‖ = |λ|·‖x‖`,
2. **Triangle inequality:** `‖x+y‖ ≤ ‖x‖+‖y‖`,
3. **Positive definite:** `‖x‖≥0`, and `=0` iff `x=0`.

### (b) Worked math — the `ℓₚ` family
- **ℓ₁ (Manhattan):** `‖x‖₁ = Σ|xᵢ|`.
- **ℓ₂ (Euclidean, default):** `‖x‖₂ = √(Σxᵢ²) = √(xᵀx)`.
- **ℓ∞ (max):** `‖x‖∞ = maxᵢ|xᵢ|`.
For `x=(3,−4)`: `‖x‖₁=7`, `‖x‖₂=5`, `‖x‖∞=4`.

### (c) Python / ML
```python
x = np.array([3,-4.])
print(np.linalg.norm(x,1), np.linalg.norm(x,2), np.linalg.norm(x,np.inf))  # 7.0 5.0 4.0
```
> **ML lens — norms *are* regularizers.** `ℓ₂` (ridge) shrinks weights smoothly; `ℓ₁` (lasso) drives weights to **exactly zero** → sparse feature selection (its diamond-shaped unit ball has corners on the axes). Loss functions (MSE = squared `ℓ₂` of residuals), gradient clipping (`ℓ₂`/`ℓ∞`), and distance metrics all pick a norm.

---

## 8. Inner products & SPD matrices (§3.2)

### 🧠 Plain English
So far a bare vector space has **no ruler and no protractor** — you can't measure length or angle. The **inner product** is that measuring tool. The everyday one is the **dot product** `xᵀy = Σxᵢyᵢ`: a single number that is **big when two vectors point the same way** and **zero when they're perpendicular**. Everything geometric (length, angle, distance) is built from it.
```
   x·y > 0  (same-ish direction)   x·y = 0  (perpendicular)   x·y < 0 (opposite-ish)
       ↗ ↗                              ↑                          ↗
                                        └──►                      ↙
```

### (a) Concept
An **inner product** `⟨·,·⟩` is the machine that *creates* geometry (length, angle, orthogonality). It's a mapping `V×V→ℝ` that is:
- **bilinear** (linear in each argument),
- **symmetric** `⟨x,y⟩=⟨y,x⟩`,
- **positive definite** `⟨x,x⟩>0` for `x≠0`.

The familiar **dot product** `⟨x,y⟩ = xᵀy = Σxᵢyᵢ` is the canonical example (giving a **Euclidean space**), but others exist: any **symmetric positive-definite (SPD) matrix** `A` defines `⟨x,y⟩ = xᵀAy`.

**SPD matrices** `A`: symmetric with `xᵀAx>0` for all `x≠0`. They're everywhere in ML — **covariance matrices**, **kernel/Gram matrices**, and Hessians of convex losses are all (semi-)SPD.

### (b) Worked math (Example 3.4)
`A₁=[[9,6],[6,5]]` is SPD: `xᵀA₁x = 9x₁²+12x₁x₂+5x₂² = (3x₁+2x₂)² + x₂² > 0`.
`A₂=[[9,6],[6,3]]` is **not**: `xᵀA₂x = (3x₁+2x₂)² − x₂²`, negative e.g. at `x=(2,−3)`.

### (c) Python / ML
```python
def is_spd(A): 
    return np.allclose(A,A.T) and np.all(np.linalg.eigvalsh(A) > 0)
print(is_spd(np.array([[9,6],[6,5]])), is_spd(np.array([[9,6],[6,3]])))  # True False
```
> **ML lens.** A **kernel** `k(x,y)` in SVMs/Gaussian processes must produce an SPD Gram matrix (Mercer's condition) — that's what guarantees a valid inner-product geometry in the (possibly infinite-dimensional) feature space. **Mahalanobis distance** uses `A = Σ⁻¹` (inverse covariance) as the inner-product matrix to account for correlated features.

---

## 9. Lengths, distances & metrics (§3.3)

### (a) Concept
An inner product **induces** a norm: `‖x‖ = √⟨x,x⟩`. (But not every norm comes from an inner product — `ℓ₁` doesn't.) The **distance** between vectors is `d(x,y)=‖x−y‖=√⟨x−y,x−y⟩` (Euclidean distance if the inner product is the dot product). A **metric** `d` satisfies positive-definiteness, symmetry, and the triangle inequality.

**Cauchy–Schwarz inequality** (the key bound that makes angles well-defined):
```
|⟨x,y⟩| ≤ ‖x‖·‖y‖.
```

### (b) Worked math
Different inner products give different lengths. For `x=(1,1)`: with the dot product `‖x‖=√2`; with `⟨x,y⟩=xᵀ[[1,−½],[−½,1]]y`, `⟨x,x⟩ = 1−1+1 = 1` so `‖x‖=1` — the *same arrow* is "shorter" under a different geometry.

**Inner products and distances behave oppositely:** similar vectors → **large** inner product but **small** distance.

### (c) ML lens
Distances power **k-NN**, **k-means**, hierarchical clustering, t-SNE/UMAP. The choice of metric *is* a modeling decision. Cosine (next section) vs Euclidean vs Mahalanobis changes what "similar" means.

---

## 10. Angles & orthogonality (§3.4)

### (a) Concept
Cauchy–Schwarz guarantees `−1 ≤ ⟨x,y⟩/(‖x‖‖y‖) ≤ 1`, so we can **define the angle**:
```
cos ω = ⟨x,y⟩ / (‖x‖·‖y‖),   ω ∈ [0,π].
```
Two vectors are **orthogonal** (`x⊥y`) iff `⟨x,y⟩=0`. If additionally both are unit length, they're **orthonormal**. Orthogonality generalizes "perpendicular" — and it's **inner-product-dependent**: vectors orthogonal under one inner product need not be under another.

### (b) Worked math (Example 3.6)
`x=(1,1), y=(1,2)`, dot product:
`cos ω = (1·1+1·2)/(√2·√5) = 3/√10 ≈ 0.9487`, so `ω = arccos(0.9487) ≈ 18.43°`.

### (c) Python / ML — cosine similarity
```python
x=np.array([1,1.]); y=np.array([1,2.])
cos = x@y/(np.linalg.norm(x)*np.linalg.norm(y))
print(round(cos,4), round(np.degrees(np.arccos(cos)),2))  # 0.9487 18.43
```
> **ML lens.** This exact formula is **cosine similarity** — the backbone of text/embedding retrieval (TF-IDF, word2vec, transformer embeddings, RAG search). Orthogonal features are uncorrelated/non-redundant; orthogonality underlies PCA components and error-correcting codes.

---

## 11. Orthonormal basis & orthogonal matrices (§3.5)

### (a) Concept
An **orthonormal basis (ONB)** `{b₁,…,bₙ}` satisfies `⟨bᵢ,bⱼ⟩=0` for `i≠j` and `⟨bᵢ,bᵢ⟩=1` — mutually perpendicular **unit** vectors. ONBs are the "nicest" coordinate systems: coordinates are just dot products (`αᵢ = ⟨x,bᵢ⟩`), no matrix inversion needed.

A square matrix `Q` whose **columns are orthonormal** is an **orthogonal matrix**:
```
QᵀQ = QQᵀ = I   ⇒   Q⁻¹ = Qᵀ.
```
Orthogonal transforms **preserve lengths and angles** (`‖Qx‖=‖x‖`) — they are **rotations/reflections**. Inverting one is free (just transpose).

### (b) Worked math
`b₁=1/√2·(1,1), b₂=1/√2·(1,−1)`: `b₁ᵀb₂ = ½(1−1)=0`, `‖b₁‖=‖b₂‖=1` → ONB.
Length preservation: `‖Qx‖² = (Qx)ᵀ(Qx) = xᵀQᵀQx = xᵀx = ‖x‖²`.

### (c) Python / ML
```python
Q = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
print(np.allclose(Q.T@Q, np.eye(2)), np.allclose(np.linalg.inv(Q), Q.T))  # True True
```
> **ML lens.** Orthogonal matrices give **numerically stable** computations (they don't amplify error — condition number 1, tying back to Ch.1 §8). **QR** and **SVD** decompositions produce orthogonal factors; **PCA** returns an orthogonal rotation to decorrelated axes; orthogonal weight initialization stabilizes deep nets.

---

## 12. Orthogonal complement (§3.6)

### (a) Concept
Given a subspace `U ⊆ V` (dim `M`) inside dim-`D` space, its **orthogonal complement** `U⊥` is the set of all vectors orthogonal to *everything* in `U`. It has dimension `D−M`, and `U ⊕ U⊥ = V`: every `x` splits **uniquely** into a part in `U` plus a part in `U⊥`.

A plane in `ℝ³` is described by its **normal vector** `w` (with `‖w‖=1`) — `w` spans the 1-D complement `U⊥`. This is how hyperplanes are specified.

### (c) ML lens
This decomposition is the geometric heart of **PCA** (keep the top subspace `U`, discard `U⊥`) and of **projections/least squares** (§13): the residual lives in `U⊥`. An SVM hyperplane is defined by its normal vector `w`.

---

## 13. Orthogonal projections — the ML workhorse (§3.8)

### 🧠 Plain English
A **projection** is the **shadow** a vector casts straight down onto a subspace (a line or plane). It finds the **closest point** in that subspace to your vector — and the leftover "error" points straight up, **perpendicular** to the subspace. This one idea *is* least-squares fitting and *is* PCA.
```
        x ●
          │╲   ← error (x − proj), perpendicular to the line
          │ ╲
   ───────●────────►  line/subspace U
        proj_U(x)  = the shadow = closest point in U to x
```

### (a) Concept
A **projection** `π: V→U` is a linear map with `π² = π` (projecting again changes nothing). An **orthogonal projection** sends `x` to the **closest** point in subspace `U`; equivalently, the error `x − π_U(x)` is **orthogonal to `U`**. This "closest point" property is *exactly* what least-squares fitting and dimensionality reduction need.

### (b) Worked math

**Projection onto a line** spanned by `b` (through origin), dot product:
```
coordinate:  λ = (bᵀx)/(bᵀb)
point:       π_U(x) = λb = (bbᵀ)/(bᵀb) · x
matrix:      P = (bbᵀ)/(bᵀb)         (symmetric, rank 1, P²=P)
```
**Example 3.10** — `b=(1,2,2)ᵀ`: `P = 1/9·[[1,2,2],[2,4,4],[2,4,4]]`. Projecting `x=(1,1,1)`: `π=1/9·(5,10,10)ᵀ`.

**Projection onto a general subspace** with basis `B=[b₁,…,b_m]` (columns). Orthogonality of the residual to every `bᵢ` gives the **normal equations**:
```
BᵀB λ = Bᵀx   ⇒   λ = (BᵀB)⁻¹Bᵀx
π_U(x) = B(BᵀB)⁻¹Bᵀx
P = B(BᵀB)⁻¹Bᵀ.
```
`(BᵀB)⁻¹Bᵀ` is the **pseudo-inverse** — the *same object* from Chapter 1's least squares! **Example 3.11** — `B=[[1,0],[1,1],[1,2]]`, `x=(6,0,0)ᵀ`: solve `[[3,3],[3,5]]λ=[6,6]ᵀ` → `λ=(5,−3)ᵀ`, projection `(5,2,−1)ᵀ`.

### (c) Python / ML
```python
# line projection (Example 3.10)
b=np.array([1,2,2.]); P=np.outer(b,b)/(b@b)
print(P @ np.array([1,1,1.]))       # -> [0.5556 1.1111 1.1111] = 1/9[5,10,10]
print(np.allclose(P@P,P))            # -> True (idempotent)

# subspace projection (Example 3.11)
B=np.array([[1,0],[1,1],[1,2]],float); x=np.array([6,0,0.])
lam=np.linalg.solve(B.T@B, B.T@x); print(lam)          # -> [ 5. -3.]
print(B@np.linalg.inv(B.T@B)@B.T @ x)                  # -> [ 5. 2. -1.]
```

> **ML lens — least squares *is* a projection.** Fitting `Xw ≈ y` means projecting `y` onto the column space of `X`; the fitted `ŷ = X(XᵀX)⁻¹Xᵀy = Py` is the projection, and the residual `y−ŷ` is orthogonal to every feature (`Xᵀ(y−ŷ)=0`). PCA projects data onto the top principal subspace to compress with minimal error. The "jitter"/ridge term `εI` added to `BᵀB` for stability is literally **ridge regression**.

```python
# Least squares residual is orthogonal to the feature columns:
rng=np.random.default_rng(1); X=rng.normal(size=(50,3))
y=X@np.array([1.,-2.,0.5])+0.05*rng.normal(size=50)
w=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@w
print(np.round(X.T@r,6))    # -> [-0. 0. 0.]   (orthogonality of projection)
```

---

## 14. Gram–Schmidt process (Handout Lab 2)

### 🧠 Plain English
You have some slanted, awkward axes and you want **clean perpendicular unit axes** that cover the same space. **Gram–Schmidt** does it one vector at a time: keep the first direction, then for each next vector **subtract off its shadow** on the directions you've already fixed (removing overlap), and **normalize** to length 1. Result: a tidy orthonormal set.
```
   ṽ₂ ●╲                       subtract ṽ₂'s shadow on u₁  →  w₂ ⟂ u₁  →  normalize
       │ ╲
       │  ╲ (shadow on u₁)
   ────●───►  u₁ (already fixed)
```

### (a) Concept
Given *any* basis `{ṽ₁,…,ṽₙ}` (possibly skewed, non-unit), **Gram–Schmidt** builds an **orthonormal** basis spanning the same space: take each vector, **subtract its projections onto the already-fixed directions**, then **normalize**.

### (b) Worked math
```
u₁ = ṽ₁/‖ṽ₁‖
w₂ = ṽ₂ − ⟨ṽ₂,u₁⟩u₁ ;   u₂ = w₂/‖w₂‖
w₃ = ṽ₃ − ⟨ṽ₃,u₁⟩u₁ − ⟨ṽ₃,u₂⟩u₂ ;   u₃ = w₃/‖w₃‖   …
```
Each `wᵢ` is the component of `ṽᵢ` orthogonal to everything chosen so far (its projection onto the current `U⊥`). This is exactly the **QR decomposition** `V = QR` with `Q` orthonormal.

### (c) Python / ML — Lab 2 implementation
```python
def gram_schmidt(V):
    V=V.astype(float); U=np.zeros_like(V)
    for i in range(V.shape[1]):
        w=V[:,i].copy()
        for j in range(i):
            w -= (U[:,j]@V[:,i])*U[:,j]     # subtract projection onto u_j
        U[:,i]=w/np.linalg.norm(w)          # normalize
    return U

V=np.array([[1,1,0],[1,0,1],[0,1,1]],float)
Q=gram_schmidt(V)
print(np.allclose(Q.T@Q, np.eye(3)))                       # -> True (orthonormal)
Qnp,_=np.linalg.qr(V); print(np.allclose(np.abs(Q),np.abs(Qnp)))  # -> True (matches QR)
```
> **ML lens.** Orthogonalization underlies **QR-based least squares** (more stable than normal equations), **whitening/decorrelation** of features, and the **power/Lanczos iterations** used to compute PCA components. The handout's Lab 2 is your hands-on version of this.

---

## 15. 🧩 Problem-Solving Workshop (data & ML scenarios)

> Same loop as before: **① read the story → ② turn it into vectors/geometry → ③ solve by hand → ④ ML interpretation → ⑤ verify in code.** These three problems turn the chapter's geometry (angles, distances, projections) into real ML tasks.

### Problem 1 — A mini search engine with cosine similarity
**Story.** Your vocabulary has just 3 words: **[machine, learning, cooking]**. Each document becomes a **word-count vector**. A user searches **"machine learning"** → query `q = (1,1,0)`. Rank these documents by relevance:
- **A:** "machine learning course" → `(1,1,0)`
- **B:** "cooking cooking recipe" → `(0,0,2)`
- **C:** "machine machine learning" → `(2,1,0)`

**② → ③ Solve by hand.** Relevance = **cosine similarity** `cos ω = (q·d)/(‖q‖‖d‖)` (angle between vectors; ignores length, compares *direction*).
- `cos(q,A) = (1·1+1·1+0)/(√2·√2) = 2/2 = 1.000` → **identical direction** (perfect match).
- `cos(q,B) = (0+0+0)/(√2·2) = 0.000` → **orthogonal** (nothing in common).
- `cos(q,C) = (2+1+0)/(√2·√5) = 3/√10 = 0.9487` → **very close**, but C over-weights "machine".
✅ **Ranking: A (1.00) > C (0.95) > B (0.00).**

**④ ML lens.** This is *exactly* how search engines, recommendation systems, and **RAG / vector databases** rank results: embed query and documents as vectors, sort by cosine. Why cosine and not raw dot product? Because it's **length-invariant** — a long document isn't "more relevant" just for repeating words. Notice B's score of 0 = the geometric meaning of **orthogonal = unrelated**.

**⑤ Python (verified).**
```python
q = np.array([1,1,0], float)
def cos(u,v): return u@v/(np.linalg.norm(u)*np.linalg.norm(v))
print(cos(q, np.array([1,1,0.])))   # A -> 1.0000
print(cos(q, np.array([0,0,2.])))   # B -> 0.0000
print(cos(q, np.array([2,1,0.])))   # C -> 0.9487
```

---

### Problem 2 — k-NN: your distance metric changes the answer
**Story.** A classifier labels a new point by its **nearest neighbour**. Query `q = (0,0)`. Training points: **A = (3,0)** labelled 🔴RED, **B = (2,2)** labelled 🔵BLUE. Which label does `q` get?

**② → ③ Solve by hand — two different rulers.**
- **Euclidean (ℓ₂)** `‖·‖₂ = √(Σxᵢ²)`: `d(q,A)=√(3²)=3.00`, `d(q,B)=√(2²+2²)=√8=2.83`. → **B is nearer → 🔵BLUE.**
- **Manhattan (ℓ₁)** `‖·‖₁ = Σ|xᵢ|`: `d(q,A)=3`, `d(q,B)=2+2=4`. → **A is nearer → 🔴RED.**
✅ **Same data, opposite predictions** — purely because we changed the distance.
```
        B(2,2)
         •        ℓ₂ "as the crow flies": B wins (2.83 < 3)
        ╱|        ℓ₁ "city blocks":       A wins (3 < 4)
   q •──┘ •A(3,0)
```

**④ ML lens.** **The metric is a modelling decision, not a given.** ℓ₂ rewards being close in a straight line; ℓ₁ (used in lasso, robust methods) is more forgiving of one large coordinate gap. k-NN, k-means, DBSCAN, t-SNE all live or die by this choice — and for correlated features you'd use **Mahalanobis distance** (§8's inner product with `A = Σ⁻¹`).

**⑤ Python (verified).**
```python
q=np.array([0,0.]); A=np.array([3,0.]); B=np.array([2,2.])
print(np.linalg.norm(q-A),   np.linalg.norm(q-B))    # L2: 3.0   2.8284  -> B (BLUE)
print(np.linalg.norm(q-A,1), np.linalg.norm(q-B,1))  # L1: 3.0   4.0     -> A (RED)
```

---

### Problem 3 — Projecting 2-D data onto one direction (the seed of PCA)
**Story.** You have 3 data points in 2-D and want to summarise each with a **single number** by projecting onto the diagonal direction `u = (1,1)/√2` (a unit vector). Points: `(2,0), (0,2), (2,2)`.

**② → ③ Solve by hand.** The projection **coordinate** of a point `x` onto unit vector `u` is just the dot product `u·x` (how far along `u` the shadow falls).
- `(2,0): u·x = (2+0)/√2 = √2 ≈ 1.414` → reconstructed point `1.414·u = (1,1)`.
- `(0,2): u·x = (0+2)/√2 = √2 ≈ 1.414` → reconstructed `(1,1)`.
- `(2,2): u·x = (2+2)/√2 = 2√2 ≈ 2.828` → reconstructed `(2,2)`.
✅ We compressed each 2-D point to **one coordinate** `[1.414, 1.414, 2.828]`. Note `(2,0)` and `(0,2)` collapse to the *same* summary — projection **loses** the perpendicular information (that's the price of compression).

**④ ML lens.** This is the mechanical heart of **PCA / dimensionality reduction**: represent high-dimensional data by its coordinates along a few chosen directions. PCA's extra idea (next modules) is to *pick the direction that keeps the most variance*. Here, along `u` the variance captured is `0.444` out of a total `1.778` — this diagonal keeps only part of the spread, which is exactly why PCA searches for the *best* direction instead of guessing one.

**⑤ Python (verified).**
```python
u = np.array([1,1], float)/np.sqrt(2)          # unit direction
pts = np.array([[2,0],[0,2],[2,2]], float)
coords = pts @ u                               # 1-number summary per point
print(np.round(coords,4))                      # -> [1.4142 1.4142 2.8284]
print(np.round(np.outer(coords,u),4))          # reconstructions: (1,1),(1,1),(2,2)
print(round(np.var(coords),4))                 # variance kept along u -> 0.4444
```
> 🧠 **Intuition:** a projection is a **shadow**. Choosing *where to shine the light* (which direction `u`) decides how much of the data's shape survives — the whole game of feature extraction.

---

## 16. Big-picture ML synthesis

| Concept | Where it powers ML |
|---|---|
| Subspace / span | Column space = model's reachable predictions; PCA subspace |
| Linear independence / rank | Multicollinearity; intrinsic data dimensionality; low-rank compression |
| Basis / coordinates / change of basis | Embeddings, whitening, PCA rotation, "features" themselves |
| Linear map = matrix | Every dense NN layer `Wx`; data augmentation transforms |
| Kernel / image / Rank–Nullity | Unidentifiable weights → regularization; prediction space |
| Affine space | Bias term `Wx+b`; SVM/classifier decision hyperplanes |
| Norms | `ℓ₁` lasso (sparsity), `ℓ₂` ridge, loss functions, grad clipping |
| Inner product / SPD | Kernels (SVM, GP), covariance, Mahalanobis distance |
| Angle / cosine | Cosine similarity in embedding/text retrieval (RAG) |
| Orthonormal / orthogonal matrix | Numerical stability, QR/SVD, PCA axes, orthogonal init |
| Orthogonal projection | **Least squares = projection**; PCA = projection onto top subspace |
| Gram–Schmidt / QR | Stable regression, feature whitening (Lab 2) |

## Cheat-sheet (exam-ready)
- **Subspace test:** contains `0`, closed under `+` and scaling. Every subspace of `ℝⁿ` = null space of some `A`.
- **Independence test:** columns → RREF; all pivots ⟺ independent. `rk = #pivots`, and **row rank = column rank**.
- **Basis** = minimal spanning = maximal independent set; gives **unique** coordinates; `dim` = #basis vectors.
- **Rank–Nullity:** `rank + nullity = n`. `A` invertible ⟺ full rank.
- **Linear map ⟺ matrix**; coordinates transform by `ŷ = A_Φ x̂`; **a basis is a coordinate system**.
- **Norm** = length (`ℓ₁,ℓ₂,ℓ∞`); **inner product** creates length+angle; `‖x‖=√⟨x,x⟩`.
- **cos ω = ⟨x,y⟩/(‖x‖‖y‖)**; orthogonal ⟺ `⟨x,y⟩=0`. **Cosine similarity** in ML.
- **Orthogonal matrix:** `QᵀQ=I`, `Q⁻¹=Qᵀ`, preserves length/angle (rotation), condition number 1.
- **Projection matrix:** onto line `P=bbᵀ/bᵀb`; onto subspace `P=B(BᵀB)⁻¹Bᵀ`; always `P²=P`, symmetric. **Least squares = projection**, residual ⟂ features.
- **Gram–Schmidt:** subtract projections, normalize → ONB = QR (Lab 2).

**Mini-glossary recap**
- **vector space / subspace** — a playground closed under `+` and scaling / a flat sheet through `0` inside it.
- **span / linear combination** — all reachable points / a scaled-and-added mix of vectors.
- **independent / dependent** — no vector redundant / at least one is a combo of the others.
- **basis / dimension** — minimal spanning axes / how many axes there are.
- **rank / nullity** — # independent columns / dimension of the kernel; `rank + nullity = n`.
- **linear map / image / kernel** — `Φ(x)=Ax` / the outputs it can produce / the inputs it sends to `0`.
- **affine space** — a subspace shifted off the origin (`ŷ = Wx + b`).
- **norm / inner product** — length `‖x‖` / the tool `⟨x,y⟩` that makes length & angle (dot product = `xᵀy`).
- **orthogonal / orthonormal / ONB** — perpendicular / perpendicular + unit length / an orthonormal basis.
- **orthogonal matrix** — `QᵀQ=I`, a pure rotation/reflection, `Q⁻¹=Qᵀ`.
- **projection** — the shadow onto a subspace = closest point there; error is perpendicular.
- **SPD matrix** — symmetric with `xᵀAx>0`; covariance & kernel matrices.
- **Gram–Schmidt** — turn skewed axes into clean orthonormal ones (Lab 2).

---

### Reproduce all outputs
Run `python _verify_ch2.py` (core examples) and `python _verify_workshop.py` (the §15 workshop problems) in this `Summaries` folder. Every numeric result above was generated and verified (NumPy 2.4).
