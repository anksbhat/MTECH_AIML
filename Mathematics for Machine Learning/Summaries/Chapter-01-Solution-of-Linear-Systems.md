# Chapter 1 — Solution of Linear Systems (Beginner-Friendly, In Depth)

**Course:** AMLSIZC416 — Mathematical Foundations for Machine Learning
**Module 1 / Session 1:** *Systems of linear equations, matrices, solving systems of linear equations.*
**Reading:** **T1** — Deisenroth, Faisal & Ong, *Mathematics for Machine Learning*, **§2.1, §2.2, §2.3**.
**Related lab (handout Lab 1):** *Solving `Ax = b` and studying how solution accuracy depends on the condition number of `A`* — covered in §10.

> ### 📖 How to read this document (please read once)
> I assume you are **new to both the math and ML**. So every topic is taught in layers:
> - **🧠 Plain English** — the idea in everyday words, often with a picture.
> - **(a) Concept & intuition** — what it really means and *why we care*.
> - **(b) Worked-by-hand math** — slow, step-by-step numbers you can redo with pen & paper.
> - **(c) Python / ML** — the same thing in code, plus where it shows up in Machine Learning.
>
> All Python outputs are **real** (verified with NumPy 2.4). Run `_verify_ch1.py` to reproduce them.
> Don't rush. If a symbol confuses you, check the **Jargon Buster** just below — it defines everything.

---

## 🔤 Jargon Buster (keep this open while reading)

| Symbol / term | Say it as | Plain meaning |
|---|---|---|
| **scalar** | "scale-er" | a single ordinary number, e.g. `3`, `-0.5`. |
| **vector** `x` | "vector x" | an ordered list of numbers, e.g. `(2, 5, 1)`. Think: one data point, or an arrow in space. |
| **`ℝ`** | "the reals" | the set of all real numbers (any decimal). |
| **`ℝⁿ`** | "R-n" | all vectors with `n` numbers. `ℝ²` = the flat plane, `ℝ³` = 3-D space. |
| **matrix** `A` | "matrix A" | a rectangular grid of numbers (rows × columns). A table. |
| **`ℝ^{m×n}`** | "R m-by-n" | all matrices with `m` rows and `n` columns. |
| **entry** `aᵢⱼ` | "a-i-j" | the number in row `i`, column `j` of matrix `A`. |
| **unknown** `xᵢ` | "x-i" | a number we are trying to find. |
| **coefficient** | — | the known multiplier in front of an unknown (the `a`'s). |
| **linear** | — | unknowns appear only to the **first power** — no `x²`, no `x·y`, no `sin x`. Just "number × unknown", added up. |
| **system** | — | several equations that must **all** be true at once. |
| **solution** | — | a set of values for the unknowns that satisfies every equation. |
| **`Ax = b`** | "A x equals b" | the compact matrix form of a whole linear system (defined in §5). |
| **rank** | — | the number of *genuinely different* equations/rows (no leftovers). Full meaning in Ch. 2. |
| **`Iₙ`** | "identity n" | the `n×n` matrix with `1`s on the diagonal, `0`s elsewhere — the "do nothing" matrix. |
| **`Aᵀ`** | "A transpose" | matrix `A` flipped so rows become columns. |
| **`A⁻¹`** | "A inverse" | the matrix that undoes `A` (like division for matrices). |
| **`‖x‖`** | "norm of x" | the length of vector `x` (defined fully in Ch. 2). |
| **`Σ`** | "sigma / sum" | "add all these up." |
| **`∈`** | "in / belongs to" | membership, e.g. `x ∈ ℝ³` = "x is a 3-number vector." |

> 🧠 **One-line mantra for the whole chapter:** *Machine learning is mostly "set up a big `Ax = b`, then solve it — carefully."*

---

## 🗺️ Mental map of this chapter

```
                        THE BIG QUESTION:   solve   A x = b
                                              │
        ┌─────────────────────────┬───────────┴────────────┬────────────────────────┐
        ▼                         ▼                        ▼                        ▼
  What IS a linear         How MANY solutions       HOW do we find them?      How TRUSTWORTHY
  system? (§2,§5)          exist? (§5)              (§8 Gaussian elim)        is the answer? (§10)
   equations → matrix       none / one / infinite    row-ops → staircase       condition number κ(A)
                                 (rank test)          → back-substitute
                                                            │
                                                            ▼
                                          ALL solutions = one particular answer
                                                 + the "null space" (§9)
                                                            │
                        Real ML data has NO exact solution (noisy, too many rows)
                                                            ▼
                                      LEAST SQUARES  ⇒  this is LINEAR REGRESSION (§10)
```

Keep glancing back at this map — every section is one box in it.

---

## Table of Contents
1. Why linear algebra is *the* language of ML
2. The building blocks: scalars, vectors, matrices (foundations)
3. Systems of linear equations (§2.1)
4. Matrices and their algebra (§2.2)
5. Inverse and transpose (§2.2)
6. Solving systems: Gaussian elimination (§2.3)
7. Particular + general solution, null space, minus-1 trick (§2.3)
8. Algorithms, least squares & the ML bridge (§2.3)
9. Condition number lab & numerical accuracy (Lab 1)
10. 🧩 Problem-Solving Workshop (data & ML scenarios)
11. Cheat-sheet + glossary recap

*(Section numbers below follow this list.)*

---

## 1. Why linear algebra is *the* language of ML

🧠 **Plain English:** A computer can't "see" a photo or "read" a sentence. It only stores **numbers**. So in ML we turn *everything* — images, text, users, houses — into **lists of numbers (vectors)** and **tables of numbers (matrices)**. Once data is numbers in a grid, "learning" becomes **arithmetic on grids**. Linear algebra is the rulebook for that arithmetic.

| ML thing | Linear-algebra thing |
|---|---|
| A dataset: `N` samples, each with `d` features | A matrix `X ∈ ℝ^{N×d}` (one **row** per sample) |
| One data sample (e.g. one house) | A vector `x ∈ ℝ^d` (its `d` feature numbers) |
| A dense neural-network layer | `y = Wx + b` (a matrix–vector product) |
| Linear-regression prediction | `ŷ = Xw` (a matrix–vector product) |
| "Train the model" (least squares) | Solve a linear system `AᵀA w = Aᵀy` |
| PCA, SVD, embeddings | Matrix decompositions (later chapters) |

So "solve `Ax = b`" and "multiply/invert matrices" are **not** abstract exercises — they are *literally what training and running a model does under the hood*.

---

## 2. The building blocks: scalars, vectors, matrices (foundations)

🧠 **Plain English:** Before equations, get comfortable with the three "nouns" of linear algebra. They only differ by **how many directions** they spread over.

```
 SCALAR              VECTOR (ℝ³)             MATRIX (ℝ^{2×3})
 a single number     a list of numbers        a grid of numbers
                                              
      7                 ┌ 2 ┐                  ┌ 1  0  4 ┐
                        │ 5 │                  └ 2 -1  3 ┘
                        └ 1 ┘
   0 directions       1 direction            2 directions
   (a point)          (a column)             (rows AND columns)
```

- **Scalar** = one number. Written lowercase, e.g. `λ` (lambda), `c`.
- **Vector** = an ordered list of `n` numbers → lives in `ℝⁿ`. We usually write it as a **column**:
  ```
  x = (2, 5, 1)ᵀ   means   ┌ 2 ┐
                           │ 5 │      (the ᵀ just tips a row on its side into a column)
                           └ 1 ┘
  ```
  Two things you can always do to vectors (this *defines* them):
  1. **Add** two vectors of the same length (add matching entries).
  2. **Scale** a vector by a scalar (multiply every entry).
  ```
  (2,5,1) + (0,1,3) = (2,6,4)          3·(2,5,1) = (6,15,3)
  ```
  > **Definition — vector (working definition):** any object you can *add* to its own kind and *scale* by a number, without leaving the set. Arrows, polynomials, and audio clips all qualify, but in ML we live in **`ℝⁿ`** because a list of `n` numbers maps 1-to-1 onto an array in computer memory.

- **Matrix** = a grid with `m` rows and `n` columns → lives in `ℝ^{m×n}`. Entry `aᵢⱼ` = row `i`, column `j`.
  ```
  A = ┌ 1  0  4 ┐   a₁₁=1  a₁₂=0  a₁₃=4     (row index first, column index second —
      └ 2 -1  3 ┘   a₂₁=2  a₂₂=-1 a₂₃=3      "RC", like 'Roman Catholic', row-column)
  ```
  A matrix with **one row** is a **row vector**; with **one column**, a **column vector**. So vectors are just skinny matrices.

> 🧠 **Mental picture:** a vector is *one point/arrow*; a matrix is either *a stack of data points* (each row a sample) **or** *a machine that transforms vectors* (§4). Same grid, two jobs.

---

## 3. Systems of Linear Equations (§2.1)

### 🧠 Plain English
Imagine you know some totals but not the individual pieces. "2 coffees + 1 tea cost ₹200; 1 coffee + 1 tea cost ₹130. What's each price?" Each sentence is a **linear equation**; solving them **together** is a **linear system**. ML does this at massive scale.

### (a) Concept & intuition
A **linear system** is several linear equations that share the same unknowns `x₁,…,xₙ`:

```
a₁₁x₁ + a₁₂x₂ + … + a₁ₙxₙ = b₁      ← equation 1
   ⋮                                   ⋮
aₘ₁x₁ + aₘ₂x₂ + … + aₘₙxₙ = bₘ      ← equation m
```

- The `a`'s are **coefficients** (known), the `x`'s are **unknowns** (wanted), the `b`'s are the **right-hand sides** (known).
- **"Linear"** means each unknown appears only to the **first power**, each just multiplied by a number and then added. `3x₁` ✓, `x₁ + x₂` ✓, but `x₁²` ✗, `x₁x₂` ✗, `sin(x₁)` ✗.
- A **solution** is a tuple `(x₁,…,xₙ)` that makes **every** equation true *at the same time*.

**Fundamental fact — there are only THREE possible outcomes:**

```
   (1) NO solution        (2) EXACTLY ONE          (3) INFINITELY MANY
   lines are parallel     lines cross at a point   lines lie on top of each other
        \   \                    \  /                     ╲╲
         \   \                     \/  ← the answer         ╲╲  ← every point on the
          \   \                    /\                        ╲╲    line is a solution
```

There is **never** "exactly 2" or "exactly 7" solutions. That's a deep gift of linearity — and it makes life predictable.

**Geometric picture:** each equation in 2 unknowns is a **line**; in 3 unknowns, a **plane**. The solution set is where they all **intersect**:
- lines crossing at one point → **unique**;
- identical lines / planes sharing a whole line → **infinite**;
- parallel lines / planes that never meet → **none**.

### (b) Worked-by-hand math — see all three cases
All three share the first two equations `x₁+x₂+x₃=3` and `x₁−x₂+2x₃=2`; only the **third** changes.

**Case A — Unique.** Third eq: `x₂+x₃=2`.
- Add eq1+eq2: `2x₁+3x₃=5`.
- From eq3: `x₂ = 2−x₃`. Put into eq1: `x₁+(2−x₃)+x₃ = 3 ⇒ x₁ = 1`.
- Then `2(1)+3x₃ = 5 ⇒ x₃ = 1 ⇒ x₂ = 1`.
- ✅ **Solution `(1,1,1)`, unique.**

**Case B — None.** Third eq: `2x₁+3x₃=1`.
- But eq1+eq2 already forced `2x₁+3x₃=5`. You can't have `=5` **and** `=1`. ❌ **Contradiction → no solution.**

**Case C — Infinite.** Third eq: `2x₁+3x₃=5`.
- Now eq3 is *exactly* eq1+eq2 — it adds **no new information** (it's **redundant**). Two real equations, three unknowns → one unknown is **free** to be anything.
- Let `x₃ = a` (a "free" knob). Then `x₁ = (5−3a)/2`, `x₂ = (1+a)/2`.
- ✅ **Solution set** `{ ((5−3a)/2, (1+a)/2, a) : a ∈ ℝ }` — infinitely many, one for each `a`.

> 🧠 **Takeaway:** "infinite solutions" doesn't mean *chaos* — it means the answers form a neat line/plane, parameterized by **free variables**. We'll name these carefully in §7.

### (c) Python / ML perspective
`Ax = b` is the compact matrix form of the whole system (defined next section). One crucial reading:

> **`Ax` is a *linear combination of the columns of `A`*, weighted by the entries of `x`.**
> i.e. `Ax = x₁·(col 1) + x₂·(col 2) + … + xₙ·(col n)`. Burn this in — it's the soul of the subject.

```python
import numpy as np
A = np.array([[1., 1., 1.],
              [1.,-1., 2.],
              [0., 1., 1.]])
b = np.array([3., 2., 2.])
x = np.linalg.solve(A, b)          # solves the unique case
print(x)                           # -> [1. 1. 1.]
print(A @ x)                       # -> [3. 2. 2.]  (plug back in: matches b ✓)
```

**Detecting *which* outcome you have, using rank** (this is the *Rouché–Capelli* rule; you'll fully define **rank** in Ch. 2 — for now: "number of genuinely independent equations"):

```
 unique   ⇔  rank(A) = rank([A|b]) = n        (enough independent equations, no contradiction)
 infinite ⇔  rank(A) = rank([A|b]) < n        (consistent, but too few equations → free vars)
 none     ⇔  rank(A) < rank([A|b])            (adding b created a new independent row = contradiction)
```
Here `[A|b]` (say "A augmented with b") just means: glue `b` on as an extra column.

```python
A = np.array([[1,1,1],[1,-1,2],[2,0,3]], float)
b_no  = np.array([3,2,1], float)   # Case B
b_inf = np.array([3,2,5], float)   # Case C
print(np.linalg.matrix_rank(A),
      np.linalg.matrix_rank(np.column_stack([A, b_no])))   # -> 2 3  => NO solution
print(np.linalg.matrix_rank(A),
      np.linalg.matrix_rank(np.column_stack([A, b_inf])))  # -> 2 2  (n=3) => INFINITE
```

> **ML tie-in.** Real training data almost never gives an exactly-solvable `Ax=b` — you have **more samples (rows) than weights (unknowns)** and the data is **noisy**. That "no exact solution" situation is *precisely* why ML uses **least squares** (§8) instead of exact solving.

---

## 4. Matrices and their algebra (§2.2)

### 🧠 Plain English
A matrix is a number-grid, but it wears **two hats**: (1) a tidy way to *store* a system or a dataset, and (2) a **machine that transforms vectors** (rotate/stretch them). This section is the "grammar" of how grids combine.

### (a) Concept
An **`(m,n)` matrix** `A ∈ ℝ^{m×n}` has `m` rows and `n` columns; entry `aᵢⱼ` sits at row `i`, col `j`. A `1×n` matrix is a **row vector**, an `m×1` a **column vector**.

### (b) Worked math — the operations

**Addition** (only for the **same shape**, done entry-by-entry):
```
┌1 2┐ + ┌5 6┐ = ┌1+5 2+6┐ = ┌ 6  8┐
└3 4┘   └7 8┘   └3+7 4+8┘   └10 12┘        (rule: (A+B)ᵢⱼ = aᵢⱼ + bᵢⱼ)
```

**Scalar multiplication** (scale every entry): `3·[[1,2],[3,4]] = [[3,6],[9,12]]`.

**Matrix multiplication** — *the* operation to truly understand. For `A ∈ ℝ^{m×n}` and `B ∈ ℝ^{n×k}`:
```
C = AB ∈ ℝ^{m×k},     cᵢⱼ = Σ_{l=1..n} a_{il} · b_{lj}
```
🧠 **In plain English:** to get the number in **row `i`, column `j`** of the answer, walk along **row `i` of A** and **down column `j` of B**, multiply the pairs, and add them up (a "dot product").

```
     column j of B
        │
        ▼
     ┌ b₁ⱼ ┐
     │ b₂ⱼ │
row i │ b₃ⱼ │
of A  └     ┘
[aᵢ₁ aᵢ₂ aᵢ₃] ──►  cᵢⱼ = aᵢ₁b₁ⱼ + aᵢ₂b₂ⱼ + aᵢ₃b₃ⱼ
```

Hand example, `A(2×3)` times `B(3×2)`:
```
A = [1 2 3]      B = [ 0  2]
    [3 2 1]          [ 1 -1]
                     [ 0  1]

c₁₁ = 1·0 + 2·1 + 3·0 = 2      c₁₂ = 1·2 + 2·(−1) + 3·1 = 3
c₂₁ = 3·0 + 2·1 + 1·0 = 2      c₂₂ = 3·2 + 2·(−1) + 1·1 = 5

AB = [2 3]
     [2 5]
```

**Three rules you must internalize:**
- 🔗 **Shapes must chain:** `(m×n)(n×k) = (m×k)`. The two **inner** numbers must match; the **outer** two survive.
  ```
  (2×3)·(3×2) → (2×2)   ✓ inner 3=3
  (2×3)·(2×3) → ✗ error  (inner 3≠2)
  ```
- 🔄 **Not commutative:** `AB ≠ BA` in general (often even different shapes!). Here `BA` would be `3×3` — a totally different object.
- ✋ **Not element-wise:** `cᵢⱼ ≠ aᵢⱼbᵢⱼ`. The entry-by-entry product is a *different*, simpler thing called the **Hadamard product** `A ⊙ B` (NumPy `*`) — also used a lot in ML (masks, gating, dropout), so don't confuse them.

Laws that **do** hold: associativity `(AB)C = A(BC)`, distributivity `(A+B)C = AC+BC`, and identity `IₘA = AIₙ = A`, where `Iₙ` (the **identity matrix**) has `1`s on the diagonal and `0`s elsewhere — the matrix that "does nothing."

### (c) Python / ML perspective
```python
A = np.array([[1,2,3],[3,2,1]])
B = np.array([[0,2],[1,-1],[0,1]])
print(A @ B)          # matrix product -> [[2 3],[2 5]]
print(B @ A)          # different shape -> 3x3, proves non-commutativity
# [[ 6  4  2]
#  [-2  0  2]
#  [ 3  2  1]]

# Hadamard (element-wise) vs matmul — a classic beginner bug:
P = np.array([[1,2],[3,4]]);  Q = np.array([[10,20],[30,40]])
print(P * Q)          # Hadamard -> [[10 40],[90 160]]     (NumPy '*')
print(P @ Q)          # matmul   -> [[ 70 100],[150 220]]  (NumPy '@')
```

> **ML tie-in — a neural-network layer is one matmul.** A dense layer computing `y = Wx + b` over a whole **batch** `X ∈ ℝ^{N×d}` is `Y = X Wᵀ + b`. That dreaded *"shapes (N,d) and (k,m) not aligned"* error is exactly the "inner dimensions must match" rule biting you.

---

## 5. Inverse and Transpose (§2.2)

### 🧠 Plain English
- **Transpose** = flip the grid over its diagonal (rows ↔ columns). Cheap, always possible.
- **Inverse** = the "undo" button. If `A` stretches/rotates space, `A⁻¹` puts it back. Like `÷` for matrices — but only *some* square matrices have one.

### (a) Concept
For a **square** matrix `A ∈ ℝ^{n×n}`, the **inverse** `A⁻¹` satisfies
```
A A⁻¹ = A⁻¹ A = Iₙ         (apply A then A⁻¹ = back where you started)
```
- If `A⁻¹` exists, `A` is **invertible** (also called **regular** or **nonsingular**).
- If it does **not** exist, `A` is **singular**.
- When it exists, `A⁻¹` is **unique**.

The **transpose** `Aᵀ` flips across the diagonal: `(Aᵀ)ᵢⱼ = aⱼᵢ`.
```
A = ┌1 2 3┐        Aᵀ = ┌1 4┐
    └4 5 6┘             │2 5│
                        └3 6┘   (row 1 of A becomes column 1 of Aᵀ)
```

### (b) Worked math

**2×2 inverse formula (memorize this one):** for `A = [[a,b],[c,d]]`,
```
A⁻¹ = 1/(ad − bc) · ┌ d  −b ┐        exists  ⇔  ad − bc ≠ 0
                    └−c   a ┘
```
The number `ad − bc` is the **determinant** `det(A)` (studied properly in Ch. 3). If `det = 0`, `A` is **singular** — its columns are redundant, it "squashes space flat," and squashing can't be undone (no inverse).

Example: `A=[[1,2],[3,4]]`, `det = 1·4 − 2·3 = −2 ≠ 0`, so
```
A⁻¹ = (−1/2)·[[4,−2],[−3,1]] = [[−2, 1],[1.5, −0.5]]
```
Check: `A·A⁻¹ = I` (try it by hand — great practice!).

**Key identities — notice the ORDER FLIPS:**
```
(AB)⁻¹ = B⁻¹A⁻¹          (AB)ᵀ = BᵀAᵀ
(Aᵀ)ᵀ = A               (A+B)ᵀ = Aᵀ + Bᵀ
```
🧠 Think of putting on socks then shoes: to undo, remove **shoes first, then socks** — reverse order. That's why `(AB)⁻¹ = B⁻¹A⁻¹`.
⚠️ **Warning:** `(A+B)⁻¹ ≠ A⁻¹ + B⁻¹`. Do **not** "distribute" an inverse over a sum.

> **Definition — symmetric matrix:** `A = Aᵀ` (it equals its own flip; mirror-image across the diagonal). Symmetric matrices are everywhere in ML — **covariance matrices**, **kernel/Gram matrices** `XXᵀ`, and **Hessians** are all symmetric. Sums of symmetric matrices stay symmetric; products usually don't.

### (c) Python / ML perspective
Textbook Example 2.4 (a 3×3 inverse), verified:
```python
A = np.array([[1.,2.,1.],[4.,4.,5.],[6.,7.,7.]])
print(np.linalg.inv(A))
# [[-7. -7.  6.]
#  [ 2.  1. -1.]
#  [ 4.  5. -4.]]
print(A @ np.linalg.inv(A))   # ~ identity (tiny float noise off the diagonal)
```

> **ML tie-in & a performance warning.** The closed-form linear-regression solution is `w = (XᵀX)⁻¹Xᵀy`. Neat on paper — but in code you should **almost never call `inv()`**: it's slower and numerically shakier than solving directly (`np.linalg.solve`) or using `lstsq`/`pinv`. **Rule of thumb:** write `inv` in math, use `solve` in code (see §8–§9).

---

## 6. Solving systems: Gaussian elimination (§2.3)

### 🧠 Plain English
For big systems you can't just "eyeball" the answer. **Gaussian elimination** is a fixed recipe: use simple row moves to carve the equations into a **staircase (triangle) shape**, then read the answers off from the bottom up. It's the same tidy method your school taught for 2–3 equations, made systematic.

### (a) Concept
We simplify the system using **elementary row operations** — moves that **never change the solution set**:
1. **Swap** two rows.
2. **Scale** a row by a nonzero constant.
3. **Add a multiple** of one row to another.

We apply these to the **augmented matrix** `[A | b]` (coefficients, then a bar, then the right-hand sides) until we reach a staircase.

> **Definition — Row-Echelon Form (REF):** all-zero rows sit at the bottom, and each row's **pivot** (its first nonzero number from the left) is strictly **to the right** of the pivot in the row above → a downward "staircase."
>
> **Definition — Reduced REF (RREF):** additionally every pivot equals `1` and is the **only** nonzero number in its column.

```
   REF (staircase)              RREF (cleaned up)
  ┌ ■ * * * ┐                  ┌ 1 0 * 0 ┐     ■ = pivot (first nonzero)
  │ 0 ■ * * │                  │ 0 1 * 0 │     * = anything
  │ 0 0 0 ■ │                  │ 0 0 0 1 │     each pivot alone in its column
  └ 0 0 0 0 ┘                  └ 0 0 0 0 ┘
```

- **Pivot columns → "basic" variables** (determined).
- **Non-pivot columns → "free" variables** (the knobs that give infinite solutions, §7).

### (b) Worked math — a full elimination
Solve
```
 x₁ +  x₂ +  x₃ = 6
2x₁ +  x₂ −  x₃ = 1
3x₁ − 2x₂ +  x₃ = 4
```
Write `[A|b]`, then eliminate below each pivot:
```
[1  1  1 | 6]
[2  1 -1 | 1]
[3 -2  1 | 4]

R2 ← R2 − 2·R1,   R3 ← R3 − 3·R1   (kill the 2 and 3 under the first pivot):
[1  1  1 |  6]
[0 -1 -3 |-11]
[0 -5 -2 |-14]

R3 ← R3 − 5·R2   (kill the −5 under the second pivot):
[1  1  1 |  6]
[0 -1 -3 |-11]
[0  0 13 | 41]      ← now it's a clean staircase (triangle)
```
**Back-substitute** from the bottom row up:
- Row 3: `13·x₃ = 41 ⇒ x₃ = 41/13`.
- Row 2: `−x₂ − 3x₃ = −11 ⇒ x₂ = 11 − 3·(41/13) = (143 − 123)/13 = 20/13`.
- Row 1: `x₁ = 6 − x₂ − x₃ = 6 − 20/13 − 41/13 = (78 − 61)/13 = 17/13`.

✅ **Solution `(17/13, 20/13, 41/13)`.** Three pivots (cols 1,2,3), no free variables → **unique**. (Fractions are fine — the *method* is the lesson, not round numbers.)

### (c) Python — build it yourself, then trust the library
```python
def rref(M):
    M = M.astype(float).copy(); rows, cols = M.shape; r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if abs(M[i, c]) > 1e-12), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]      # 1) swap a nonzero pivot into place
        M[r] = M[r] / M[r, c]          # 2) scale pivot row so pivot = 1
        for i in range(rows):          # 3) clear the rest of the column
            if i != r:
                M[i] -= M[i, c] * M[r]
        r += 1
        if r == rows: break
    return M
```

**Bonus — finding an inverse via `[A | I] ⇝ [I | A⁻¹]`** (textbook Example 2.9), verified:
```python
A = np.array([[1.,0.,2.,0.],[1.,1.,0.,0.],[1.,2.,0.,1.],[1.,1.,1.,1.]])
R = rref(np.column_stack([A, np.eye(4)]))
print(R[:, 4:])                       # the inverse appears on the right:
# [[-1.  2. -2.  2.]
#  [ 1. -1.  2. -2.]
#  [ 1. -1.  1. -1.]
#  [-1.  0. -1.  2.]]
print(np.allclose(R[:, 4:], np.linalg.inv(A)))   # -> True
```
🧠 **Insight:** finding an inverse is just **solving `n` systems at once** — turn `A` into `I` on the left, and whatever lands on the right is `A⁻¹`.

---

## 7. Particular + general solution, null space, minus-1 trick (§2.3)

### 🧠 Plain English
When a system has *infinitely many* answers, they aren't random. Every answer = **one specific answer** you happened to find **+ any amount of "wiggle" that doesn't affect `b`**. The set of harmless wiggles is called the **null space**. Knowing the null space = knowing *all* answers at once.

### (a) Concept — the structure of *all* solutions
For a solvable system, **every** solution can be written as
```
x  =  x_particular  +  x_homogeneous
      (one answer      (any solution of
       that works)      A x = 0)
```
**Why?** If `A·x_p = b` (a real answer) and `A·x_h = 0` (a "harmless" vector), then
`A(x_p + x_h) = b + 0 = b`. So adding *anything the matrix sends to zero* keeps you a valid solution.

> **Definition — null space (kernel):** `null(A) = { x : Ax = 0 }`, the set of all vectors that `A` crushes to zero. Its dimension = the number of **free variables**.

**Three-step recipe:** (1) find **one** `x_particular`; (2) find the **whole** null space of `A`; (3) combine: `x = x_p + λ₁n₁ + λ₂n₂ + …`.

```
   x_particular ●
                 \        every solution = start at x_p,
                  \       then slide freely along the null-space arrows n₁, n₂ …
      n₁ ↖         ●───────────────►  n₂
                (a whole line/plane of equally-valid answers)
```

### (b) Worked math — the "minus-1 trick" (a fast way to read off the null space)
Start from an RREF matrix (pivots are in columns 1, 3, 4):
```
A = [1  3  0  0  3]
    [0  0  1  0  9]
    [0  0  0  1 -4]
```
We want all `x` with `Ax = 0`. The **free** variables are in the **non-pivot** columns (2 and 5). The trick: pad `A` into a **square** matrix `Ã` by inserting, at each **missing pivot** position, a row that has `−1` on the diagonal:
```
Ã = [ 1  3  0  0  3]
    [ 0 -1  0  0  0]   ← inserted row for missing pivot in column 2
    [ 0  0  1  0  9]
    [ 0  0  0  1 -4]
    [ 0  0  0  0 -1]   ← inserted row for missing pivot in column 5
```
Now the **columns of `Ã` that contain those `−1`s** (columns 2 and 5) are a **basis of the null space**:
```
n₁ = ( 3, -1,  0,  0,  0)ᵀ          n₂ = ( 3,  0,  9, -4, -1)ᵀ
```
**General solution of `Ax = 0`:** `λ₁·n₁ + λ₂·n₂` for any real `λ₁, λ₂`. (Two free variables → a 2-D sheet of solutions.)

### (c) Python — null space via SVD (the numerically robust way)
```python
A = np.array([[1,3,0,0,3],
              [0,0,1,0,9],
              [0,0,0,1,-4]], float)
u, s, vt = np.linalg.svd(A)
null_basis = vt[np.sum(s > 1e-10):].T     # rows of Vᵀ past the rank span the null space
print(null_basis.shape[1])                # -> 2   (two free vars, matches the trick)
print(A @ null_basis)                     # -> ~0 columns, confirming Ax = 0
```
The two vectors NumPy returns span the **same** null space as `n₁, n₂` (possibly rotated/rescaled — any basis of the same subspace is equally correct).

> **ML tie-in.** The null space is exactly the set of **directions in weight space that don't change your predictions** (`Xw` is unchanged when `w` slides along `null(X)`). A non-trivial null space ⇒ the weights are **not uniquely identifiable** ⇒ infinitely many equally-good models ⇒ the very practical reason we add **regularization** (ridge / L2) to pin down one stable answer.

---

## 8. Algorithms, least squares & the ML bridge (§2.3)

### 🧠 Plain English
In the real world you usually have **more equations than unknowns** and **noisy** numbers, so *no* `x` fits perfectly. Instead of demanding perfection, we find the `x` that's **as close as possible** — the smallest total error. That "closest fit" is **least squares**, and it *is* linear regression.

Practical menu for `Ax = b`:

| Situation | Method | Formula |
|---|---|---|
| `A` square & invertible | direct solve (preferred) | `x = solve(A, b)` (conceptually `A⁻¹b`) |
| `A` **tall** (more rows than columns), full column rank — **the ML case** | **normal equations / least squares** | `x = (AᵀA)⁻¹Aᵀb` |
| general / rank-deficient | **pseudoinverse** | `x = A⁺ b` |
| very large / sparse | iterative (Jacobi, Gauss–Seidel, conjugate gradient) | refine `x` until "close enough" |

**Why least squares?** With more equations than unknowns (**over-determined**) and noise, `b` is unreachable exactly. So we minimize the squared error `‖Ax − b‖²` (the summed, squared gaps). Setting its derivative to zero gives the **normal equations** `AᵀA x = Aᵀb`, hence
```
x = (AᵀA)⁻¹ Aᵀ b .
```
The operator `A⁺ = (AᵀA)⁻¹Aᵀ` is the **Moore–Penrose pseudoinverse** — and this is *literally* the closed-form solution of **linear regression** (T1 Chapter 9).

> **Definition — least squares:** the method of choosing unknowns to make the **sum of squared errors** as small as possible. "Squared" so that positive and negative gaps can't cancel, and so big misses are punished more.

### Python — linear regression, four equivalent ways (verified)
```python
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))              # 100 samples, 3 features (a TALL matrix)
true_w = np.array([2., -1., 0.5])
y = X @ true_w + 0.1*rng.normal(size=100)  # true linear signal + a little noise

w_normal = np.linalg.inv(X.T @ X) @ X.T @ y      # normal equations (the theory form)
w_pinv   = np.linalg.pinv(X) @ y                 # pseudoinverse
w_lstsq  = np.linalg.lstsq(X, y, rcond=None)[0]  # recommended solver

print(true_w)     # [ 2.  -1.   0.5 ]
print(w_normal)   # [ 1.9988 -1.0132  0.4843 ]
print(w_pinv)     # [ 1.9988 -1.0132  0.4843 ]
print(w_lstsq)    # [ 1.9988 -1.0132  0.4843 ]
```
All three recover `true_w` up to the noise floor, and they agree — proving normal equations, pseudoinverse, and `lstsq` are the *same math*. **In production use `lstsq`/`pinv`, not `inv`** (stability — see §9).

---

## 9. Condition number lab & numerical accuracy (Handout Lab 1)

### 🧠 Plain English
Some systems are "touchy": a hair's-width change in the data flings the answer wildly. The **condition number** `κ(A)` is a single number that measures this touchiness. Small `κ` = calm & trustworthy; huge `κ` = fragile & unreliable. This is *the* Lab 1 experiment.

### Concept
> **Definition — condition number:** `κ(A) = ‖A‖ · ‖A⁻¹‖`. It bounds how much `A` **amplifies input errors** into the solution:
> ```
> (relative error in x)  ≲  κ(A) × (relative error in b)
> ```
- `κ ≈ 1`: **well-conditioned** — tiny data changes → tiny solution changes.
- `κ ≫ 1`: **ill-conditioned** — tiny data/rounding changes → large, unreliable swings. Geometrically the two "lines" are **nearly parallel**, so where they cross is super-sensitive:
```
   well-conditioned              ill-conditioned
   (lines ~perpendicular)        (lines nearly parallel)
        \    |                      \   \
         \   |                       \   \     a tiny shift of one line
      ────●──────                 ────●╌╌╌╌╌╌   slides the crossing point
         /   |                       /   /      a LONG way ⇒ fragile answer
```

### Python experiment (this is exactly the handout's Lab 1)
```python
def experiment(A, label):
    A = np.array(A, float); x_true = np.ones(A.shape[1]); b = A @ x_true
    b_pert = b + 1e-6*np.resize([1,-1], len(b))   # nudge the RHS by a tiny 1e-6
    x_pert = np.linalg.solve(A, b_pert)
    print(f"{label}: cond={np.linalg.cond(A):.2e}  ||dx||={np.linalg.norm(x_pert-x_true):.3e}")

experiment([[2,1],[1,2]],        "well-conditioned")   # cond=3.00e+00  ||dx||=1.414e-06
experiment([[1,1],[1,1.0001]],   "ill-conditioned")    # cond=4.00e+04  ||dx||=2.828e-02
```
**Reading it:** a `1e-6` nudge to `b` moves the well-conditioned answer by `~1e-6`, but the ill-conditioned answer by `~3e-2` — about **20,000× more**, right in line with its condition number `~4e4`. That's exactly what Lab 1 asks you to observe: *solution accuracy tracks the condition number.*

> **ML tie-in.** When features are highly correlated, `XᵀX` becomes ill-conditioned, `(XᵀX)⁻¹` blows up, and regression weights turn huge and jittery. Fixes: **feature scaling/normalization**, dropping redundant features, or **ridge regression** (add `λI` so `XᵀX + λI` is well-conditioned). This links Chapter 1 straight to the optimization modules later.

---

## 10. 🧩 Problem-Solving Workshop (data & ML scenarios)

> **How to build intuition:** for each problem, follow the loop **① Read the story → ② Turn words into `A`, `x`, `b` → ③ Solve step-by-step → ④ Interpret in ML terms → ⑤ Verify in code.** Do them with pen & paper first, then check against Python.

### Problem 1 — Recovering hidden prices (a unique solve)
**Story.** A canteen won't share its menu prices. You only have two receipts:
- Receipt 1: **2 cappuccinos + 1 muffin = ₹400**
- Receipt 2: **1 cappuccino + 3 muffins = ₹350**
Find the price of each item.

**① → ② Set up.** Let `c` = cappuccino price, `m` = muffin price. Unknowns `x = (c, m)`:
```
2c + 1m = 400          A = [2 1]     x = [c]     b = [400]
1c + 3m = 350              [1 3]         [m]         [350]
```
**③ Solve by hand (substitution).** From eq1, `m = 400 − 2c`. Put into eq2:
`c + 3(400 − 2c) = 350 → c + 1200 − 6c = 350 → −5c = −850 → c = 170`. Then `m = 400 − 340 = 60`.
✅ **cappuccino = ₹170, muffin = ₹60.** (Check receipt 2: `170 + 180 = 350` ✓.)

**④ ML lens.** This *is* what "learning" does: you observe **sums of features** (receipts) and **infer the hidden per-feature weights** (prices). A neural net's job has the same shape — recover parameters that explain the observed data. Here we had exactly as many independent receipts as unknowns → **one unique answer** (`rank = 2 = #unknowns`).

**⑤ Python.**
```python
A = np.array([[2,1],[1,3]], float); b = np.array([400,350], float)
print(np.linalg.solve(A, b))     # -> [170.  60.]
print(A @ np.array([170,60.]))   # -> [400. 350.]  (matches the receipts)
```

---

### Problem 2 — Fitting a price line with least squares (the ML core)
**Story.** You have 4 apartments: sizes `1,2,3,4` (in 1000 sqft) renting for `2,4,5,4` (in ₹ lakhs). No straight line hits all 4 points (real data is noisy). Find the **best-fit line** `price = w₀ + w₁·size`.

**① → ② Set up.** This is **over-determined** (4 equations, 2 unknowns). Build the **design matrix** `A` (a column of 1s for the intercept, then the sizes) and target `y`:
```
A = [1 1]      y = [2]        model:  y ≈ A w,   w = (w₀, w₁)
    [1 2]          [4]
    [1 3]          [5]
    [1 4]          [4]
```
We can't solve `Aw = y` exactly, so we minimize the squared error `‖Aw − y‖²`. Setting the gradient to zero gives the **normal equations** `AᵀA w = Aᵀy`.

**③ Solve by hand.** Compute the small `2×2` system:
```
AᵀA = [4  10]      Aᵀy = [ 15 ]        (15 = 2+4+5+4,  41 = 1·2+2·4+3·5+4·4)
      [10 30]            [ 41 ]
```
So `4w₀ + 10w₁ = 15` and `10w₀ + 30w₁ = 41`. Multiply the first by `2.5`: `10w₀ + 25w₁ = 37.5`. Subtract from the second: `5w₁ = 3.5 → w₁ = 0.7`. Back-substitute: `4w₀ + 7 = 15 → w₀ = 2.0`.
✅ **Best-fit line: `price = 2.0 + 0.7·size`.**

Predictions vs actual (the model can't be perfect — that's the point):
```
size:        1     2     3     4
predicted:  2.7   3.4   4.1   4.8
actual:      2     4     5     4
residual:  −0.7  +0.6  +0.9  −0.8      (errors, sum ≈ 0)
```
Total squared error **SSE = 0.49+0.36+0.81+0.64 = 2.30.**

**④ ML lens.** You just trained a **linear regression** by hand. Key geometric fact (bridges to Ch. 2 projections): the residual vector is **orthogonal to every feature column** (`Aᵀr = 0`) — least squares places `Aw` as the *shadow* of `y` on the columns of `A`. Predict a new `2500 sqft` flat (size = 2.5): `2.0 + 0.7·2.5 = ₹3.75 lakh`.

**⑤ Python.**
```python
size  = np.array([1,2,3,4], float)
price = np.array([2,4,5,4], float)
A = np.column_stack([np.ones_like(size), size])       # design matrix [1, x]
w = np.linalg.solve(A.T@A, A.T@price)                 # normal equations
print(w)                                              # -> [2.  0.7]
print(A@w)                                            # -> [2.7 3.4 4.1 4.8]
print(np.round(A.T@(price - A@w), 6))                 # -> [0. 0.]  residual ⟂ features
print(np.allclose(w, np.linalg.lstsq(A, price, rcond=None)[0]))  # -> True
```

---

### Problem 3 — Why redundant features break a model (multicollinearity)
**Story.** You add two size columns by mistake: `x₁ = size in 1000 sqft` and `x₂ = size in 500 sqft` — but `x₂ = 2·x₁` exactly. You fit `price ≈ w₁x₁ + w₂x₂`. What goes wrong?

**① → ② Set up.** The feature matrix `X = [x₁ , 2x₁]`. Its two columns point in the **same direction** → `rank(X) = 1`, not 2. There are **infinitely many** weight pairs giving the *same* predictions.

**③ Reason it out.** Any move `w → w + (2, −1)` changes nothing, because `2·x₁ + (−1)·(2x₁) = 0`. The direction `(2, −1)` is the **null space** of `X` (Ch. 1 §7) — a "harmless wiggle." So the weights are **not identifiable**: the model can't decide how to split credit between two identical features.

**④ ML lens.** With *exactly* redundant features you get infinite solutions; with *nearly* redundant features (the realistic case) the system becomes **ill-conditioned** — weights explode and flip sign with tiny data changes (recall the condition number, §9). **Fixes:** drop one feature, or add **ridge/L2 regularization** to pick the smallest stable weights.

**⑤ Python (verified).**
```python
x1 = np.array([1,2,3,4], float); y = np.array([2,4,5,4], float)
X = np.column_stack([x1, 2*x1])                  # x2 = 2*x1 exactly
print(np.linalg.matrix_rank(X))                  # -> 1  (redundant: only 1 real direction)

w   = np.linalg.lstsq(X, y, rcond=None)[0]       # -> [0.2733 0.5467]  (min-norm pick)
w2  = w + np.array([2, -1.])                     # slide along the null direction
print(np.round(X@w, 4), np.round(X@w2, 4))       # identical predictions! [1.3667 2.7333 4.1 5.4667]

# near-collinear => ill-conditioned:
Xn = np.column_stack([x1, 2*x1 + 1e-3*np.array([1,-1,1,-1.])])
print("%.2e" % np.linalg.cond(Xn.T@Xn))          # -> 1.94e+08  (huge => unstable weights)
```
> 🧠 **Intuition:** two features that say the *same thing* give the model no way to weigh them — like asking "is it the metre or the centimetre that predicts price?" The math screams back "infinitely many answers / wildly unstable."

---

## 11. Cheat-sheet (exam-ready)

**Core facts**
- **Only three solution counts:** none / one / infinite. Detect via rank: unique iff `rank(A) = rank([A|b]) = n`.
- **`Ax` = a linear combination of the columns of `A`.** (The single most important sentence.)
- **Matmul:** row · column; `(m×n)(n×k) = (m×k)`; **not commutative**; **not element-wise** (that's Hadamard `⊙`).
- **Inverse** only for square, `det ≠ 0`. `(AB)⁻¹ = B⁻¹A⁻¹`, `(AB)ᵀ = BᵀAᵀ` (**order flips**). `(A+B)⁻¹ ≠ A⁻¹+B⁻¹`.
- **Gaussian elimination → RREF** solves systems, exposes free/basic variables, and gives the inverse via `[A|I] ⇝ [I|A⁻¹]`.
- **All solutions = particular + null space** (`x_p + Σλᵢnᵢ`); read the null space off with the **minus-1 trick**.
- **Over-determined & noisy (ML reality):** least squares `x = (AᵀA)⁻¹Aᵀb = A⁺b` = linear regression.
- **Code smell:** avoid `inv()`; prefer `np.linalg.solve` / `lstsq` / `pinv`.
- **Condition number** predicts error amplification; ill-conditioning ⇒ scale features / regularize.

**Mini-glossary recap**
- **scalar / vector / matrix** — one number / a list / a grid.
- **coefficient / unknown / RHS** — the `a`'s / the `x`'s / the `b`'s.
- **rank** — count of genuinely independent rows (Ch. 2 formalizes it).
- **augmented matrix `[A|b]`** — coefficients with `b` glued on as a column.
- **pivot** — first nonzero in a row after elimination; marks a basic variable.
- **free variable** — an unknown you can set freely; source of infinite solutions.
- **null space / kernel** — all `x` with `Ax = 0` (the "harmless wiggles").
- **inverse / transpose / identity** — the undo matrix / the flipped matrix / the do-nothing matrix.
- **least squares / pseudoinverse** — best approximate solution / the operator `(AᵀA)⁻¹Aᵀ` that computes it.
- **condition number `κ(A)`** — how much the system amplifies errors.

---

### ✅ Reproduce all outputs
Run `python _verify_ch1.py` (core examples) and `python _verify_workshop.py` (the §10 workshop problems) in this `Summaries` folder. Every numeric result above was generated and verified (NumPy 2.4).
