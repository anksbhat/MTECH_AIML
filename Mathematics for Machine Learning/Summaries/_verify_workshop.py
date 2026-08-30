import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sec(t): print("\n"+"="*10+" "+t+" "+"="*10)

# ============================================================
# CHAPTER 1 WORKSHOP
# ============================================================

sec("CH1-P1  Recover hidden prices (unique solve)")
# 2c + 1m = 400 ; 1c + 3m = 350
A = np.array([[2,1],[1,3]], float); b = np.array([400,350], float)
x = np.linalg.solve(A, b)
print("cappuccino, muffin =", x)          # [170. 60.]
print("check A@x =", A@x)                  # [400. 350.]

sec("CH1-P2  Fit a price line by least squares")
size = np.array([1,2,3,4], float)          # 1000 sqft
price = np.array([2,4,5,4], float)         # lakhs
A = np.column_stack([np.ones_like(size), size])   # design matrix [1, x]
AtA = A.T@A; Atb = A.T@price
print("A^T A =\n", AtA)                    # [[4 10],[10 30]]
print("A^T y =", Atb)                      # [15 41]
w = np.linalg.solve(AtA, Atb)
print("w (intercept, slope) =", w)         # [2.  0.7]
pred = A@w
print("predictions =", pred)               # [2.7 3.4 4.1 4.8]
resid = price - pred
print("residuals =", resid)                # [-0.7 0.6 0.9 -0.8]
print("SSE =", round(resid@resid,4))       # 2.3
print("lstsq agrees:", np.allclose(w, np.linalg.lstsq(A, price, rcond=None)[0]))
print("residual orthogonal to columns (A^T r ~0):", np.round(A.T@resid,6))

sec("CH1-P3  Redundant features -> unstable / non-unique weights")
x1 = np.array([1,2,3,4], float)
# exact collinear feature x2 = 2*x1  => rank deficient
Xexact = np.column_stack([x1, 2*x1])
print("rank of [x, 2x] =", np.linalg.matrix_rank(Xexact), "of 2 cols -> redundant")
y = np.array([2,4,5,4], float)
w_min = np.linalg.lstsq(Xexact, y, rcond=None)[0]   # minimum-norm solution
print("min-norm w =", w_min, " preds =", np.round(Xexact@w_min,4))
# add null-space direction (2,-1): 2*x1 - 1*(2*x1) = 0 -> predictions unchanged
w_alt = w_min + np.array([2,-1.])
print("alt w      =", w_alt, " preds =", np.round(Xexact@w_alt,4), "(identical!)")
# near-collinear -> ill conditioned
Xnear = np.column_stack([x1, 2*x1 + 1e-3*np.array([1,-1,1,-1.])])
print("cond(X^T X) near-collinear = %.2e" % np.linalg.cond(Xnear.T@Xnear))
print("cond(X^T X) if x2 independent = %.2e" %
      np.linalg.cond(np.column_stack([x1, x1**2]).T @ np.column_stack([x1, x1**2])))

# ============================================================
# CHAPTER 2 WORKSHOP
# ============================================================

sec("CH2-P1  Mini search engine (cosine similarity)")
# vocab = [machine, learning, cooking]
q = np.array([1,1,0], float)               # query "machine learning"
docs = {"A: machine learning course": np.array([1,1,0.]),
        "B: cooking cooking recipe":  np.array([0,0,2.]),
        "C: machine machine learning":np.array([2,1,0.])}
def cos(u,v): return u@v/(np.linalg.norm(u)*np.linalg.norm(v))
for name,d in docs.items():
    print(f"  cos(q, {name:28s}) = {cos(q,d):.4f}")

sec("CH2-P2  k-NN: L1 vs L2 pick different neighbours")
q = np.array([0,0], float)
A = np.array([3,0], float)   # label RED
B = np.array([2,2], float)   # label BLUE
print("L2: d(q,A)=%.4f  d(q,B)=%.4f  -> nearest %s" %
      (np.linalg.norm(q-A), np.linalg.norm(q-B), "B(BLUE)" if np.linalg.norm(q-B)<np.linalg.norm(q-A) else "A(RED)"))
print("L1: d(q,A)=%.1f  d(q,B)=%.1f  -> nearest %s" %
      (np.linalg.norm(q-A,1), np.linalg.norm(q-B,1), "A(RED)" if np.linalg.norm(q-A,1)<np.linalg.norm(q-B,1) else "B(BLUE)"))

sec("CH2-P3  Project 2D data onto one direction (1D compression / PCA seed)")
u = np.array([1,1], float)/np.sqrt(2)      # unit direction
pts = np.array([[2,0],[0,2],[2,2]], float)
coords = pts@u                             # scalar coordinate along u
print("unit direction u =", np.round(u,4))
for p,c in zip(pts, coords):
    print(f"  point {p} -> coordinate {c:.4f} -> reconstructed {np.round(c*u,4)}")
print("captured variance along u =", round(np.var(coords),4),
      " vs total =", round(np.var(pts[:,0])+np.var(pts[:,1]),4))

print("\nALL OK")
