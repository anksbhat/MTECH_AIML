import numpy as np
np.set_printoptions(precision=4, suppress=True)

def sec(t): print("\n" + "="*8 + " " + t + " " + "="*8)

# ---- 1. Solve a 3x3 unique system ----
sec("1. Unique solution 3x3")
A = np.array([[1.,1.,1.],
              [1.,-1.,2.],
              [0.,1.,1.]])
b = np.array([3.,2.,2.])
x = np.linalg.solve(A, b)
print("x =", x)
print("check A@x =", A @ x)

# ---- 2. Three outcomes ----
sec("2. No / infinite via rank")
# No solution: (1)+(2) contradicts (3)
A_no = np.array([[1,1,1],[1,-1,2],[2,0,3]], float)
b_no = np.array([3,2,1], float)
aug = np.column_stack([A_no, b_no])
print("rank A =", np.linalg.matrix_rank(A_no), " rank [A|b] =", np.linalg.matrix_rank(aug))
# Infinite: (1)+(2)=(3)
b_inf = np.array([3,2,5], float)
aug2 = np.column_stack([A_no, b_inf])
print("infinite: rank A =", np.linalg.matrix_rank(A_no), " rank [A|b] =", np.linalg.matrix_rank(aug2), " n =", A_no.shape[1])

# ---- 3. Matrix multiply vs Hadamard ----
sec("3. Matmul vs Hadamard")
A = np.array([[1,2,3],[3,2,1]])
B = np.array([[0,2],[1,-1],[0,1]])
print("AB =\n", A @ B)
print("BA =\n", B @ A)
H = np.array([[1,2],[3,4]]) * np.array([[10,20],[30,40]])
print("Hadamard =\n", H)

# ---- 4. Inverse 3x3 ----
sec("4. Inverse")
A = np.array([[1.,2.,1.],[4.,4.,5.],[6.,7.,7.]])
Ainv = np.linalg.inv(A)
print("A^-1 =\n", Ainv)
print("A@A^-1 =\n", (A @ Ainv))

# ---- 5. Gaussian elimination from scratch ----
sec("5. Gaussian elimination (RREF) from scratch")
def rref(M):
    M = M.astype(float).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if abs(M[i, c]) > 1e-12:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = M[r] / M[r, c]
        for i in range(rows):
            if i != r:
                M[i] = M[i] - M[i, c] * M[r]
        r += 1
        if r == rows:
            break
    return M

A = np.array([[1.,0.,2.,0.],
              [1.,1.,0.,0.],
              [1.,2.,0.,1.],
              [1.,1.,1.,1.]])
I = np.eye(4)
aug = np.column_stack([A, I])
R = rref(aug)
print("RREF[A|I] right block (=A^-1) =\n", R[:, 4:])
print("matches np.linalg.inv:", np.allclose(R[:,4:], np.linalg.inv(A)))

# ---- 6. Particular + general solution / null space ----
sec("6. Null space via SVD")
A = np.array([[1,3,0,0,3],
              [0,0,1,0,9],
              [0,0,0,1,-4]], float)
# null space basis
u,s,vt = np.linalg.svd(A)
ns = vt[np.sum(s>1e-10):].T
print("nullspace dim =", ns.shape[1])
print("A @ nullspace ~ 0:\n", A @ ns)

# ---- 7. Least squares / normal equations / pseudoinverse ----
sec("7. Least squares regression")
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
true_w = np.array([2., -1., 0.5])
y = X @ true_w + 0.1*rng.normal(size=100)
# normal equations
w_ne = np.linalg.inv(X.T @ X) @ X.T @ y
w_pinv = np.linalg.pinv(X) @ y
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
print("true    :", true_w)
print("normal  :", w_ne)
print("pinv    :", w_pinv)
print("lstsq   :", w_lstsq)

# ---- 8. Condition number (lab from handout) ----
sec("8. Condition number & solution accuracy")
def experiment(A, label):
    A = np.array(A, float)
    x_true = np.ones(A.shape[1])
    b = A @ x_true
    b_pert = b + 1e-6*np.array([1,-1] if len(b)==2 else np.resize([1,-1],len(b)))
    x_pert = np.linalg.solve(A, b_pert)
    print(f"{label}: cond={np.linalg.cond(A):.2e}  ||dx||={np.linalg.norm(x_pert-x_true):.3e}")
experiment([[2,1],[1,2]], "well-conditioned")
experiment([[1,1],[1,1.0001]], "ill-conditioned")

print("\nALL OK")
