import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sec(t): print("\n"+"="*10+" "+t+" "+"="*10)

# 1. Determinant & trace
sec("1. Determinant & trace")
print("det[[1,2],[3,4]]   =", round(np.linalg.det(np.array([[1,2],[3,4]],float))))       # -2
A3 = np.array([[1,2,3],[4,5,6],[7,8,10]], float)
print("det 3x3            =", round(np.linalg.det(A3)))                                    # -3
T = np.array([[2,7,1],[0,3,5],[0,0,4]], float)
print("det triangular     =", round(np.linalg.det(T)), "= 2*3*4 =", 2*3*4)                 # 24
print("trace[[1,2],[3,4]] =", np.trace(np.array([[1,2],[3,4]])))                           # 5

# 2. Eigenvalues / eigenvectors
sec("2. Eigenvalues / eigenvectors")
A = np.array([[2,1],[1,2]], float)
w,v = np.linalg.eig(A)
print("A=[[2,1],[1,2]] eigenvalues =", np.round(w))                                        # [3,1]
print("eigenvectors (cols):\n", np.round(v,4))
B = np.array([[4,1],[2,3]], float)
wB,vB = np.linalg.eig(B)
print("B=[[4,1],[2,3]] eigenvalues =", np.round(np.sort(wB)))                              # [2,5]
print("check trace=sum eig:", np.trace(B), "=", round(wB.sum()),
      " det=prod eig:", round(np.linalg.det(B)), "=", round(wB.prod()))

# 3. Eigendecomposition & diagonalization + matrix powers (Lab 3)
sec("3. Diagonalization A=PDP^-1 and powers (Lab 3)")
P = np.array([[1,1],[1,-1]], float)          # eigenvectors of [[2,1],[1,2]]
D = np.diag([3.,1.])
recon = P@D@np.linalg.inv(P)
print("P D P^-1 =\n", recon, " == A:", np.allclose(recon, A))
A10_fast = P@np.diag([3.**10,1.**10])@np.linalg.inv(P)
print("A^10 via eig == np power:", np.allclose(A10_fast, np.linalg.matrix_power(A.astype(int),10)))
print("A^10[0,0] =", round(A10_fast[0,0]))                                                 # 29525

# 4. Cholesky (SPD = L L^T)
sec("4. Cholesky decomposition")
S2 = np.array([[4,2],[2,5]], float)
L2 = np.linalg.cholesky(S2)
print("L (2x2) =\n", L2, " LL^T==S:", np.allclose(L2@L2.T, S2))
S3 = np.array([[4,2,2],[2,5,3],[2,3,6]], float)
L3 = np.linalg.cholesky(S3)
print("L (3x3) =\n", L3, " LL^T==S:", np.allclose(L3@L3.T, S3))

# 5. SVD
sec("5. Singular Value Decomposition")
M = np.array([[1,0,1],[0,1,1]], float)
U,S,Vt = np.linalg.svd(M)
print("singular values =", np.round(S,4), " (sqrt3, 1) =", np.round([np.sqrt(3),1],4))
print("reconstruct == M:", np.allclose((U[:,:2]*S)@Vt[:2], M))
print("sigma^2 == eig(M M^T):", np.round(S**2,4), np.round(np.sort(np.linalg.eigvals(M@M.T))[::-1],4))

# 6. Low-rank approximation (Lab 5)
sec("6. Low-rank approximation (Eckart-Young)")
R = np.array([[5,5,0,0],
              [5,5,0,0],
              [0,0,4,4],
              [0,0,4,4]], float)      # block "ratings" matrix, rank 2
U,S,Vt = np.linalg.svd(R)
print("singular values =", np.round(S,4))
rank1 = S[0]*np.outer(U[:,0], Vt[0])
print("rank-1 approx =\n", np.round(rank1,4))
err1 = np.linalg.norm(R-rank1, 2)
print("spectral error of rank-1 = %.4f  == next sigma %.4f" % (err1, S[1]))

# 7. Power method (Lab 4): dominant eigenvector of [[2,1],[1,2]]
sec("7. Power method (Lab 4)")
A = np.array([[2,1],[1,2]], float)
x = np.array([1.,0.])
for k in range(20):
    x = A@x; x = x/np.linalg.norm(x)
lam = x@A@x
print("converged vector ~", np.round(x,4), " (=(1,1)/sqrt2=", np.round(np.array([1,1])/np.sqrt(2),4), ")")
print("dominant eigenvalue ~", round(lam,4))                                               # 3.0

print("\nALL OK")
