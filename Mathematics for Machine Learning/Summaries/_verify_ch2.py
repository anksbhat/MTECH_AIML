import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sec(t): print("\n"+"="*8+" "+t+" "+"="*8)

# 1. Linear independence via rank (Example 2.14 & 2.15)
sec("1. Linear (in)dependence")
X = np.array([[1,-1,1],[2,1,2],[3,0,-1],[-4,2,1]], float)  # Ex 2.14 columns
print("rank =", np.linalg.matrix_rank(X), "cols =", X.shape[1], "-> independent" )
A = np.array([[1,-4,2,17],[-2,-2,3,-10],[1,0,-1,11],[-1,4,-3,1]], float)  # Ex2.15 exact
print("Ex2.15 rank =", np.linalg.matrix_rank(A), "of", A.shape[1], "-> dependent")
print("  x4 = -7x1-15x2-18x3 ?", np.allclose(A[:,3], -7*A[:,0]-15*A[:,1]-18*A[:,2]))

# 2. Basis & rank (Example 2.18)
sec("2. Rank")
A = np.array([[1,2,1],[-2,-3,1],[3,5,0]], float)
print("rank =", np.linalg.matrix_rank(A))

# 3. Linear transformations (Example 2.22)
sec("3. Linear transforms")
th=np.pi/4
R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
S=np.array([[2,0],[0,1]])
v=np.array([1,0])
print("rotate e1 by 45deg:", R@v)
print("stretch [1,1]:", S@np.array([1,1]))
print("det(R)=",round(np.linalg.det(R),4)," R preserves length:", np.allclose(np.linalg.norm(R@v),1))

# 4. Image & kernel (Example 2.25)
sec("4. Image & kernel / rank-nullity")
A=np.array([[1,2,-1,0],[1,0,0,1]],float)
r=np.linalg.matrix_rank(A)
u,s,vt=np.linalg.svd(A); ker=vt[np.sum(s>1e-10):].T
print("rank(Im)=",r,"  dim(ker)=",ker.shape[1],"  r+null=",r+ker.shape[1],"= n =",A.shape[1])
print("A@ker ~0:\n",A@ker)

# 5. Norms
sec("5. Norms")
x=np.array([3,-4.])
print("L1=",np.linalg.norm(x,1)," L2=",np.linalg.norm(x,2)," Linf=",np.linalg.norm(x,np.inf))

# 6. Angle between vectors (Example 3.6)
sec("6. Angle / cosine similarity")
x=np.array([1,1.]); y=np.array([1,2.])
cos=x@y/(np.linalg.norm(x)*np.linalg.norm(y))
print("cos =",round(cos,4)," (=3/sqrt10=",round(3/np.sqrt(10),4),")  angle deg=",round(np.degrees(np.arccos(cos)),2))

# 7. Orthogonal matrix preserves length & angle
sec("7. Orthogonal matrix")
Q=R  # rotation is orthogonal
print("Q^T Q = I:", np.allclose(Q.T@Q,np.eye(2)), " inv==T:", np.allclose(np.linalg.inv(Q),Q.T))

# 8. Orthonormal basis check
sec("8. ONB")
b1=np.array([1,1.])/np.sqrt(2); b2=np.array([1,-1.])/np.sqrt(2)
print("b1.b2=",round(b1@b2,4)," |b1|=",round(np.linalg.norm(b1),4)," |b2|=",round(np.linalg.norm(b2),4))

# 9. Gram-Schmidt (Lab 2) vs QR
sec("9. Gram-Schmidt (Lab 2)")
def gram_schmidt(V):
    V=V.astype(float); U=np.zeros_like(V)
    for i in range(V.shape[1]):
        w=V[:,i].copy()
        for j in range(i):
            w-= (U[:,j]@V[:,i])*U[:,j]
        U[:,i]=w/np.linalg.norm(w)
    return U
V=np.array([[1,1,0],[1,0,1],[0,1,1]],float)
Q=gram_schmidt(V)
print("Q^T Q = I:", np.allclose(Q.T@Q,np.eye(3)))
Qnp,_=np.linalg.qr(V)
print("matches numpy QR (up to sign):", np.allclose(np.abs(Q),np.abs(Qnp)))

# 10. Projection onto a line (Example 3.10)
sec("10. Projection onto a line")
b=np.array([1,2,2.]); P=np.outer(b,b)/(b@b)
print("P =\n",P)
x=np.array([1,1,1.]); print("proj =",P@x," (=1/9[5,10,10])")
print("idempotent P@P==P:", np.allclose(P@P,P))

# 11. Projection onto subspace (Example 3.11)
sec("11. Projection onto plane")
B=np.array([[1,0],[1,1],[1,2]],float); x=np.array([6,0,0.])
lam=np.linalg.solve(B.T@B, B.T@x)
Pi=B@np.linalg.inv(B.T@B)@B.T
print("lambda =",lam," proj =",Pi@x)

# 12. ML: least squares = projection of y onto column space of X
sec("12. Least squares = projection")
rng=np.random.default_rng(1)
Xd=rng.normal(size=(50,3)); w=np.array([1.,-2.,0.5]); y=Xd@w+0.05*rng.normal(size=50)
w_hat=np.linalg.lstsq(Xd,y,rcond=None)[0]
yhat=Xd@w_hat
resid=y-yhat
print("w_hat=",w_hat)
print("residual orthogonal to columns of X (X^T r ~0):",np.round(Xd.T@resid,6))

print("\nALL OK")
