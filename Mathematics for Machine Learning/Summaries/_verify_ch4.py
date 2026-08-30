import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sec(t): print("\n"+"="*10+" "+t+" "+"="*10)

# finite-difference helpers
def _s(v): return np.asarray(v,float).ravel()[0]
def grad(f, x, h=1e-6):
    x=np.asarray(x,float); g=np.zeros_like(x)
    for i in range(x.size):
        e=np.zeros_like(x); e[i]=h
        g[i]=(_s(f(x+e))-_s(f(x-e)))/(2*h)
    return g
def jacobian(f, x, h=1e-6):
    x=np.asarray(x,float); f0=np.asarray(f(x),float); J=np.zeros((f0.size,x.size))
    for i in range(x.size):
        e=np.zeros_like(x); e[i]=h
        J[:,i]=(np.asarray(f(x+e))-np.asarray(f(x-e)))/(2*h)
    return J
def hessian(f, x, h=1e-4):
    x=np.asarray(x,float); n=x.size; H=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ei=np.zeros(n); ei[i]=h; ej=np.zeros(n); ej[j]=h
            H[i,j]=(_s(f(x+ei+ej))-_s(f(x+ei-ej))-_s(f(x-ei+ej))+_s(f(x-ei-ej)))/(4*h*h)
    return H

# 1. Univariate derivative
sec("1. Univariate derivative f(x)=x^3, f'=3x^2")
f=lambda x: x**3
print("numeric f'(2) =", round(grad(f,[2.])[0],4), " exact 3*2^2 =", 12)

# 2. Gradient of f(x,y)=x^2+3xy+y^2
sec("2. Gradient, f=x^2+3xy+y^2, grad=(2x+3y, 3x+2y)")
f=lambda v: v[0]**2+3*v[0]*v[1]+v[1]**2
print("numeric grad at (1,2) =", np.round(grad(f,[1,2]),4), " exact (8,7)")

# 3. Gradient of quadratic form x^T A x (symmetric) = 2 A x
sec("3. Gradient of x^T A x = 2Ax (A symmetric)")
A=np.array([[2,1],[1,3]],float)
f=lambda v: v@A@v
print("numeric grad at (1,1) =", np.round(grad(f,[1,1]),4), " exact 2A[1,1]=", 2*A@np.array([1,1.]))

# 4. Least-squares gradient  grad ||Ax-b||^2 = 2 A^T (Ax-b)
sec("4. Least-squares gradient 2A^T(Ax-b)")
Am=np.array([[1,1],[1,2],[1,3]],float); b=np.array([1,2,2.])
f=lambda x: np.sum((Am@x-b)**2)
print("numeric grad at (0,0) =", np.round(grad(f,[0,0]),4), " exact -2A^T b =", -2*Am.T@b)

# 5. Jacobian of f(x,y)=(x^2 y, x+y)
sec("5. Jacobian of vector function")
F=lambda v: np.array([v[0]**2*v[1], v[0]+v[1]])
print("numeric J at (1,2) =\n", np.round(jacobian(F,[1,2]),4), "\n exact [[4,1],[1,1]]")

# 6. Chain rule h(x)=sin(x^2+1); h'=cos(x^2+1)*2x
sec("6. Chain rule h(x)=sin(x^2+1)")
h=lambda x: np.sin(x[0]**2+1)
print("numeric h'(1) =", round(grad(h,[1.])[0],4), " exact 2cos(2) =", round(2*np.cos(2),4))

# 7. Hessian of f=x^2 y  -> [[2y,2x],[2x,0]]
sec("7. Hessian of f=x^2 y")
f=lambda v: v[0]**2*v[1]
print("numeric H at (1,2) =\n", np.round(hessian(f,[1,2]),3), "\n exact [[4,2],[2,0]]")

# 8. Taylor series
sec("8. Taylor series")
x=0.1
print("e^x ~ 1+x+x^2/2 =", 1+x+x**2/2, " actual e^0.1 =", round(np.exp(0.1),6),
      " err =", round(abs(np.exp(0.1)-(1+x+x**2/2)),6))
# multivariate 2nd-order Taylor is EXACT for a quadratic
f=lambda v: v[0]**2+3*v[0]*v[1]+v[1]**2
H=np.array([[2,3],[3,2]],float); d=np.array([1,2.])
print("2nd-order Taylor of quadratic at (1,2) =", 0.5*d@H@d, " actual f(1,2) =", f([1,2]))

# 9. Backprop through one sigmoid neuron: L=(sigma(wx+b)-y)^2
sec("9. Backprop (one neuron) — chain rule in action")
def sigmoid(z): return 1/(1+np.exp(-z))
x,w,b,y = 1.0, 0.5, 0.0, 0.0
z = w*x+b; a = sigmoid(z); L=(a-y)**2
print("forward: z=%.4f a=%.5f L=%.5f" % (z,a,L))
dL_da = 2*(a-y)
da_dz = a*(1-a)
dL_dz = dL_da*da_dz
dL_dw = dL_dz*x; dL_db = dL_dz*1; dL_dx = dL_dz*w
print("backward: dL/dw=%.5f dL/db=%.5f dL/dx=%.5f" % (dL_dw,dL_db,dL_dx))
Lf = lambda p: (sigmoid(p[0]*x+p[1])-y)**2      # p=(w,b)
print("numeric   dL/dw,dL/db =", np.round(grad(Lf,[w,b]),5))

print("\nALL OK")
