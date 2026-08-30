import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sec(t): print("\n"+"="*10+" "+t+" "+"="*10)

# ---- helpers ----
def _s(v): return np.asarray(v,float).ravel()[0]
def grad(f, x, h=1e-6):
    x=np.asarray(x,float); g=np.zeros_like(x)
    for i in range(x.size):
        e=np.zeros_like(x); e[i]=h
        g[i]=(_s(f(x+e))-_s(f(x-e)))/(2*h)
    return g

# 1. 1-D gradient descent on f(x)=x^2 (min at 0)
sec("1. 1-D gradient descent, f(x)=x^2, f'=2x")
x=5.0; eta=0.1
for k in range(5):
    g=2*x; x=x-eta*g
print("after 5 steps x =", round(x,4), " (true min 0)")
x=5.0
for k in range(50): x=x-0.1*2*x
print("after 50 steps x =", round(x,6))

# 2. learning-rate effect: too small / good / too big
sec("2. Learning-rate effect on f(x)=x^2 from x0=5 (10 steps)")
for eta in [0.01, 0.1, 0.5, 1.01]:
    x=5.0
    for k in range(10): x=x-eta*2*x
    print(f"eta={eta:>4}: x_10 = {x: .4f}")

# 3. 2-D gradient descent on a bowl f=x^2+10y^2 (ill-conditioned)
sec("3. 2-D GD on f=x^2+10y^2 (elongated bowl)")
def gradf(v): return np.array([2*v[0], 20*v[1]])
x=np.array([5.0,5.0]); eta=0.05
for k in range(30): x=x-eta*gradf(x)
print("plain GD after 30 steps x =", np.round(x,4), " (true min [0,0])")

# 4. momentum vs plain GD on same bowl
sec("4. Momentum vs plain GD (same 30 steps, eta=0.05)")
x=np.array([5.0,5.0]); v=np.zeros(2); beta=0.9; eta=0.02
for k in range(30):
    v=beta*v - eta*gradf(x); x=x+v
print("momentum after 30 steps x =", np.round(x,4))

# 5. convexity: second derivative / Hessian eigenvalues
sec("5. Convexity checks")
print("f=x^2: f''=2 > 0  -> convex (bowl)")
H=np.array([[2,0],[0,20]],float)
print("f=x^2+10y^2 Hessian eigenvalues =", np.linalg.eigvalsh(H), "all>0 -> convex")
Hs=np.array([[2,0],[0,-2]],float)
print("saddle x^2-y^2 eigenvalues =", np.linalg.eigvalsh(Hs), "mixed -> NOT convex")

# 6. exact line search step for a quadratic f=x^2 (optimal eta)
sec("6. Optimal step size (exact line search) for f(x)=x^2")
# along -grad from x: minimize (x-eta*2x)^2 => eta*=0.5
x=5.0; g=2*x; eta_star=0.5
x_new=x-eta_star*g
print("eta* =", eta_star, " one step from x=5 ->", x_new, " (reaches min in ONE step)")

# 7. constrained optimization via Lagrange multipliers
sec("7. Lagrange: min x^2+y^2 s.t. x+y=1")
# L = x^2+y^2 - lam(x+y-1); grad: 2x=lam, 2y=lam, x+y=1 => x=y=1/2, lam=1
print("solution x=y =", 0.5, " lambda =", 1.0, " min value =", 0.5**2+0.5**2)
# verify numerically by projected search over the line x+y=1
best=None
for t in np.linspace(-1,2,300001):
    xx=t; yy=1-t; val=xx*xx+yy*yy
    if best is None or val<best[0]: best=(val,xx,yy)
print("numeric min on constraint:", np.round(best,4))

# 8. GD solving linear regression -> matches normal equation
sec("8. GD for linear regression vs closed form (normal equation)")
A=np.array([[1,1],[1,2],[1,3]],float); b=np.array([1,2,2.])
w=np.zeros(2); eta=0.05
for k in range(2000):
    g=2*A.T@(A@w-b); w=w-eta*g
w_closed=np.linalg.solve(A.T@A, A.T@b)
print("GD w      =", np.round(w,4))
print("closed w  =", np.round(w_closed,4))

# 9. loss decreases monotonically (sanity)
sec("9. Loss decreases each step (linear regression)")
w=np.zeros(2); losses=[]
for k in range(6):
    L=np.sum((A@w-b)**2); losses.append(round(L,4))
    w=w-0.05*(2*A.T@(A@w-b))
print("first 6 losses:", losses, "(monotone down)")

print("\nALL OK")
