"""
Module 7 - Support Vector Machines - Worked Examples
Run:  python Module7_examples.py
Requires: numpy, scikit-learn
Every printed number matches the worked examples in the guide.
"""

import numpy as np


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


# ---------------------------------------------------------------
# 1. Margin from ||w||
# ---------------------------------------------------------------
def margin_demo():
    line("Margin = 2/||w||")
    for w in ([2, 0], [3, 4]):
        w = np.array(w, float)
        print(f"w={w.tolist()}: ||w||={np.linalg.norm(w):.3f}  margin={2/np.linalg.norm(w):.3f}")


# ---------------------------------------------------------------
# 2. Kernel values: polynomial and RBF (match the guide)
# ---------------------------------------------------------------
def poly_kernel(a, b, c=1, d=2):
    return (np.dot(a, b) + c) ** d


def rbf_kernel(a, b, gamma):
    return np.exp(-gamma * np.sum((np.array(a) - np.array(b)) ** 2))


def kernel_demo():
    line("Kernel values")
    print("poly (a.b+1)^2, a=(1,2) b=(3,1):", poly_kernel([1, 2], [3, 1]), "(expect 36)")
    print("RBF gamma=0.5, a=(1,2) b=(2,4):", round(rbf_kernel([1, 2], [2, 4], 0.5), 3), "(expect 0.082)")
    print("RBF gamma=1, (0,0)-(1,1):", round(rbf_kernel([0, 0], [1, 1], 1), 3), "(expect 0.135)")


# ---------------------------------------------------------------
# 3. Kernel trick identity: (a.b)^2 == phi(a).phi(b)
# ---------------------------------------------------------------
def kernel_trick_demo():
    line("Kernel trick: (a.b)^2 equals explicit degree-2 feature dot product")
    a, b = np.array([1.0, 2.0]), np.array([3.0, 1.0])
    phi = lambda x: np.array([x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2])
    print("explicit phi(a).phi(b) =", round(phi(a) @ phi(b), 3))
    print("kernel   (a.b)^2       =", round((a @ b) ** 2, 3), " -> identical")


# ---------------------------------------------------------------
# 4. Tiny hard-margin SVM: two points, verify margin & SVs
# ---------------------------------------------------------------
def tiny_svm_demo():
    line("Tiny SVM on x=-1 (y=-1) and x=+1 (y=+1)")
    from sklearn.svm import SVC
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0, 0], clf.intercept_[0]
    print(f"w={w:.3f}, b={b:.3f}, margin=2/|w|={2/abs(w):.3f}")
    print("support vectors:", clf.support_vectors_.ravel().tolist())
    print("predict x=0.3 ->", int(clf.predict([[0.3]])[0]))


# ---------------------------------------------------------------
# 5. Linear vs RBF SVM on moons + effect of C
# ---------------------------------------------------------------
def moons_demo():
    line("Linear vs RBF SVM on 'moons' + effect of C")
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for kernel in ("linear", "rbf"):
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, gamma="scale")).fit(Xtr, ytr)
        print(f"{kernel:6s} SVM: test acc = {clf.score(Xte, yte):.3f}")
    print("--- RBF, vary C ---")
    for C in (0.1, 1, 100):
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale")).fit(Xtr, ytr)
        n_sv = clf[-1].n_support_.sum()
        print(f"C={C:<5}: train={clf.score(Xtr,ytr):.3f} test={clf.score(Xte,yte):.3f} #SV={n_sv}")


if __name__ == "__main__":
    margin_demo()
    kernel_demo()
    kernel_trick_demo()
    tiny_svm_demo()
    moons_demo()
    print("\nAll Module 7 examples ran successfully. See diagrams in images/.")
