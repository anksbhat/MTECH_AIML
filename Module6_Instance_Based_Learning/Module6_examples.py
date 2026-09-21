"""
Module 6 - Instance-Based Learning - Worked Examples
Run:  python Module6_examples.py
Requires: numpy, scikit-learn
Every printed number matches the worked examples in the guide.
"""

import numpy as np


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


# ---------------------------------------------------------------
# 1. Euclidean distance + 1-NN
# ---------------------------------------------------------------
def distance_demo():
    line("Euclidean distance")
    a, b = np.array([1, 2]), np.array([4, 6])
    print(f"d({a.tolist()},{b.tolist()}) = {np.sqrt(((a-b)**2).sum()):.3f}  (expect 5)")


# ---------------------------------------------------------------
# 2. k-NN from scratch (classification + regression)
# ---------------------------------------------------------------
def knn_predict(X, y, xq, k=3, task="clf"):
    d = np.sqrt(((X - xq) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    if task == "clf":
        vals, counts = np.unique(y[idx], return_counts=True)
        return vals[counts.argmax()], d[idx], idx
    return y[idx].mean(), d[idx], idx


def knn_demo():
    line("k-NN classification, query (2,2)")
    X = np.array([[1, 1], [2, 3], [3, 3], [5, 5]])
    y = np.array(["+", "+", "-", "-"])
    for k in (1, 3):
        pred, dist, idx = knn_predict(X, y, np.array([2, 2]), k=k)
        print(f"k={k}: nearest={X[idx].tolist()} labels={y[idx].tolist()} -> predict {pred}")


# ---------------------------------------------------------------
# 3. Distance-weighted regression  (w = 1/d^2)
# ---------------------------------------------------------------
def weighted_demo():
    line("Distance-weighted regression (inverse-square)")
    vals = np.array([10.0, 20.0, 30.0])
    dist = np.array([1.0, 2.0, 5.0])
    w = 1.0 / dist ** 2
    yhat = (w * vals).sum() / w.sum()
    print("weights:", w.round(3), " -> yhat =", round(yhat, 2), " (expect 12.56)")


# ---------------------------------------------------------------
# 4. Gaussian RBF activation
# ---------------------------------------------------------------
def rbf_demo():
    line("Gaussian RBF activation phi(x)=exp(-||x-c||^2/(2 sigma^2))")
    c, sigma = np.array([2.0, 2.0]), 1.0
    for x in ([3, 2], [5, 2]):
        d2 = ((np.array(x) - c) ** 2).sum()
        phi = np.exp(-d2 / (2 * sigma ** 2))
        print(f"x={x}: ||x-c||^2={d2:.0f}  phi={phi:.3f}")


# ---------------------------------------------------------------
# 5. Locally Weighted Regression (weighted normal equation) per query
# ---------------------------------------------------------------
def lwr_demo():
    line("Locally Weighted Regression on a curve (tau bandwidth)")
    rng = np.random.default_rng(0)
    X = np.linspace(0, 2 * np.pi, 60)
    y = np.sin(X) + rng.normal(0, 0.1, X.size)
    Xb = np.c_[np.ones_like(X), X]                      # design matrix with bias

    def predict(xq, tau):
        w = np.exp(-((X - xq) ** 2) / (2 * tau ** 2))   # Gaussian weights
        W = np.diag(w)
        theta = np.linalg.pinv(Xb.T @ W @ Xb) @ Xb.T @ W @ y
        return np.array([1, xq]) @ theta

    for tau in (0.3, 1.0):
        preds = np.array([predict(xq, tau) for xq in X])
        rmse = np.sqrt(np.mean((preds - np.sin(X)) ** 2))
        print(f"tau={tau}: RMSE vs true sin = {rmse:.3f}")


# ---------------------------------------------------------------
# 6. sklearn: kNN, weighted kNN, and an RBF-feature linear model
# ---------------------------------------------------------------
def sklearn_demo():
    line("sklearn: kNN vs weighted kNN vs RBF-feature Ridge")
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for w in ("uniform", "distance"):
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights=w))
        clf.fit(Xtr, ytr)
        print(f"kNN(k=7, {w:8s}) test acc = {clf.score(Xte, yte):.3f}")

    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import RidgeClassifier
    rbf = make_pipeline(StandardScaler(),
                        RBFSampler(gamma=1.0, n_components=200, random_state=0),
                        RidgeClassifier())
    rbf.fit(Xtr, ytr)
    print("RBF-feature RidgeClassifier test acc =", round(rbf.score(Xte, yte), 3))


if __name__ == "__main__":
    distance_demo()
    knn_demo()
    weighted_demo()
    rbf_demo()
    lwr_demo()
    sklearn_demo()
    print("\nAll Module 6 examples ran successfully. See diagrams in images/.")
