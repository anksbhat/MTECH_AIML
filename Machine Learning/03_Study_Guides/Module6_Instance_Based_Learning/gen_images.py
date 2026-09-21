"""
Module 6 - generate diagrams referenced by the guide.
Run:  python gen_images.py
Requires: numpy, matplotlib, scikit-learn
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG, exist_ok=True)


def knn_regions():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=60, centers=3, cluster_std=1.2, random_state=3)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, (1, 15)):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="viridis", s=25)
        ax.set_title(f"k = {k}  ({'wiggly / high variance' if k==1 else 'smooth / high bias'})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("k-NN decision regions: effect of k")
    plt.tight_layout(); plt.savefig(f"{IMG}/m6_01_knn_regions.png", dpi=110); plt.close()


def lwr_plot():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 2 * np.pi, 80)
    y = np.sin(X) + rng.normal(0, 0.15, X.size)
    Xb = np.c_[np.ones_like(X), X]

    def predict(xq, tau):
        w = np.exp(-((X - xq) ** 2) / (2 * tau ** 2))
        W = np.diag(w)
        theta = np.linalg.pinv(Xb.T @ W @ Xb) @ Xb.T @ W @ y
        return np.array([1, xq]) @ theta

    grid = np.linspace(0, 2 * np.pi, 200)
    plt.figure(figsize=(7, 4))
    plt.scatter(X, y, s=15, c="gray", label="data")
    for tau, col in [(0.3, "tab:red"), (1.0, "tab:blue")]:
        plt.plot(grid, [predict(g, tau) for g in grid], col, lw=2, label=f"LWR tau={tau}")
    plt.plot(grid, np.sin(grid), "k--", lw=1, label="true sin")
    plt.title("Locally Weighted Regression (bandwidth tau)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{IMG}/m6_02_lwr.png", dpi=110); plt.close()


if __name__ == "__main__":
    knn_regions(); lwr_plot()
    print("Module 6 images written to", IMG)
