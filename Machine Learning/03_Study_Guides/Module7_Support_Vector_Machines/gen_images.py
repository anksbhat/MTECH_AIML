"""
Module 7 - generate diagrams referenced by the guide.
Run:  python gen_images.py
Requires: numpy, matplotlib, scikit-learn
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

IMG = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG, exist_ok=True)


def margin_plot():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal([1, 1], 0.4, (20, 2)), rng.normal([4, 4], 0.4, (20, 2))]
    y = np.array([0] * 20 + [1] * 20)
    clf = SVC(kernel="linear", C=1000).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(0, 5, 50)
    ys = -(w[0] * xs + b) / w[1]
    m = 1 / np.linalg.norm(w)
    dy = m * np.sqrt(1 + (w[0] / w[1]) ** 2)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    plt.plot(xs, ys, "k-", label="boundary w·x+b=0")
    plt.plot(xs, ys + dy, "k--", label="margin ±1")
    plt.plot(xs, ys - dy, "k--")
    sv = clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=160, facecolors="none", edgecolors="lime",
                linewidths=2, label="support vectors")
    plt.title("Maximum-margin hyperplane"); plt.legend(fontsize=8)
    plt.xlim(0, 5); plt.ylim(0, 5)
    plt.tight_layout(); plt.savefig(f"{IMG}/m7_01_margin.png", dpi=110); plt.close()


def soft_margin_plot():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.2, random_state=6)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, C in zip(axes, (0.05, 100)):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, alpha=0.2, cmap="coolwarm")
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
        ax.set_title(f"C={C}  ({'soft/wide' if C < 1 else 'hard/narrow'})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Soft margin: effect of C")
    plt.tight_layout(); plt.savefig(f"{IMG}/m7_02_soft_margin.png", dpi=110); plt.close()


def kernel_plot():
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=200, noise=0.08, factor=0.4, random_state=0)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, ("linear", "rbf")):
        clf = SVC(kernel=k, C=1.0, gamma="scale").fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
        ax.set_title(f"{k} kernel  (acc={clf.score(X,y):.2f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Kernel trick: linear can't separate circles, RBF can")
    plt.tight_layout(); plt.savefig(f"{IMG}/m7_03_kernel.png", dpi=110); plt.close()


if __name__ == "__main__":
    margin_plot(); soft_margin_plot(); kernel_plot()
    print("Module 7 images written to", IMG)
