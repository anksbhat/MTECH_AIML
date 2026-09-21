"""
Module 5 - generate the diagrams referenced by the guide.
Run:  python gen_images.py     (writes PNGs into images/)
Requires: numpy, matplotlib, scikit-learn
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG, exist_ok=True)


def entropy_curve():
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    plt.figure(figsize=(6, 4))
    plt.plot(p, H, lw=2)
    plt.axvline(0.5, ls="--", c="gray")
    plt.scatter([0.5], [1.0], c="red", zorder=5)
    plt.annotate("max impurity\nH=1 at p=0.5", (0.5, 1.0), (0.55, 0.6),
                 arrowprops=dict(arrowstyle="->"))
    plt.title("Entropy of a binary node")
    plt.xlabel("p (fraction positive)"); plt.ylabel("H(p)  [bits]")
    plt.tight_layout(); plt.savefig(f"{IMG}/m5_01_entropy.png", dpi=110); plt.close()


def playtennis_tree():
    plt.figure(figsize=(8, 5)); ax = plt.gca(); ax.axis("off")

    def box(x, y, txt, c):
        ax.add_patch(plt.Rectangle((x - .09, y - .04), .18, .08, fc=c, ec="black"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)

    def edge(x1, y1, x2, y2, txt):
        ax.plot([x1, x2], [y1, y2], "k-", lw=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, txt, fontsize=8, color="blue",
                bbox=dict(fc="white", ec="none"))

    box(.5, .9, "Outlook", "#cfe8ff")
    box(.2, .6, "Humidity", "#cfe8ff"); box(.5, .6, "Yes", "#c9f7c9"); box(.8, .6, "Wind", "#cfe8ff")
    edge(.5, .86, .2, .64, "Sunny"); edge(.5, .86, .5, .64, "Overcast"); edge(.5, .86, .8, .64, "Rain")
    box(.1, .3, "No", "#f7c9c9"); box(.3, .3, "Yes", "#c9f7c9")
    edge(.2, .56, .1, .34, "High"); edge(.2, .56, .3, .34, "Normal")
    box(.7, .3, "Yes", "#c9f7c9"); box(.9, .3, "No", "#f7c9c9")
    edge(.8, .56, .7, .34, "Weak"); edge(.8, .56, .9, .34, "Strong")
    plt.title("Learned PlayTennis decision tree")
    plt.tight_layout(); plt.savefig(f"{IMG}/m5_02_playtennis_tree.png", dpi=110); plt.close()


def overfitting_curve():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                               n_redundant=2, flip_y=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    depths = range(1, 16)
    tr = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xtr, ytr) for d in depths]
    te = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xte, yte) for d in depths]
    plt.figure(figsize=(6, 4))
    plt.plot(list(depths), tr, "o-", label="train")
    plt.plot(list(depths), te, "s-", label="test")
    plt.axvline(int(np.argmax(te)) + 1, ls="--", c="gray", label="best test depth")
    plt.title("Decision-tree overfitting vs depth")
    plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{IMG}/m5_03_overfitting.png", dpi=110); plt.close()


if __name__ == "__main__":
    entropy_curve(); playtennis_tree(); overfitting_curve()
    print("Module 5 images written to", IMG)
