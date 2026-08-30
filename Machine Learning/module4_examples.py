"""
Module 4 - Linear Models for Classification - Worked Examples
Run:  python module4_examples.py
Requires: numpy, scikit-learn
"""

import numpy as np


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------
# 1. SIGMOID + hours-studied worked example
# ---------------------------------------------------------------
def sigmoid_demo():
    line("Sigmoid: predict pass/fail (w=1.5, b=-4)")
    for x in (1, 3, 5):
        z = 1.5 * x - 4
        p = sigmoid(z)
        print(f"hours={x}: z={z:+.2f}  P(pass)={p:.3f}  -> {'Pass' if p >= 0.5 else 'Fail'}")


# ---------------------------------------------------------------
# 2. DISCRIMINANT function: sign of w.x + b
# ---------------------------------------------------------------
def discriminant_demo():
    line("Discriminant: class = sign(w.x + b),  w=[1,1], b=-6")
    w, b = np.array([1, 1]), -6
    for pt in ([5, 4], [1, 2], [3, 3]):
        z = w @ np.array(pt) + b
        print(f"point {pt}: z={z:+d} -> class {1 if z > 0 else 0}")


# ---------------------------------------------------------------
# 3. LOG-LOSS (cross-entropy) for individual predictions
# ---------------------------------------------------------------
def logloss_demo():
    line("Log-loss = -[y log p + (1-y) log(1-p)]")
    for y, p in [(1, 0.9), (1, 0.1), (0, 0.2), (1, 0.5)]:
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        print(f"y={y}, p={p}: loss={loss:.3f}")


# ---------------------------------------------------------------
# 4. Manual GRADIENT DESCENT step (matches guide)
# ---------------------------------------------------------------
def gd_step_demo():
    line("One gradient-descent step for logistic regression")
    x = np.array([1., 2., 3., 4.]); y = np.array([0., 0., 1., 1.])
    w, b, eta, n = 0.0, 0.0, 0.1, len(x)
    p = sigmoid(w * x + b)
    err = p - y
    gw = np.sum(err * x) / n
    gb = np.sum(err) / n
    w -= eta * gw; b -= eta * gb
    print(f"start w=0,b=0 -> all p=0.5, errors={err}")
    print(f"grad_w={gw:.3f}, grad_b={gb:.3f} -> new w={w:.3f}, b={b:.3f}")


# ---------------------------------------------------------------
# 5. Full LOGISTIC REGRESSION with sklearn + boundary
# ---------------------------------------------------------------
def logreg_sklearn():
    line("sklearn LogisticRegression on 2D data")
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal([2, 2], 0.7, (40, 2)), rng.normal([5, 5], 0.7, (40, 2))])
    y = np.array([0] * 40 + [1] * 40)
    clf = LogisticRegression().fit(X, y)
    print("weights w =", clf.coef_[0].round(3), " bias b =", round(clf.intercept_[0], 3))
    print("accuracy =", round(clf.score(X, y), 3))
    print("P(y=1) for point (4,4):", round(clf.predict_proba([[4, 4]])[0, 1], 3))


# ---------------------------------------------------------------
# 6. SOFTMAX multi-class worked example
# ---------------------------------------------------------------
def softmax_demo():
    line("Softmax: scores -> probabilities that sum to 1")
    z = np.array([2.0, 1.0, 0.1])
    e = np.exp(z - z.max())          # stable softmax
    p = e / e.sum()
    print("scores :", z)
    print("softmax:", p.round(3), " sum =", round(p.sum(), 3), "-> predict class", int(p.argmax()))


# ---------------------------------------------------------------
# 7. EVALUATION: confusion matrix, precision/recall/F1, ROC-AUC
# ---------------------------------------------------------------
def evaluation_demo():
    line("Evaluation: confusion matrix, precision/recall/F1, ROC-AUC")
    from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                                 f1_score, accuracy_score, roc_auc_score)
    # spam worked example: 15 TP, 5 FN, 3 FP, 77 TN
    y_true = np.array([1] * 20 + [0] * 80)
    y_pred = np.array([1] * 15 + [0] * 5 + [1] * 3 + [0] * 77)
    print("confusion [[TN FP][FN TP]]:\n", confusion_matrix(y_true, y_pred))
    print("accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("precision:", round(precision_score(y_true, y_pred), 3))
    print("recall   :", round(recall_score(y_true, y_pred), 3))
    print("f1       :", round(f1_score(y_true, y_pred), 3))

    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([2, 2], 1.2, (100, 2)), rng.normal([4, 4], 1.2, (100, 2))])
    yy = np.array([0] * 100 + [1] * 100)
    clf = LogisticRegression().fit(X, yy)
    print("ROC-AUC  :", round(roc_auc_score(yy, clf.predict_proba(X)[:, 1]), 3))


if __name__ == "__main__":
    sigmoid_demo()
    discriminant_demo()
    logloss_demo()
    gd_step_demo()
    logreg_sklearn()
    softmax_demo()
    evaluation_demo()
    print("\nAll Module 4 examples ran successfully. See diagrams in module4_images/.")
