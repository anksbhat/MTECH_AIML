"""
Module 2 (Contact Session 2) - Worked Examples
Run:  python module2_examples.py

Covers the ML Workflow guide + Mathematical Preliminaries guide.
Requires: numpy, pandas, scikit-learn
"""

import numpy as np


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


# ---------------------------------------------------------------
# 1. OUTLIERS - IQR rule (slide exercise data)
# ---------------------------------------------------------------
def iqr_outliers():
    line("Outliers via IQR (slide exercise)")
    data = np.array([10, 2, 11, 15, 11, 14, 13, 17, 12, 22, 14, 11])
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"Q1={q1}, Q3={q3}, IQR={iqr}")
    print(f"bounds = [{lo}, {hi}]")
    print("outliers ->", data[(data < lo) | (data > hi)])


# ---------------------------------------------------------------
# 2. OUTLIERS - 3-sigma / Z-score
# ---------------------------------------------------------------
def zscore_outliers():
    line("Outliers via 3-sigma (Z-score)")
    x = np.array([48, 51, 49, 50, 52, 47, 53, 95])  # 95 is an outlier
    mu, sigma = x.mean(), x.std()
    z = (x - mu) / sigma
    print(f"mean={mu:.2f}, std={sigma:.2f}")
    print("z-scores:", z.round(2))
    print("outliers (|z|>3):", x[np.abs(z) > 3], " -- (use >2 for small samples:", x[np.abs(z) > 2], ")")


# ---------------------------------------------------------------
# 3. MISSING VALUE IMPUTATION (mean)
# ---------------------------------------------------------------
def imputation():
    line("Missing-value imputation (mean)")
    from sklearn.impute import SimpleImputer
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, 6.0], [4.0, np.nan]])
    imp = SimpleImputer(strategy="mean")
    print("before:\n", X)
    print("after (NaNs filled with column mean):\n", imp.fit_transform(X))


# ---------------------------------------------------------------
# 4. FEATURE SCALING - min-max and standardization (slide numbers)
# ---------------------------------------------------------------
def scaling():
    line("Feature scaling: min-max & z-score (slide worked values)")
    # Min-max: income 12000..98000 -> [0,1], value 73600
    v, mn, mx = 73600, 12000, 98000
    print("Min-max of 73600 =", round((v - mn) / (mx - mn), 3), "(slide: 0.716)")
    # Z-score: mu=54000 sigma=16000, value 73600
    print("Z-score of 73600 =", round((73600 - 54000) / 16000, 3), "(slide: 1.225)")

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    col = np.array([[12000], [30000], [54000], [73600], [98000]])
    print("MinMaxScaler:", MinMaxScaler().fit_transform(col).ravel().round(3))
    print("StandardScaler:", StandardScaler().fit_transform(col).ravel().round(3))


# ---------------------------------------------------------------
# 5. ENCODING categorical features: one-hot vs label
# ---------------------------------------------------------------
def encoding():
    line("Encoding categorical features")
    import pandas as pd
    df = pd.DataFrame({"Color": ["Red", "Green", "Blue", "Green"],
                       "Size":  ["Low", "Med", "High", "Med"]})
    print("One-hot (nominal Color):")
    print(pd.get_dummies(df["Color"], prefix="is").astype(int))

    from sklearn.preprocessing import LabelEncoder
    order = {"Low": 0, "Med": 1, "High": 2}   # ordinal -> keep order
    print("Label/ordinal encode Size:", df["Size"].map(order).tolist())


# ---------------------------------------------------------------
# 6. EQUAL-WIDTH BINNING (discretization)
# ---------------------------------------------------------------
def binning():
    line("Equal-width binning (discretization)")
    import pandas as pd
    ages = pd.Series([5, 17, 22, 34, 46, 58, 63, 77, 88, 99])
    bins = pd.cut(ages, bins=5)   # width = (99-5)/5 approx 18.8
    print(pd.DataFrame({"age": ages, "bin": bins}).to_string(index=False))


# ---------------------------------------------------------------
# 7. STRATIFIED train/test split (keep class proportions)
# ---------------------------------------------------------------
def stratified_split():
    line("Stratified split keeps class proportions (IRIS pitfall)")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33,
                                          stratify=y, random_state=0)
    print("train class counts:", np.bincount(ytr))
    print("test  class counts:", np.bincount(yte), "(balanced ~ each class)")


# ---------------------------------------------------------------
# 8. CLASSIFICATION METRICS (confusion matrix, precision, recall, F1)
# ---------------------------------------------------------------
def metrics_demo():
    line("Classification metrics (spam worked example)")
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    # Build 100 emails: 20 spam(1), 80 ham(0). Model: 15 TP, 3 FP, 5 FN, 77 TN
    y_true = np.array([1] * 20 + [0] * 80)
    y_pred = np.array([1] * 15 + [0] * 5 + [1] * 3 + [0] * 77)  # 15TP,5FN,3FP,77TN
    print("confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_true, y_pred))
    print("accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("precision:", round(precision_score(y_true, y_pred), 3))
    print("recall   :", round(recall_score(y_true, y_pred), 3))
    print("f1       :", round(f1_score(y_true, y_pred), 3))


# ---------------------------------------------------------------
# 9. MATH PRELIMS: dot product, gradient descent, Bayes, entropy
# ---------------------------------------------------------------
def math_prelims():
    line("Math preliminaries: dot product / gradient / Bayes / entropy")
    a, b = np.array([3, 4]), np.array([4, 1])
    print("dot product a.b =", a @ b, " | norm |a| =", np.linalg.norm(a))

    w, eta = 2.6, 0.2
    path = []
    for _ in range(6):
        w = w - eta * (2 * w)   # minimise w^2
        path.append(round(w, 4))
    print("gradient descent on w^2:", path)

    P_spam, P_ham, l_s, l_h = 0.2, 0.8, 0.6, 0.05
    P_free = l_s * P_spam + l_h * P_ham
    print("Bayes P(Spam|'free') =", round(l_s * P_spam / P_free, 3))

    def entropy(p):
        p = np.array(p); p = p[p > 0]
        return -np.sum(p * np.log2(p))
    print("entropy fair coin:", round(entropy([0.5, 0.5]), 3),
          "| entropy 9Y/5N:", round(entropy([9 / 14, 5 / 14]), 3))


if __name__ == "__main__":
    iqr_outliers()
    zscore_outliers()
    imputation()
    scaling()
    encoding()
    binning()
    stratified_split()
    metrics_demo()
    math_prelims()
    print("\nAll Module 2 examples ran successfully. See diagrams in images/.")
