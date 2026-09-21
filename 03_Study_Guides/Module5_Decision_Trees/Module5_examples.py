"""
Module 5 - Decision Trees - Worked Examples
Run:  python Module5_examples.py
Requires: numpy, pandas, scikit-learn
Every printed number matches the worked examples in the guide.
"""

import numpy as np
import pandas as pd


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


# ---------------------------------------------------------------
# 0. The classic PlayTennis dataset (Mitchell, Ch.3)
# ---------------------------------------------------------------
def playtennis():
    data = [
        ["Sunny", "Hot", "High", "Weak", "No"],
        ["Sunny", "Hot", "High", "Strong", "No"],
        ["Overcast", "Hot", "High", "Weak", "Yes"],
        ["Rain", "Mild", "High", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Strong", "No"],
        ["Overcast", "Cool", "Normal", "Strong", "Yes"],
        ["Sunny", "Mild", "High", "Weak", "No"],
        ["Sunny", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Mild", "Normal", "Weak", "Yes"],
        ["Sunny", "Mild", "Normal", "Strong", "Yes"],
        ["Overcast", "Mild", "High", "Strong", "Yes"],
        ["Overcast", "Hot", "Normal", "Weak", "Yes"],
        ["Rain", "Mild", "High", "Strong", "No"],
    ]
    return pd.DataFrame(data, columns=["Outlook", "Temp", "Humidity", "Wind", "Play"])


# ---------------------------------------------------------------
# 1. ENTROPY and INFORMATION GAIN from scratch
# ---------------------------------------------------------------
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def info_gain(df, attr, target="Play"):
    H_parent = entropy(df[target])
    H_children = sum(len(sub) / len(df) * entropy(sub[target])
                     for _, sub in df.groupby(attr))
    return H_parent - H_children


def split_info(df, attr):
    frac = df[attr].value_counts(normalize=True).values
    return float(-np.sum(frac * np.log2(frac)))


def entropy_demo():
    line("Entropy of some sets (bits)")
    for s in (["Y"] * 9 + ["N"] * 5, ["Y"] * 3 + ["N"] * 4, ["Y"] * 8 + ["N"] * 2, ["Y"] * 4 + ["N"] * 4):
        print(f"[{s.count('Y')}+,{s.count('N')}-] -> H = {entropy(s):.3f}")


def gain_demo():
    line("Information Gain of each root attribute (PlayTennis)")
    df = playtennis()
    print(f"Parent entropy H(S) = {entropy(df['Play']):.3f}")
    for a in ["Outlook", "Temp", "Humidity", "Wind"]:
        print(f"Gain({a:8s}) = {info_gain(df, a):.3f}")
    print("-> Root = Outlook (largest gain, 0.246)")


def gain_ratio_demo():
    line("Gain Ratio curbs the many-valued 'Day' attribute")
    df = playtennis().copy()
    df["Day"] = [f"D{i+1}" for i in range(len(df))]
    for a in ["Outlook", "Day"]:
        g, si = info_gain(df, a), split_info(df, a)
        print(f"{a:8s}: Gain={g:.3f}  SplitInfo={si:.3f}  GainRatio={g/si:.3f}")


# ---------------------------------------------------------------
# 2. CONTINUOUS threshold selection
# ---------------------------------------------------------------
def continuous_demo():
    line("Continuous attribute: candidate thresholds at label changes")
    temp = np.array([48, 60, 72, 80, 90])
    lab = np.array(["No", "Yes", "Yes", "Yes", "No"])
    cands = [(temp[i] + temp[i + 1]) / 2 for i in range(len(temp) - 1) if lab[i] != lab[i + 1]]
    print("sorted temps:", temp.tolist(), "labels:", lab.tolist())
    print("candidate thresholds (label changes):", cands)
    df_full = pd.DataFrame({"t": temp, "Play": lab})
    for c in cands:
        d = df_full.assign(bin=np.where(df_full.t < c, f"<{c}", f">={c}"))
        print(f"  Temp<{c}: Gain = {info_gain(d, 'bin'):.3f}")


# ---------------------------------------------------------------
# 3. sklearn DecisionTree (entropy) + printed rules
# ---------------------------------------------------------------
def sklearn_demo():
    line("scikit-learn DecisionTreeClassifier (criterion='entropy')")
    from sklearn.tree import DecisionTreeClassifier, export_text
    df = playtennis()
    X = pd.get_dummies(df.drop(columns="Play"))
    y = (df["Play"] == "Yes").astype(int)
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    print(export_text(clf, feature_names=list(X.columns)).strip())
    print("train accuracy:", round(clf.score(X, y), 3))


# ---------------------------------------------------------------
# 4. Overfitting vs depth, and cost-complexity pruning
# ---------------------------------------------------------------
def pruning_demo():
    line("Overfitting vs depth + ccp_alpha post-pruning")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                               n_redundant=2, flip_y=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for d in (1, 3, 5, 10, None):
        c = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        print(f"depth={str(d):>4}: train={c.score(Xtr,ytr):.3f}  test={c.score(Xte,yte):.3f}")
    for a in (0.0, 0.005, 0.02):
        c = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(Xtr, ytr)
        print(f"ccp_alpha={a:<5}: test={c.score(Xte,yte):.3f}  #leaves={c.get_n_leaves()}")


if __name__ == "__main__":
    entropy_demo()
    gain_demo()
    gain_ratio_demo()
    continuous_demo()
    sklearn_demo()
    pruning_demo()
    print("\nAll Module 5 examples ran successfully. See diagrams in images/.")
