"""
Module 1 - Worked Examples (Beginner friendly)
Run:  python module1_examples.py

Each section matches the Module1_Beginner_Guide.md so you can SEE the ideas run.
Requires: numpy, scikit-learn  (pip install numpy scikit-learn)
"""

import numpy as np


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
# 4.1 CLASSIFICATION  -> predict a CATEGORY (Job Offered? Yes/No)
# ---------------------------------------------------------------
def classification_demo():
    line("4.1 CLASSIFICATION  (Employability prediction)")
    from sklearn.tree import DecisionTreeClassifier

    # Encode words as numbers: Poor=0, Average=1, Good=2, Excellent=3
    # columns: CGPA, Communication, Aptitude, Programming
    X = np.array([
        [9.1, 1, 2, 3],
        [8.4, 2, 2, 2],
        [8.3, 0, 1, 1],
        [7.1, 1, 2, 1],
        [8.2, 2, 3, 3],
    ])
    y = np.array(["Yes", "Yes", "No", "No", "No"])   # categorical label

    clf = DecisionTreeClassifier(random_state=0).fit(X, y)

    new_candidate = [[8.6, 2, 2, 3]]  # CGPA 8.6, Comm Good, Apt Good, Prog Excellent
    print("Training accuracy:", clf.score(X, y))
    print("New candidate features:", new_candidate[0])
    print("Predicted Job Offered? ->", clf.predict(new_candidate)[0])


# ---------------------------------------------------------------
# 4.2 REGRESSION  -> predict a NUMBER (used-car price)
# ---------------------------------------------------------------
def regression_demo():
    line("4.2 REGRESSION  (Used-car price from distance travelled)")
    from sklearn.linear_model import LinearRegression

    dist = np.array([15, 30, 45, 60, 75, 90, 110, 130]).reshape(-1, 1)  # 000 km
    price = np.array([6.0, 5.2, 4.4, 3.9, 3.2, 2.8, 2.1, 1.6])          # lakh Rs

    reg = LinearRegression().fit(dist, price)
    m, c = reg.coef_[0], reg.intercept_
    print(f"Learned line:  price = {m:.3f} * distance + {c:.2f}   (lakh Rs)")
    print(f"Meaning: each +10,000 km lowers price by ~Rs {abs(m)*10:.2f} lakh")
    for d in (50, 100, 125):
        print(f"  Predicted price at {d}k km: Rs {reg.predict([[d]])[0]:.2f} lakh")

    # Show the loss (Mean Squared Error) on the training data
    preds = reg.predict(dist)
    mse = np.mean((price - preds) ** 2)
    print(f"Mean Squared Error on data: {mse:.4f}")


# ---------------------------------------------------------------
# 4.3 CLUSTERING (unsupervised) -> find groups without labels
# ---------------------------------------------------------------
def clustering_demo():
    line("4.3 CLUSTERING  (Market segmentation, no labels)")
    from sklearn.cluster import KMeans

    # [avg money spent, visits per month]
    X = np.array([
        [2, 2], [2.3, 1.8], [1.8, 2.2],
        [6, 3], [5.8, 3.2], [6.2, 2.8],
        [4, 6], [4.2, 5.8], [3.8, 6.1],
    ])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    print("Cluster label assigned to each customer:", km.labels_)
    print("Cluster centroids (avg_spent, visits):")
    print(km.cluster_centers_.round(2))
    print("Inertia (total within-cluster distance, lower=tighter):",
          round(km.inertia_, 3))

    new_customer = [[5.9, 3.1]]
    print("New customer", new_customer[0], "-> cluster",
          km.predict(new_customer)[0])


# ---------------------------------------------------------------
# 3.4 The LMS weight-update rule from "Design a Learning System"
#     (tiny hand-rolled gradient step, no library)
# ---------------------------------------------------------------
def lms_update_demo():
    line("3.4 LMS update rule  (how weights learn - checkers idea)")
    # Pretend a board has 2 features; learn weights so V_hat approaches V_train
    x = np.array([1.0, 3.0, 2.0])   # x0=1 (bias), x1=3, x2=2
    w = np.array([0.0, 0.0, 0.0])   # start all weights at 0
    eta = 0.05                      # learning rate
    V_train = 10.0                  # the "true" value we want V_hat to reach

    for step in range(1, 8):
        V_hat = np.dot(w, x)                 # current prediction
        error = V_train - V_hat              # how wrong we are
        w = w + eta * error * x              # nudge every weight
        print(f"step {step}: V_hat={V_hat:6.3f}  error={error:6.3f}  weights={w.round(3)}")
    print("-> V_hat is climbing toward V_train (10). That's learning!")


if __name__ == "__main__":
    classification_demo()
    regression_demo()
    clustering_demo()
    lms_update_demo()
    print("\nAll Module 1 examples ran successfully. Open the diagrams in module1_images/.")
