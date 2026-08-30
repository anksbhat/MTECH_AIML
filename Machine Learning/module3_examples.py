"""
Module 3 - Linear Models for Regression - Worked Examples
Run:  python module3_examples.py
Requires: numpy, scikit-learn
"""

import numpy as np


def line(t):
    print("\n" + "=" * 62 + "\n" + t + "\n" + "=" * 62)


# ---------------------------------------------------------------
# 1. COST FUNCTION (MSE) for a given line
# ---------------------------------------------------------------
def cost_demo():
    line("Cost function J(theta) = (1/2n) sum (h(x)-y)^2")
    x = np.array([1., 2., 3.]); y = np.array([3., 5., 7.])
    for (t0, t1) in [(0, 1), (1, 2)]:
        h = t0 + t1 * x
        J = np.sum((h - y) ** 2) / (2 * len(x))
        print(f"line h(x)={t0}+{t1}x -> J = {J:.4f}")


# ---------------------------------------------------------------
# 2. CLOSED FORM (Normal Equation)  vs  sklearn
# ---------------------------------------------------------------
def closed_form():
    line("Closed form: theta = (X^T X)^-1 X^T y")
    x = np.array([1., 2., 3.]); y = np.array([3., 5., 7.])
    X = np.c_[np.ones_like(x), x]
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    print("normal equation theta =", theta.round(4), "-> h(x)=1+2x")

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    print("sklearn intercept/slope =", round(reg.intercept_, 4), round(reg.coef_[0], 4))


# ---------------------------------------------------------------
# 3. GRADIENT DESCENT - fully worked iterations (matches the guide)
# ---------------------------------------------------------------
def gradient_descent():
    line("Batch Gradient Descent - first iterations (matches guide)")
    x = np.array([1., 2., 3.]); y = np.array([3., 5., 7.])
    t0, t1, eta, n = 0.0, 0.0, 0.1, len(x)
    for it in range(1, 6):
        h = t0 + t1 * x
        err = h - y
        g0 = np.sum(err) / n
        g1 = np.sum(err * x) / n
        t0 -= eta * g0
        t1 -= eta * g1
        print(f"iter{it}: grad=({g0:7.4f},{g1:7.4f}) -> theta0={t0:.4f}, theta1={t1:.4f}")
    print("... converging toward the exact answer (1, 2)")


# ---------------------------------------------------------------
# 4. BATCH vs STOCHASTIC GD (sklearn SGDRegressor)
# ---------------------------------------------------------------
def sgd_demo():
    line("Stochastic Gradient Descent (sklearn SGDRegressor)")
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (200, 1))
    y = 4 + 3 * X.ravel() + rng.normal(0, 1, 200)   # true: y = 4 + 3x
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)                         # scaling helps GD converge
    sgd = SGDRegressor(max_iter=1000, eta0=0.01, random_state=0).fit(Xs, y)
    from sklearn.metrics import r2_score
    print("true line: y = 4 + 3x")
    print("SGD R^2 on data:", round(r2_score(y, sgd.predict(Xs)), 4))
    for xq in (2.0, 5.0):
        pred = sgd.predict(scaler.transform([[xq]]))[0]
        print(f"  predict x={xq}: SGD={pred:.2f}  vs true={4 + 3 * xq:.2f}")


# ---------------------------------------------------------------
# 5. POLYNOMIAL REGRESSION - underfit vs good vs overfit (train/test MSE)
# ---------------------------------------------------------------
def polynomial_demo():
    line("Polynomial regression: underfit vs good vs overfit")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    rng = np.random.default_rng(1)
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    y = 2 - 3 * (X.ravel() - 0.5) ** 2 + rng.normal(0, 0.05, 60)   # quadratic truth
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for deg in (1, 2, 12):
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(Xtr, ytr)
        tr = mean_squared_error(ytr, m.predict(Xtr))
        te = mean_squared_error(yte, m.predict(Xte))
        tag = "underfit" if deg == 1 else ("good" if deg == 2 else "overfit")
        print(f"degree {deg:2d}: train MSE={tr:.4f}  test MSE={te:.4f}  ({tag})")


# ---------------------------------------------------------------
# 6. R-squared metric
# ---------------------------------------------------------------
def r2_demo():
    line("R-squared (goodness of fit)")
    from sklearn.metrics import r2_score, mean_squared_error
    y = np.array([3., 5., 7.]); yhat = np.array([2.8, 5.1, 7.1])
    print("R^2 =", round(r2_score(y, yhat), 4), " RMSE =", round(mean_squared_error(y, yhat) ** 0.5, 4))


# ---------------------------------------------------------------
# 7. REGULARIZATION: Ridge vs Lasso vs ElasticNet (Lasso zeros weights)
# ---------------------------------------------------------------
def regularization_demo():
    line("Regularization: Ridge (L2) vs Lasso (L1) vs ElasticNet")
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
    rng = np.random.default_rng(2)
    n = 100
    X = rng.normal(0, 1, (n, 6))
    # only features 0 and 1 truly matter; 2-5 are noise
    y = 5 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 0.5, n)
    for name, model in [("OLS", LinearRegression()),
                        ("Ridge(a=1)", Ridge(alpha=1.0)),
                        ("Lasso(a=0.1)", Lasso(alpha=0.1)),
                        ("ElasticNet(a=0.1,r=0.5)", ElasticNet(alpha=0.1, l1_ratio=0.5))]:
        model.fit(X, y)
        print(f"{name:24s} coefs = {np.round(model.coef_, 2)}")
    print("Notice: Lasso pushes the noise-feature coefficients to exactly 0 (feature selection).")


if __name__ == "__main__":
    cost_demo()
    closed_form()
    gradient_descent()
    sgd_demo()
    polynomial_demo()
    r2_demo()
    regularization_demo()
    print("\nAll Module 3 examples ran successfully. See diagrams in module3_images/.")
