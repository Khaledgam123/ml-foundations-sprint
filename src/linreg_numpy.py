import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- core functions ----------
def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """X @ w + b"""
    return X @ w + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean-squared error"""
    return np.mean((y_true - y_pred) ** 2)


def gradients(X, y, w, b):
    """Analytical ∂MSE/∂w and ∂MSE/∂b"""
    n = len(y)
    y_hat = predict(X, w, b)
    grad_w = (-2 / n) * X.T @ (y - y_hat)
    grad_b = (-2 / n) * np.sum(y - y_hat)
    return grad_w, grad_b


def fit_gd(X, y, lr=1e-2, epochs=500):
    """Simple gradient-descent trainer; returns weights, bias, loss history"""
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(epochs):
        gw, gb = gradients(X, y, w, b)
        w -= lr * gw
        b -= lr * gb
        losses.append(mse(y, predict(X, w, b)))
    return w, b, np.array(losses)


# ---------- demo / verification ----------
if __name__ == "__main__":
    np.random.seed(0)

    # synthetic dataset (1000 × 3)
    X = np.random.randn(1000, 3)
    true_w = np.array([3.0, -2.0, 1.5])
    true_b = 0.7
    y = X @ true_w + true_b + 0.5 * np.random.randn(1000)

    # closed-form solution
    w_cf = np.linalg.inv(X.T @ X) @ X.T @ y
    b_cf = y.mean() - X.mean(0) @ w_cf
    mse_cf = mse(y, predict(X, w_cf, b_cf))

    # gradient descent
    w_gd, b_gd, losses = fit_gd(X, y, lr=1e-2, epochs=500)
    mse_gd = mse(y, predict(X, w_gd, b_gd))

    print("closed-form MSE:", round(mse_cf, 6))
    print("GD        MSE:", round(mse_gd, 6))

    # plot loss curve
    Path("experiments").mkdir(exist_ok=True)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("GD loss curve")
    plt.savefig("experiments/day2_loss.png", dpi=150, bbox_inches="tight")
    plt.close()