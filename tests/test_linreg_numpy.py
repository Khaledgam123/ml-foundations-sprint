import numpy as np
from src.linreg_numpy import predict, mse, gradients, fit_gd

def test_gradient_shapes():
    X = np.random.randn(8, 4); y = np.random.randn(8)
    gw, gb = gradients(X, y, np.zeros(4), 0.0)
    assert gw.shape == (4,)
    assert isinstance(gb, float)

def test_gd_converges():
    np.random.seed(0)
    X = np.random.randn(200, 2)
    w_true = np.array([2.0, -1.0]); b_true = 0.5
    y = X @ w_true + b_true + 0.1*np.random.randn(200)

    w_cf = np.linalg.inv(X.T @ X) @ X.T @ y
    b_cf = y.mean() - X.mean(0) @ w_cf
    mse_cf = mse(y, predict(X, w_cf, b_cf))

    _, _, losses = fit_gd(X, y, lr=5e-2, epochs=300)
    assert losses[-1] <= 1.01 * mse_cf