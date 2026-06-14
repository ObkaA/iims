import numpy as np


def mse_loss(X, y, theta):
    """MSE: (1/2n) * ||X theta - y||^2"""
    residuals = X @ theta - y
    return 0.5 * np.mean(residuals ** 2)


def mse_grad(X, y, theta):
    """Gradient MSE wzgledem theta: (1/n) * X^T (X theta - y)"""
    n = len(y)
    return (X.T @ (X @ theta - y)) / n


def add_bias(X):
    """Dodaje kolumne jedynek (wyraz wolny)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse_hessian(X, y, theta):
    n = X.shape[0]
    return (2 / n) * (X.T @ X)


def predict(X, theta):
    return X @ theta


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
