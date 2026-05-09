import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_loss(X, y, theta):
    """Binary cross-entropy: -(1/n) sum[ y log(p) + (1-y) log(1-p) ]"""
    p = sigmoid(X @ theta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_grad(X, y, theta):
    """Gradient log-loss: (1/n) * X^T (p - y)"""
    n = len(y)
    p = sigmoid(X @ theta)
    return (X.T @ (p - y)) / n


def log_hessian(X, y, theta):
    """
    Hessian log-loss (macierz Fishera): (1/n) * X^T W X,
    gdzie W = diag(p * (1 - p)).
    """
    n = len(y)
    p = sigmoid(X @ theta)
    w = p * (1 - p)
    return (X.T * w) @ X / n


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict_proba(X, theta):
    return sigmoid(X @ theta)


def predict(X, theta, threshold=0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """Zwraca macierz [[TN, FP], [FN, TP]]."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
