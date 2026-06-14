"""Data generation and loading utilities."""
import numpy as np
import csv
from pathlib import Path


def generate_linear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n_samples, 1))
    y = 2.5 * X[:, 0] + 1.0 + rng.normal(0, noise * 3, n_samples)
    return X, y


def generate_nonlinear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n_samples, 1))
    y = np.sin(X[:, 0]) * 2 + rng.normal(0, noise, n_samples)
    return X, y


def generate_logistic_regression(n_samples: int = 300, noise: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_each = n_samples // 2
    X0 = rng.multivariate_normal([1, 1], [[1, 0.3], [0.3, 1]], n_each)
    X1 = rng.multivariate_normal([-1, -1], [[1, -0.3], [-0.3, 1]], n_each)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_circles(n_samples: int = 300, noise: float = 0.05, seed: int = 42):
    """Linearly-separable approximation of two concentric classes."""
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    return X, y.astype(float)


DATASETS = {
    "Linear Data": generate_linear_regression,
    "Noisy Linear Data": lambda: generate_linear_regression(noise=1.2),
    "Logistic 2D": generate_logistic_regression,
    "Circles": generate_circles,
}


def load_csv(path: str):
    """Load a CSV where the last column is the target. Returns (X, y)."""
    data = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                data.append([float(v) for v in row])
            except ValueError:
                continue
    arr = np.array(data)
    return arr[:, :-1], arr[:, -1]
