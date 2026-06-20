"""
Dataset generators and CSV loader.
All implementations from scratch — pure NumPy, no sklearn, no scipy.
"""
import numpy as np
import csv


def generate_linear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    """y = 2.5·x + 1 + ε,   ε ~ N(0, (3·noise)²)"""
    rng = np.random.default_rng(seed)
    X   = rng.uniform(-3, 3, (n_samples, 1))
    y   = 2.5 * X[:, 0] + 1.0 + rng.normal(0, noise * 3, n_samples)
    return X, y


def generate_nonlinear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    """y = 2·sin(x) + ε"""
    rng = np.random.default_rng(seed)
    X   = rng.uniform(-3, 3, (n_samples, 1))
    y   = np.sin(X[:, 0]) * 2 + rng.normal(0, noise, n_samples)
    return X, y


def generate_logistic_regression(n_samples: int = 300, noise: float = 0.15, seed: int = 42):
    """
    Two 2-D Gaussian blobs: class 0 centred at (+1,+1), class 1 at (−1,−1).

    Sampling without numpy.random.multivariate_normal:
        x = L·z + μ,   z ~ N(0,I),   L = Cholesky(Σ)
    """
    rng    = np.random.default_rng(seed)
    n_each = n_samples // 2

    def sample_mvn(mean, cov, n):
        L = np.linalg.cholesky(np.array(cov, dtype=np.float64))
        return rng.standard_normal((n, 2)) @ L.T + np.array(mean)

    X0 = sample_mvn([ 1,  1], [[1, 0.3], [0.3, 1]],  n_each)
    X1 = sample_mvn([-1, -1], [[1,-0.3], [-0.3, 1]], n_each)
    X  = np.vstack([X0, X1])
    y  = np.hstack([np.zeros(n_each), np.ones(n_each)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_circles(n_samples: int = 300, noise: float = 0.05, seed: int = 42):
    """
    Two concentric circles — implemented from scratch (no sklearn).

    Inner ring: radius 0.5,  label 0
    Outer ring: radius 1.0,  label 1

    Points sampled uniformly by angle θ ~ U[0, 2π]:
        x = r·cos(θ) + ε,   y = r·sin(θ) + ε
    """
    rng    = np.random.default_rng(seed)
    n_each = n_samples // 2

    θ0 = rng.uniform(0, 2 * np.pi, n_each)
    θ1 = rng.uniform(0, 2 * np.pi, n_each)

    X0 = np.column_stack([
        0.5 * np.cos(θ0) + rng.normal(0, noise, n_each),
        0.5 * np.sin(θ0) + rng.normal(0, noise, n_each),
    ])
    X1 = np.column_stack([
        1.0 * np.cos(θ1) + rng.normal(0, noise, n_each),
        1.0 * np.sin(θ1) + rng.normal(0, noise, n_each),
    ])

    X   = np.vstack([X0, X1])
    y   = np.hstack([np.zeros(n_each), np.ones(n_each)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


DATASETS = {
    "Linear Data":       generate_linear_regression,
    "Noisy Linear Data": lambda: generate_linear_regression(noise=1.2),
    "Logistic 2D":       generate_logistic_regression,
    "Circles":           generate_circles,
}


def load_csv(path: str):
    """
    Load a plain CSV (last column = target).  Returns (X, y).
    First row skipped if it contains non-numeric data (header).
    """
    data = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first  = next(reader, None)
        if first:
            try:
                data.append([float(v) for v in first])
            except ValueError:
                pass   # header row — skip
        for row in reader:
            try:
                data.append([float(v) for v in row])
            except ValueError:
                continue
    arr = np.array(data, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]
