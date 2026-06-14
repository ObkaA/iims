import numpy as np
from sklearn.datasets import make_moons, make_circles

def make_linear(n=200, p=5, noise=1.0, seed=42):
    """
    Generuje dane do regresji liniowej: y = X @ beta + epsilon.
    Zwraca X (n x p), y (n,), prawdziwe wspolczynniki beta (p,).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y, beta


def make_classification(n=500, p=2, sep=1.5, seed=42):
    """
    Generuje liniowo separowalne dane binarne (dwie gaussowskie chmury).
    Zwraca X (n x p), y (n,) z wartosciami {0, 1}.
    sep kontroluje odleglosc miedzy centroidami.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, p))
    X1 = rng.standard_normal((n - half, p)) + sep
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]

def generate_moons(n=500, noise=0.1, seed=42):

    X, y = make_moons(
        n_samples=n,
        noise=noise,
        random_state=seed
    )

    return X, y


def generate_circles(n=500, noise=0.05, factor=0.5, seed=42):

    X, y = make_circles(
        n_samples=n,
        noise=noise,
        factor=factor,
        random_state=seed
    )

    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    """Podzial na zbior treningowy i testowy."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """Standaryzacja: odejmuje srednia i dzieli przez odchylenie z X_train."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std
