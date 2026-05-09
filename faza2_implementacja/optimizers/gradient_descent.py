import numpy as np


class GradientDescent:
    """
    Gradient Descent z opcjonalnym line search (backtracking Armijo).

    Parametry
    ---------
    lr : float
        Poczatkowy krok uczenia (learning rate).
    max_iter : int
        Maksymalna liczba iteracji.
    tol : float
        Kryterium zatrzymania: ||grad||_2 < tol.
    line_search : bool
        Jesli True, stosuje backtracking line search zamiast stalego lr.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6, line_search=False):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.line_search = line_search
        self.loss_history = []
        self.theta = None

    def _backtrack(self, X, y, theta, grad, loss_fn, grad_fn,
                   alpha=0.5, beta=0.8):
        """Backtracking line search (warunek Armijo)."""
        lr = self.lr
        f0 = loss_fn(X, y, theta)
        g_norm_sq = np.dot(grad, grad)
        for _ in range(50):
            if loss_fn(X, y, theta - lr * grad) <= f0 - alpha * lr * g_norm_sq:
                break
            lr *= beta
        return lr

    def fit(self, X, y, loss_fn, grad_fn, theta_init=None):
        """
        Dopasowuje model metodą gradientu prostego.

        Parametry
        ---------
        X : ndarray (n, p)
        y : ndarray (n,)
        loss_fn : callable(X, y, theta) -> float
        grad_fn : callable(X, y, theta) -> ndarray (p,)
        theta_init : ndarray lub None
        """
        p = X.shape[1]
        self.theta = np.zeros(p) if theta_init is None else theta_init.copy()
        self.loss_history = []

        for _ in range(self.max_iter):
            loss = loss_fn(X, y, self.theta)
            self.loss_history.append(loss)

            grad = grad_fn(X, y, self.theta)
            if np.linalg.norm(grad) < self.tol:
                break

            lr = (self._backtrack(X, y, self.theta, grad, loss_fn, grad_fn)
                  if self.line_search else self.lr)

            self.theta -= lr * grad

        return self
