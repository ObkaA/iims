import numpy as np


class NewtonMethod:
    """
    Metoda Newtona (Newton-Raphson) dla problemow optymalizacji wypuklej.

    Na kazda iteracje rozwiazuje uklad: H(theta) * delta = -grad(theta),
    gdzie H to macierz Hessiana. Dla zagwarantowania dodatniej określoności
    stosuje regularyzacje: H + lambda * I.

    Parametry
    ---------
    max_iter : int
    tol : float
        Kryterium zatrzymania: ||grad||_2 < tol.
    reg : float
        Wspolczynnik regularyzacji Hessiana (damped Newton).
    """

    def __init__(self, max_iter=100, tol=1e-8, reg=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.loss_history = []
        self.theta = None

    def fit(self, X, y, loss_fn, grad_fn, hess_fn, theta_init=None):
        """
        Parametry
        ---------
        X : ndarray (n, p)
        y : ndarray (n,)
        loss_fn  : callable(X, y, theta) -> float
        grad_fn  : callable(X, y, theta) -> ndarray (p,)
        hess_fn  : callable(X, y, theta) -> ndarray (p, p)
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

            H = hess_fn(X, y, self.theta)
            H_reg = H + self.reg * np.eye(p)

            # Rozwiazanie ukladu: H_reg * delta = -grad
            delta = np.linalg.solve(H_reg, -grad)
            self.theta += delta

        return self
