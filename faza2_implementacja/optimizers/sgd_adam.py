import numpy as np


class Adam:
    """
    Mini-batch SGD z optymalizatorem Adam (Adaptive Moment Estimation).

    Algorytm (Kingma & Ba, 2015):
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t         # moment 1. rzedu
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2       # moment 2. rzedu
        m_hat = m_t / (1 - beta1^t)                        # korekcja biasu
        v_hat = v_t / (1 - beta2^t)                        # korekcja biasu
        theta -= lr * m_hat / (sqrt(v_hat) + eps)

    Parametry
    ---------
    lr : float          Krok uczenia (domyslnie 0.001).
    beta1 : float       Wspolczynnik dla momentu 1. rzedu (domyslnie 0.9).
    beta2 : float       Wspolczynnik dla momentu 2. rzedu (domyslnie 0.999).
    eps : float         Stabilizator numeryczny (domyslnie 1e-8).
    max_iter : int      Liczba epok.
    batch_size : int    Rozmiar mini-batcha.
    tol : float         Kryterium zatrzymania (norma gradientu pelnego batcha).
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 max_iter=200, batch_size=32, tol=1e-6, seed=42):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed
        self.loss_history = []
        self.theta = None

    def fit(self, X, y, loss_fn, grad_fn, theta_init=None):
        """
        Parametry
        ---------
        X : ndarray (n, p)
        y : ndarray (n,)
        loss_fn : callable(X, y, theta) -> float  (uzywany dla calego zbioru)
        grad_fn : callable(X_batch, y_batch, theta) -> ndarray (p,)
        theta_init : ndarray lub None
        """
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        self.theta = np.zeros(p) if theta_init is None else theta_init.copy()
        self.loss_history = []

        m = np.zeros(p)   # moment 1. rzedu
        v = np.zeros(p)   # moment 2. rzedu
        t = 0             # licznik krokow

        for epoch in range(self.max_iter):
            # Strata na pelnym zbiorze (do historii zbieznosci)
            self.loss_history.append(loss_fn(X, y, self.theta))

            # Kryterium zatrzymania na pelnym zbiorze
            full_grad = grad_fn(X, y, self.theta)
            if np.linalg.norm(full_grad) < self.tol:
                break

            # Shuffle i mini-batche
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                X_b, y_b = X[idx], y[idx]

                g = grad_fn(X_b, y_b, self.theta)
                t += 1

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)

                self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return self
