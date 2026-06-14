"""Alternating Least Squares (ALS) optimizer for Matrix Factorization."""
from __future__ import annotations
import numpy as np
from .base import BaseOptimizer


class ALS(BaseOptimizer):
    """
    Alternating Least Squares — not a gradient-based method.
    Alternates between solving exactly for user factors (U) and item factors (V).
    Closed-form update: U_i = (V^T V + λI)^{-1} V^T r_i
    Used here as a special-purpose optimizer for MF — doesn't follow the
    standard step(params, grad) interface, but is registered for comparison plots.
    """

    def __init__(self, learning_rate: float = 0.01, reg_lambda: float = 0.1):
        super().__init__(learning_rate)
        self.reg_lambda = reg_lambda
        self._als_loss_history: list[float] = []

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Fallback GD step — real ALS is called via als_update() in MF model.
        This allows ALS to appear in comparison plots alongside GD/SGD/Adam.
        """
        return params - self.learning_rate * gradients

    def als_update(
        self,
        factor_matrix: np.ndarray,   # shape (n, k) — the factor being updated
        fixed_matrix: np.ndarray,    # shape (m, k) — the factor held fixed
        R: np.ndarray,               # shape (n, m) — ratings matrix (0 where missing)
        mask: np.ndarray,            # shape (n, m) — 1 where rating exists
        reg: float | None = None,
    ) -> np.ndarray:
        """
        Closed-form ALS update for one factor matrix.
        For each entity i:  f_i = (V_i^T V_i + λI)^{-1} V_i^T r_i
        where V_i are the rows of fixed_matrix for items rated by user i.
        """
        lam = reg if reg is not None else self.reg_lambda
        n, k = factor_matrix.shape
        updated = np.zeros_like(factor_matrix)
        I = np.eye(k)
        for i in range(n):
            rated_idx = np.where(mask[i] > 0)[0]
            if len(rated_idx) == 0:
                updated[i] = factor_matrix[i]
                continue
            V_i = fixed_matrix[rated_idx]       # (n_rated, k)
            r_i = R[i, rated_idx]               # (n_rated,)
            A = V_i.T @ V_i + lam * I           # (k, k)
            b = V_i.T @ r_i                     # (k,)
            updated[i] = np.linalg.solve(A, b)
        return updated

    @property
    def name(self) -> str:
        return "ALS"
