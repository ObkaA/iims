"""Newton's Method optimizer — uses Hessian for second-order updates."""
from __future__ import annotations
import numpy as np
from .base import BaseOptimizer


class NewtonMethod(BaseOptimizer):
    """
    Newton's Method: θ ← θ - H⁻¹ · ∇L
    Uses the inverse Hessian for faster convergence near optimum.
    Regularization (damping) prevents singular Hessian issues.
    """

    def __init__(self, learning_rate: float = 1.0, damping: float = 1e-4):
        super().__init__(learning_rate)
        self.damping = damping  # Tikhonov regularization λI added to H

    def step(self, params: np.ndarray, gradients: np.ndarray, hessian: np.ndarray | None = None) -> np.ndarray:
        """
        If a Hessian is supplied (by model.hessian()), use it.
        Otherwise falls back to gradient descent (diagonal approximation).
        """
        if hessian is not None and hessian.shape == (len(params), len(params)):
            H_reg = hessian + self.damping * np.eye(len(params))
            try:
                delta = np.linalg.solve(H_reg, gradients)
                return params - self.learning_rate * delta
            except np.linalg.LinAlgError:
                pass  # singular — fall back to GD
        # Fallback: gradient descent
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "Newton Method"
