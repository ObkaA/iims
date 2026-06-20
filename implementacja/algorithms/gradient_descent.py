"""
Vanilla Batch Gradient Descent — implemented from first principles, pure NumPy.

Update rule:
    θ_{t+1} = θ_t - α · ∇L(θ_t)

Convergence for L-smooth convex functions:
    L(θ_t) - L(θ*) ≤ ‖θ_0 - θ*‖² / (2αt)   →  O(1/t) rate
"""
import numpy as np
from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Full-batch Gradient Descent. Uses ALL data every step."""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradients: np.ndarray, **_) -> np.ndarray:
        # θ_{t+1} = θ_t - α·∇L
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "Gradient Descent"
