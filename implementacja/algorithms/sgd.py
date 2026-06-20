"""
Stochastic Gradient Descent with Momentum — pure NumPy, from scratch.

Mini-batch SGD update:
    θ_{t+1} = θ_t - α · ∇L(θ_t; B_t)   where B_t is a random mini-batch

With Polyak momentum (1964):
    v_{t+1} = β·v_t + α·∇L(θ_t; B_t)
    θ_{t+1} = θ_t - v_{t+1}

Intuition: v is an exponential moving average of past gradients.
β=0.9 → effective look-back window ≈ 1/(1-β)=10 steps.
"""
import numpy as np
from .base import BaseOptimizer


class SGD(BaseOptimizer):
    """Mini-batch SGD with classical (Polyak) momentum."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum  = momentum
        self._velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, gradients: np.ndarray, **_) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(params, dtype=np.float64)
        # v  ← β·v + α·g
        self._velocity = self.momentum * self._velocity + self.learning_rate * gradients
        # θ  ← θ - v
        return params - self._velocity

    def _reset_state(self):
        self._velocity = None

    @property
    def name(self) -> str:
        return "SGD"
