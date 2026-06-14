"""Stochastic Gradient Descent with optional momentum."""
import numpy as np
from .base import BaseOptimizer


class SGD(BaseOptimizer):
    """SGD — updates parameters using mini-batches."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(params)
        self._velocity = self.momentum * self._velocity + self.learning_rate * gradients
        return params - self._velocity

    def _reset_state(self):
        self._velocity = None

    @property
    def name(self) -> str:
        return "SGD"
