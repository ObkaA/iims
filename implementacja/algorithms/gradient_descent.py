"""Vanilla (Batch) Gradient Descent."""
import numpy as np
from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Full-batch gradient descent — uses all data every step."""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "Gradient Descent"
