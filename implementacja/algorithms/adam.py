"""Adam optimizer (Adaptive Moment Estimation)."""
import numpy as np
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam — adaptive learning rates with first and second moment estimates."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t: int = 0

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * gradients
        self._v = self.beta2 * self._v + (1 - self.beta2) * gradients ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def _reset_state(self):
        self._m = None
        self._v = None
        self._t = 0

    @property
    def name(self) -> str:
        return "Adam"
