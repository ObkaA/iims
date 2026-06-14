"""Base class for all optimization algorithms."""
from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """Abstract base optimizer — all algorithms share this interface."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.history: dict = {
            "loss": [],
            "params": [],
            "gradients": [],
            "iteration": [],
        }
        self.iteration = 0

    @abstractmethod
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform one optimization step. Returns updated params."""
        ...

    def reset(self):
        self.history = {"loss": [], "params": [], "gradients": [], "iteration": []}
        self.iteration = 0
        self._reset_state()

    def _reset_state(self):
        """Override in subclasses to reset algorithm-specific state."""
        pass

    def record(self, loss: float, params: np.ndarray, gradients: np.ndarray):
        self.history["loss"].append(float(loss))
        self.history["params"].append(params.copy())
        self.history["gradients"].append(gradients.copy())
        self.history["iteration"].append(self.iteration)
        self.iteration += 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def current_loss(self) -> float:
        return self.history["loss"][-1] if self.history["loss"] else float("inf")
