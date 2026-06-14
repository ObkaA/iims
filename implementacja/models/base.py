"""Abstract base class for ML models."""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self):
        self.params: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit_data(self, X: np.ndarray, y: np.ndarray):
        """Store training data and initialise parameters."""
        self._X = X
        self._y = y
        n_features = X.shape[1] if X.ndim > 1 else 1
        self.params = np.zeros(n_features + 1)  # +1 for bias

    @abstractmethod
    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float: ...

    @abstractmethod
    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def loss_gradient(self, params, X=None, y=None):
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        return self.loss(params, X, y), self.gradient(params, X, y)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def task(self) -> str: ...  # 'regression' | 'classification'
