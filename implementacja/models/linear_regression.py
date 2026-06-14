"""Linear regression — MSE loss."""
import numpy as np
from .base import BaseModel


class LinearRegressionModel(BaseModel):
    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self._forward(params, X)
        return float(np.mean((y_hat - y) ** 2))

    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = len(y)
        y_hat = self._forward(params, X)
        error = y_hat - y
        X_aug = np.c_[np.ones(m), X]
        return (2 / m) * X_aug.T @ error

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        return self._forward(self.params, X)

    def _forward(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_aug = np.c_[np.ones(m), X]
        return X_aug @ params

    @property
    def name(self) -> str:
        return "Linear Regression"

    @property
    def task(self) -> str:
        return "regression"
