"""Ridge regression — MSE loss with L2 regularisation."""
import numpy as np

from .base import BaseModel


class RidgeRegressionModel(BaseModel):
    """Linear regression with an L2 penalty on weights (not on the bias)."""

    def __init__(self, regularization: float = 1.0):
        super().__init__()
        if regularization < 0:
            raise ValueError("Regularization must be non-negative.")
        self.regularization = float(regularization)

    @staticmethod
    def _augment(X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X]

    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        error = self._augment(X) @ params - y
        mse = np.mean(error ** 2)
        penalty = self.regularization * np.sum(params[1:] ** 2)
        return float(mse + penalty)

    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        error = X_aug @ params - y
        gradient = (2 / len(y)) * X_aug.T @ error
        regularization_gradient = 2 * self.regularization * params.copy()
        regularization_gradient[0] = 0.0
        return gradient + regularization_gradient

    def hessian(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        regularization_matrix = 2 * self.regularization * np.eye(X_aug.shape[1])
        regularization_matrix[0, 0] = 0.0
        return (2 / len(y)) * X_aug.T @ X_aug + regularization_matrix

    def loss_gradient_hessian(self, params, X=None, y=None):
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        return (
            self.loss(params, X, y),
            self.gradient(params, X, y),
            self.hessian(params, X, y),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        return self._augment(X) @ self.params

    @property
    def name(self) -> str:
        return "Ridge Regression"

    @property
    def task(self) -> str:
        return "regression"
