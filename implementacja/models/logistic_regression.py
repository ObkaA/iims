"""Logistic regression — binary cross-entropy loss + Hessian for Newton's method."""
import numpy as np
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_aug = np.c_[np.ones(m), X]
        return self._sigmoid(X_aug @ params)

    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        m = len(y)
        p = self._forward(params, X)
        eps = 1e-12
        return -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = len(y)
        p = self._forward(params, X)
        error = p - y
        X_aug = np.c_[np.ones(m), X]
        return (1 / m) * X_aug.T @ error

    def hessian(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact Hessian: H = (1/m) X^T diag(p*(1-p)) X"""
        m = len(y)
        p = self._forward(params, X)
        X_aug = np.c_[np.ones(m), X]
        W = p * (1 - p)
        return (1 / m) * (X_aug.T * W) @ X_aug

    def loss_gradient(self, params, X=None, y=None):
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        return self.loss(params, X, y), self.gradient(params, X, y)

    def loss_gradient_hessian(self, params, X=None, y=None):
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        return self.loss(params, X, y), self.gradient(params, X, y), self.hessian(params, X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        return self._forward(self.params, X)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict_class(X) == y))

    def hessian(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Hessian of BCE loss: H = (1/m) X_aug^T diag(p*(1-p)) X_aug"""
        m = len(y)
        p = self._forward(params, X)
        X_aug = np.c_[np.ones(m), X]
        weights = p * (1 - p)  # (m,)
        return (1 / m) * (X_aug.T * weights) @ X_aug  # (k+1, k+1)

    def loss_gradient_hessian(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Returns (loss, gradient, hessian) — used by Newton Method."""
        return (
            self.loss(params, X, y),
            self.gradient(params, X, y),
            self.hessian(params, X, y),
        )

    @property
    def name(self) -> str:
        return "Logistic Regression"

    @property
    def task(self) -> str:
        return "classification"
