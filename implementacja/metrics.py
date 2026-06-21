"""Model evaluation metrics implemented with NumPy."""
import numpy as np


def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return ``[[TN, FP], [FN, TP]]`` for binary labels 0 and 1."""
    actual = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted labels must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot calculate a confusion matrix for an empty dataset.")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Binary confusion matrix accepts only labels 0 and 1.")

    return np.array([
        [np.sum((actual == 0) & (predicted == 0)), np.sum((actual == 0) & (predicted == 1))],
        [np.sum((actual == 1) & (predicted == 0)), np.sum((actual == 1) & (predicted == 1))],
    ], dtype=np.int64)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return ``(R², RMSE)`` for regression predictions."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot calculate regression scores for an empty dataset.")
    residual_sum = float(np.sum((actual - predicted) ** 2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else float(residual_sum == 0)
    rmse = float(np.sqrt(residual_sum / actual.size))
    return r2, rmse
