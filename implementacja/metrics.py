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
