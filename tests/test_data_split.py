import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from data import (
    DATASETS,
    DATASET_TASKS,
    standardize_from_training,
    train_test_split,
)


class DataSplitTests(unittest.TestCase):
    def test_every_builtin_dataset_declares_a_valid_task(self):
        self.assertEqual(set(DATASETS), set(DATASET_TASKS))
        self.assertTrue(set(DATASET_TASKS.values()) <= {"regression", "classification"})

    def test_classification_split_preserves_both_classes(self):
        X = np.arange(80, dtype=float).reshape(40, 2)
        y = np.array([0] * 20 + [1] * 20)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_fraction=0.25, task="classification", seed=7
        )

        self.assertEqual(X_train.shape[0], 30)
        self.assertEqual(X_test.shape[0], 10)
        self.assertEqual(set(y_train), {0, 1})
        self.assertEqual(set(y_test), {0, 1})
        self.assertEqual(np.bincount(y_test.astype(int)).tolist(), [5, 5])

    def test_split_is_reproducible(self):
        X = np.arange(60, dtype=float).reshape(30, 2)
        y = np.arange(30, dtype=float)

        first = train_test_split(X, y, test_fraction=0.2, seed=42)
        second = train_test_split(X, y, test_fraction=0.2, seed=42)

        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)

    def test_standardization_uses_training_statistics_only(self):
        X_train = np.array([[0.0], [2.0]])
        X_test = np.array([[10.0]])
        X_full = np.vstack([X_train, X_test])

        train_scaled, test_scaled, full_scaled = standardize_from_training(
            X_train, X_test, X_full
        )

        np.testing.assert_array_equal(train_scaled[:, 0], [-1.0, 1.0])
        np.testing.assert_array_equal(test_scaled[:, 0], [9.0])
        np.testing.assert_array_equal(full_scaled[:, 0], [-1.0, 1.0, 9.0])

    def test_class_with_one_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs at least 2 samples"):
            train_test_split(
                np.arange(8).reshape(4, 2),
                np.array([0, 0, 0, 1]),
                task="classification",
            )


if __name__ == "__main__":
    unittest.main()
