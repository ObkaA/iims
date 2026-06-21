import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from metrics import binary_confusion_matrix, regression_scores


class MetricsTests(unittest.TestCase):
    def test_binary_confusion_matrix_layout(self):
        actual = np.array([0, 0, 0, 1, 1, 1])
        predicted = np.array([0, 1, 0, 1, 0, 1])

        matrix = binary_confusion_matrix(actual, predicted)

        np.testing.assert_array_equal(matrix, [[2, 1], [1, 2]])

    def test_rejects_non_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "only labels 0 and 1"):
            binary_confusion_matrix(np.array([0, 2]), np.array([0, 1]))

    def test_rejects_different_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_confusion_matrix(np.array([0, 1]), np.array([0]))

    def test_regression_scores_for_perfect_predictions(self):
        r2, rmse = regression_scores(
            np.array([1.0, 2.0, 4.0]),
            np.array([1.0, 2.0, 4.0]),
        )

        self.assertEqual(r2, 1.0)
        self.assertEqual(rmse, 0.0)

    def test_regression_scores_match_known_values(self):
        r2, rmse = regression_scores(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 2.0]),
        )

        self.assertAlmostEqual(r2, 0.5)
        self.assertAlmostEqual(rmse, np.sqrt(1 / 3))


if __name__ == "__main__":
    unittest.main()
