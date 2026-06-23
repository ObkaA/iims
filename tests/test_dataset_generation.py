import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from data import DATASETS, generate_ill_conditioned_regression


class DatasetGenerationTests(unittest.TestCase):
    def test_every_generator_respects_selected_size(self):
        for name, generator in DATASETS.items():
            with self.subTest(dataset=name):
                X, y = generator(n_samples=137)
                self.assertEqual(X.shape[0], 137)
                self.assertEqual(y.shape, (137,))
                self.assertTrue(np.isfinite(X).all())
                self.assertTrue(np.isfinite(y).all())

    def test_generators_remain_reproducible(self):
        for name, generator in DATASETS.items():
            with self.subTest(dataset=name):
                X_first, y_first = generator(n_samples=50)
                X_second, y_second = generator(n_samples=50)
                np.testing.assert_array_equal(X_first, X_second)
                np.testing.assert_array_equal(y_first, y_second)

    def test_adam_demo_keeps_the_intended_feature_scale(self):
        X, y = generate_ill_conditioned_regression(n_samples=2000)
        self.assertEqual(X.shape, (2000, 1))
        self.assertGreater(float(X.std()), 8.0)
        self.assertLess(float(X.std()), 12.0)
        self.assertGreater(np.corrcoef(X[:, 0], y)[0, 1], 0.9)


if __name__ == "__main__":
    unittest.main()
