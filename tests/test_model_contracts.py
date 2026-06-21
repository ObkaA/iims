import ast
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from models import MODELS


class ModelContractTests(unittest.TestCase):
    def test_registered_models_fit_predict_and_declare_task(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        for name, model_class in MODELS.items():
            with self.subTest(model=name):
                model = model_class()
                model.fit_data(X, y)
                predictions = model.predict(X)
                self.assertEqual(predictions.shape, y.shape)
                self.assertIn(model.task, {"regression", "classification"})

    def test_logistic_model_has_no_duplicate_method_definitions(self):
        source_path = PROJECT_ROOT / "implementacja" / "models" / "logistic_regression.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        model_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LogisticRegressionModel"
        )
        method_names = [
            node.name for node in model_class.body if isinstance(node, ast.FunctionDef)
        ]

        self.assertEqual(len(method_names), len(set(method_names)))


if __name__ == "__main__":
    unittest.main()
