import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from algorithms import Adam, GradientDescent, SGD
from data import generate_ill_conditioned_regression, train_test_split
from models.linear_regression import LinearRegressionModel
from training_engine import TrainingWorker, optimizer_steps_for_epochs


class AdamDemoTests(unittest.TestCase):
    def test_adam_wins_the_documented_ill_conditioned_configuration(self):
        X, y = generate_ill_conditioned_regression(n_samples=2000, seed=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_fraction=0.2, task="regression", seed=42
        )
        losses = {}
        train_size = len(y_train)
        for optimizer_class in (Adam, SGD, GradientDescent):
            model = LinearRegressionModel()
            model.fit_data(X_train, y_train)
            optimizer = optimizer_class(learning_rate=0.01)
            optimizer.history["params"].append(model.params.copy())
            batch_size = train_size if optimizer_class is GradientDescent else 32
            worker = TrainingWorker(
                model,
                optimizer,
                X_train,
                y_train,
                X_eval=X_test,
                y_eval=y_test,
                n_iterations=optimizer_steps_for_epochs(
                    10, train_size, batch_size
                ),
                batch_size=batch_size,
                emit_every=10000,
                random_seed=42,
            )
            worker._run()
            losses[optimizer.name] = model.loss(model.params, X_test, y_test)

        self.assertLess(losses["Adam"], losses["SGD"] * 0.9)
        self.assertLess(losses["Adam"], losses["Gradient Descent"])
