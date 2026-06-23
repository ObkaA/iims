import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from algorithms.gradient_descent import GradientDescent
from models.linear_regression import LinearRegressionModel
from training_engine import TrainingWorker


class TrainingEngineDiagnosticsTests(unittest.TestCase):
    def _run_minibatch_gd(self, X, y, seed):
        model = LinearRegressionModel()
        model.fit_data(X, y)
        optimizer = GradientDescent(learning_rate=0.05)
        optimizer.history["params"].append(model.params.copy())
        worker = TrainingWorker(
            model,
            optimizer,
            X,
            y,
            n_iterations=12,
            batch_size=5,
            emit_every=12,
            random_seed=seed,
        )
        worker._run()
        return model.params.copy(), np.asarray(optimizer.history["loss"])

    def test_minibatch_schedule_is_reproducible(self):
        X = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)
        y = 1.7 * X[:, 0] - 0.2
        params_a, losses_a = self._run_minibatch_gd(X, y, seed=123)
        params_b, losses_b = self._run_minibatch_gd(X, y, seed=123)
        np.testing.assert_allclose(params_a, params_b)
        np.testing.assert_allclose(losses_a, losses_b)

    def test_evaluation_history_is_observational_only(self):
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 0.5

        worker_model = LinearRegressionModel()
        worker_model.fit_data(X, y)
        worker_optimizer = GradientDescent(learning_rate=0.05)
        worker_optimizer.history["params"].append(worker_model.params.copy())
        worker = TrainingWorker(
            worker_model,
            worker_optimizer,
            X,
            y,
            X_eval=X[::2],
            y_eval=y[::2],
            n_iterations=5,
            batch_size=len(y),
            emit_every=5,
        )
        worker._run()

        reference_model = LinearRegressionModel()
        reference_model.fit_data(X, y)
        reference_optimizer = GradientDescent(learning_rate=0.05)
        params = reference_model.params.copy()
        for _ in range(5):
            _, gradient = reference_model.loss_gradient(params, X, y)
            params = reference_optimizer.step(params, gradient)

        np.testing.assert_allclose(worker_model.params, params)
        self.assertEqual(len(worker_optimizer.history["eval_loss"]), 5)
        self.assertEqual(worker_optimizer.history["eval_iteration"], list(range(5)))
