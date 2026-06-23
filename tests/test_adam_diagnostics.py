import sys
import unittest
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from visualization.adam_diagnostics import (
    analyze_adam_history,
    analyze_optimizer_comparison,
    build_parameter_loss_contours,
    plot_adam_diagnostics,
    plot_optimizer_comparison,
)
from models.linear_regression import LinearRegressionModel


def history(losses, gradient_scale=1.0):
    losses = np.asarray(losses, dtype=float)
    params = [np.array([3.0])]
    for index in range(len(losses)):
        params.append(np.array([3.0 / (index + 2)]))
    gradients = [np.array([gradient_scale / (index + 1)]) for index in range(len(losses))]
    return {"loss": list(losses), "params": params, "gradients": gradients}


class AdamDiagnosticsTests(unittest.TestCase):
    def test_converging_history_is_ok(self):
        result = analyze_adam_history(history(np.geomspace(10.0, 0.01, 80)))
        self.assertEqual(result["verdict"], "OK")
        self.assertGreater(result["metrics"]["loss_reduction"], 0.9)

    def test_diverging_history_is_failure(self):
        result = analyze_adam_history(history(np.geomspace(1.0, 100.0, 80)))
        self.assertEqual(result["verdict"], "FAILURE")

    def test_flat_history_with_large_gradient_is_warning(self):
        flat = history(np.ones(80), gradient_scale=10.0)
        flat["gradients"] = [np.array([10.0]) for _ in range(80)]
        result = analyze_adam_history(flat)
        self.assertEqual(result["verdict"], "WARNING")

    def test_plot_contains_four_diagnostics(self):
        diagnostic = analyze_adam_history(history(np.geomspace(5.0, 0.05, 40)))
        figure = plot_adam_diagnostics(diagnostic, Figure(figsize=(10, 6)))
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(figure.axes[2].get_ylabel(), "||Δθ|| (log)")

    def test_comparison_does_not_mark_healthy_adam_as_warning_when_baseline_wins(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 2.0, 80)),
            "SGD": history(np.geomspace(10.0, 0.5, 80)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 80)),
        })
        self.assertEqual(result["verdict"], "OK")
        self.assertEqual(result["metrics"]["comparison_winner"], "SGD")
        self.assertEqual(result["metrics"]["adam_rank"], 3)

    def test_health_analysis_uses_training_loss_not_evaluation_loss(self):
        training_loss = np.geomspace(10.0, 0.1, 80)
        run = history(training_loss, gradient_scale=1.0)
        run["eval_loss"] = list(np.geomspace(1.0, 100.0, 80))
        result = analyze_adam_history(run)
        self.assertEqual(result["verdict"], "OK")
        self.assertEqual(result["loss_source"], "mini-batch training")
        np.testing.assert_allclose(result["losses"], training_loss)

    def test_comparison_ranking_uses_common_evaluation_loss(self):
        adam = history(np.geomspace(10.0, 0.1, 80))
        sgd = history(np.geomspace(10.0, 1.0, 80))
        adam["eval_loss"] = list(np.geomspace(10.0, 2.0, 80))
        sgd["eval_loss"] = list(np.geomspace(10.0, 0.5, 80))
        adam["eval_iteration"] = list(range(0, 160, 2))
        sgd["eval_iteration"] = list(range(0, 160, 2))
        result = analyze_optimizer_comparison({"Adam": adam, "SGD": sgd})
        self.assertEqual(result["verdict"], "OK")
        self.assertEqual(result["metrics"]["comparison_winner"], "SGD")
        self.assertEqual(result["runs"]["Adam"]["comparison_loss_source"], "evaluation")
        self.assertEqual(result["runs"]["Adam"]["comparison_iterations"][-1], 158)

    def test_ranking_uses_shared_processed_data_budget(self):
        adam = history([10.0, 5.0, 1.0])
        gd = history([10.0, 1.0, 0.01])
        for run, samples_per_step in ((adam, 1), (gd, 10)):
            run["eval_loss"] = list(run["loss"])
            run["eval_iteration"] = [0, 1, 2]
            run["diagnostic_gradient_norm"] = [3.0, 2.0, 1.0]
            run["samples_per_step"] = samples_per_step
            run["train_size"] = 10
        result = analyze_optimizer_comparison({
            "Adam": adam,
            "Gradient Descent": gd,
        })
        self.assertAlmostEqual(result["metrics"]["shared_budget_epochs"], 0.3)
        self.assertEqual(result["metrics"]["comparison_winner"], "Adam")
        self.assertEqual(
            result["runs"]["Adam"]["gradient_source"],
            "shared training sample",
        )

    def test_comparison_is_ok_when_adam_is_best(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 0.1, 80)),
            "SGD": history(np.geomspace(10.0, 0.5, 80)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 80)),
        })
        self.assertEqual(result["verdict"], "OK")

    def test_near_equal_losses_are_reported_as_tie(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 0.7543, 80)),
            "SGD": history(np.geomspace(10.0, 0.7579, 80)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7534, 80)),
        })
        self.assertIsNone(result["metrics"]["comparison_winner"])
        self.assertTrue(result["metrics"]["comparison_label"].startswith("TIE:"))
        self.assertIn("Adam", result["metrics"]["comparison_winners"])

    def test_comparison_plot_contains_trajectory_and_diagnostics(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 0.1, 40)),
            "SGD": history(np.geomspace(10.0, 0.5, 40)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 40)),
        })
        figure = plot_optimizer_comparison(result, Figure(figsize=(10, 6)))
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(figure.axes[1].get_title(), "Parameter trajectory")
        self.assertEqual(figure.axes[2].get_title(), "Evaluation loss at 40.00 epochs")
        self.assertEqual(figure.axes[0].get_xlabel(), "Epoch")

    def test_multidimensional_trajectory_uses_shared_pca_projection(self):
        histories = {}
        for name, scale in (("Adam", 1.0), ("SGD", 0.7)):
            run = history(np.geomspace(10.0, 0.1, 40))
            run["params"] = [
                np.array([scale * t, scale * t ** 2, scale * np.sin(t)])
                for t in np.linspace(0.0, 1.0, 41)
            ]
            histories[name] = run
        result = analyze_optimizer_comparison(histories)
        self.assertEqual(result["trajectory"]["mode"], "PCA")
        self.assertEqual(result["trajectory"]["paths"]["Adam"].shape, (41, 2))

    def test_two_parameter_surface_marks_estimated_minimum(self):
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        y = 0.5 + 2.0 * X[:, 0]
        model = LinearRegressionModel()
        model.fit_data(X, y)
        histories = {
            "Adam": {
                "params": [np.array([0.0, 0.0]), np.array([0.5, 2.0])]
            }
        }
        surface = build_parameter_loss_contours(model, X, y, histories)
        self.assertIsNotNone(surface)
        np.testing.assert_allclose(surface["minimum"], [0.5, 2.0], atol=0.08)


if __name__ == "__main__":
    unittest.main()
