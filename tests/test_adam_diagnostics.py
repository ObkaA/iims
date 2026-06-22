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
    plot_adam_diagnostics,
    plot_optimizer_comparison,
)


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

    def test_comparison_warns_when_adam_is_worse_than_baselines(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 2.0, 80)),
            "SGD": history(np.geomspace(10.0, 0.5, 80)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 80)),
        })
        self.assertEqual(result["verdict"], "WARNING")
        self.assertEqual(result["metrics"]["best_baseline"], "SGD")

    def test_comparison_is_ok_when_adam_is_best(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 0.1, 80)),
            "SGD": history(np.geomspace(10.0, 0.5, 80)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 80)),
        })
        self.assertEqual(result["verdict"], "OK")

    def test_comparison_plot_contains_four_charts(self):
        result = analyze_optimizer_comparison({
            "Adam": history(np.geomspace(10.0, 0.1, 40)),
            "SGD": history(np.geomspace(10.0, 0.5, 40)),
            "Gradient Descent": history(np.geomspace(10.0, 0.7, 40)),
        })
        figure = plot_optimizer_comparison(result, Figure(figsize=(10, 6)))
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(figure.axes[3].get_title(), "Final trend loss")


if __name__ == "__main__":
    unittest.main()
