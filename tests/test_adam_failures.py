import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from algorithms.adam_failures import (
    scenario_bad_hyperparams,
    scenario_nonstationary,
    scenario_sharp_vs_flat,
    scenario_sparse_gradients,
)


class AdamFailureScenarioTests(unittest.TestCase):
    def test_sharp_flat_plot_marks_real_minima_and_distinct_basins(self):
        data = scenario_sharp_vs_flat()
        minima = data["minima"]

        self.assertEqual([item["theta"] for item in minima], [-1.0, 1.0])
        self.assertGreater(minima[0]["curvature"], minima[1]["curvature"])
        self.assertLess(minima[0]["loss"], minima[1]["loss"])
        self.assertAlmostEqual(data["results"]["Adam"]["thetas"][-1], -1.0, places=6)
        self.assertAlmostEqual(data["results"]["SGD"]["thetas"][-1], 1.0, places=6)

    def test_scenario_outputs_are_finite(self):
        for scenario in (
            scenario_sparse_gradients,
            scenario_nonstationary,
            scenario_sharp_vs_flat,
            scenario_bad_hyperparams,
        ):
            with self.subTest(scenario=scenario.__name__):
                data = scenario()
                for result in data["results"].values():
                    self.assertTrue(np.isfinite(result["losses"]).all())

    def test_nonstationary_shorter_second_moment_memory_recovers_closer(self):
        data = scenario_nonstationary()
        default_error = abs(data["results"]["Adam"]["thetas"][-1] + 5)
        short_memory_error = abs(data["results"]["Adam β₂=0.9"]["thetas"][-1] + 5)

        self.assertLess(short_memory_error, default_error)


if __name__ == "__main__":
    unittest.main()
