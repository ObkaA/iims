"""
Adam — Adaptive Moment Estimation (Kingma & Ba, 2014). Pure NumPy, from scratch.

Algorithm (per Kingma & Ba, Algorithm 1):
    Initialise: m=0, v=0, t=0

    Each step:
        t  ← t + 1
        m  ← β₁·m + (1-β₁)·g          # biased 1st moment (mean)
        v  ← β₂·v + (1-β₂)·g²         # biased 2nd moment (uncentered variance)
        m̂  ← m / (1 - β₁ᵗ)            # bias correction
        v̂  ← v / (1 - β₂ᵗ)
        θ  ← θ - α·m̂ / (√v̂ + ε)      # adaptive update

WHAT THE ADAM CHECK DIAGNOSES ON A REAL TRAINING RUN:
────────────────────────────────────────────────────────────────────────
1. LOSS TREND: Adam's mini-batch training loss and its smoothed trend.
2. GRADIENT NORM: whether the optimization signal is shrinking.
3. PARAMETER-STEP NORM: whether Adam settles near a solution or keeps jumping.
4. REGRESSION FROM THE BEST POINT: whether the final iterations undo progress.
5. NUMERICAL HEALTH: NaN, infinity, explosion, stagnation, and instability.

The heuristic health diagnosis uses the exact model, dataset and optimizer
history produced by the Optimization tab. The separate optimizer benchmark
uses one shared held-out sample and a common processed-data budget expressed
as epochs. It does not change Adam's health verdict and does not run
separate synthetic failure scenarios.
"""
import numpy as np
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam optimiser — first + second adaptive moment estimation.

    Default hyperparameters from the paper:
        α=0.001, β₁=0.9, β₂=0.999, ε=1e-8
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1:         float = 0.9,
        beta2:         float = 0.999,
        epsilon:       float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self._m: np.ndarray | None = None   # 1st moment vector
        self._v: np.ndarray | None = None   # 2nd moment vector
        self._t: int = 0

    def step(self, params: np.ndarray, gradients: np.ndarray, **_) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(params, dtype=np.float64)
            self._v = np.zeros_like(params, dtype=np.float64)

        self._t += 1
        g = gradients.astype(np.float64)

        # Biased moment updates
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * g
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * g ** 2

        # Bias correction: without this, early updates are tiny (m≈0, v≈0)
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)

        # Adaptive per-parameter update
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def _reset_state(self):
        self._m = None
        self._v = None
        self._t = 0

    @property
    def name(self) -> str:
        return "Adam"
