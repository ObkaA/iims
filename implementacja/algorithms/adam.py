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

WHY ADAM SOMETIMES FAILS (see adam_failure_scenarios.py for live demos):
────────────────────────────────────────────────────────────────────────
1. NON-STATIONARY OBJECTIVES: v accumulates a long memory of past gradient
   magnitudes. If the loss landscape changes suddenly (e.g. curriculum
   learning, changing data distribution), v̂ is stale → wrong scale.

2. SPARSE + INFREQUENT GRADIENTS with WRONG β₂: If a feature fires rarely,
   its v accumulates near zero → huge effective step when it does fire.
   High β₂ (0.999) amplifies this with long memory.

3. SHARP MINIMA / GENERALIZATION GAP: Adam's adaptive step can overshoot
   or settle into "wide" flat minima. SGD with momentum often finds
   "sharp" minima that generalize better (Wilson et al., 2017).

4. VERY SMALL DATASETS: With few samples, the stochastic noise in gradients
   is low. Adam's correction for variance becomes unnecessary overhead and
   can introduce oscillation around the optimum.

5. POORLY TUNED β₁ ≥ 1 or α TOO LARGE: m̂ = m/(1-β₁ᵗ) blows up if
   β₁ is set too high (≥ 0.999), causing divergence. Unlike SGD where
   the learning rate directly bounds the step, Adam's effective step is
   α/(√v̂ + ε) and can be large even with small α.
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
