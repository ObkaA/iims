"""
Newton-Raphson Method — second-order optimisation. Pure NumPy, from scratch.

Update rule:
    Solve:  (H + λI) · Δθ = ∇L    (linear system via LU decomposition)
    Apply:  θ_{t+1} = θ_t - α·Δθ

where:
    ∇L   gradient of loss,    shape (d,)
    H    Hessian of loss,     shape (d, d)
    λI   Tikhonov damping     (ensures H+λI ≻ 0, prevents singular solve)
    α    step size (1.0 = pure Newton, <1.0 = damped Newton)

Convergence:
    GD:     linear,    ‖θ_{t+1}-θ*‖ ≤ ρ·‖θ_t-θ*‖,    ρ<1
    Newton: quadratic, ‖θ_{t+1}-θ*‖ ≤ c·‖θ_t-θ*‖²
    → doubles correct decimal places each step!

Implementation detail:
    np.linalg.solve(H_reg, g) uses LAPACK dgesv (LU decomp) — O(d³).
    Never forms H⁻¹ explicitly (numerically unstable).
"""
import numpy as np
from .base import BaseOptimizer


class NewtonMethod(BaseOptimizer):
    """
    Damped Newton's Method with Tikhonov regularisation.

    Falls back to gradient descent if Hessian not available.
    """

    def __init__(self, learning_rate: float = 1.0, damping: float = 1e-4):
        super().__init__(learning_rate)
        self.damping = damping

    def step(
        self,
        params:    np.ndarray,
        gradients: np.ndarray,
        hessian:   np.ndarray | None = None,
        **_,
    ) -> np.ndarray:
        if hessian is not None:
            d     = len(params)
            H_reg = hessian + self.damping * np.eye(d, dtype=np.float64)
            try:
                # Solve (H + λI)·Δθ = ∇L  via LU factorisation
                delta = np.linalg.solve(H_reg, gradients.astype(np.float64))
                return params - self.learning_rate * delta
            except np.linalg.LinAlgError:
                pass   # singular → fall through to GD
        # Fallback: plain gradient step
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "Newton Method"
