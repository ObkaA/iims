"""
Alternating Least Squares (ALS) — closed-form, pure NumPy, from scratch.

Problem: given sparse R ∈ ℝ^{n×m}, find U∈ℝ^{n×k}, V∈ℝ^{m×k} s.t. R ≈ U·Vᵀ

Objective (Frobenius on observed entries Ω):
    min_{U,V}  Σ_{(i,j)∈Ω} (R_{ij} - uᵢ·vⱼ)² + λ(‖U‖_F² + ‖V‖_F²)

ALS trick: the joint problem is non-convex, BUT fixing V the sub-problem in U is:
    min_{uᵢ}  Σ_{j∈Ωᵢ} (R_{ij} - uᵢ·vⱼ)² + λ‖uᵢ‖²

This is ridge regression with closed-form solution:
    uᵢ = (Vᵢᵀ Vᵢ + λI)⁻¹ Vᵢᵀ rᵢ

Similarly for vⱼ when U is fixed. ALS alternates these two solves.
Each step is guaranteed to decrease the objective → monotone convergence.
"""
import numpy as np
from .base import BaseOptimizer


class ALS(BaseOptimizer):
    """
    ALS for Matrix Factorization.
    Satisfies BaseOptimizer interface (falls back to GD if used in regression panel).
    Real ALS logic lives in MatrixFactorization._train().
    """

    def __init__(self, learning_rate: float = 0.01, reg_lambda: float = 0.1):
        super().__init__(learning_rate)
        self.reg_lambda = reg_lambda

    def step(self, params: np.ndarray, gradients: np.ndarray, **_) -> np.ndarray:
        # Fallback: standard GD (used when ALS is selected in the regression panel)
        return params - self.learning_rate * gradients

    @staticmethod
    def solve_one(F: np.ndarray, r: np.ndarray, lam: float) -> np.ndarray:
        """
        Ridge regression closed-form for one entity:
            A = Fᵀ F + λI
            b = Fᵀ r
            x = solve(A, b)

        Args:
            F:   (|Ω|, k) sub-matrix of fixed factors for observed entries
            r:   (|Ω|,)   observed ratings / interactions
            lam: λ regularisation strength

        Returns:
            x: (k,) updated factor vector
        """
        k   = F.shape[1]
        A   = F.T @ F + lam * np.eye(k, dtype=np.float64)
        b   = F.T @ r
        return np.linalg.solve(A, b)

    @property
    def name(self) -> str:
        return "ALS"
