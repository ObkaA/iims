"""
adam_failures.py — Four concrete scenarios where Adam underperforms.
All implemented from scratch with NumPy only.

Used by the "⚠ Adam Failures" tab in main_window.py.
Each function returns a dict with everything needed to render explanatory plots.
"""
from __future__ import annotations
import numpy as np
from .adam import Adam
from .gradient_descent import GradientDescent
from .sgd import SGD


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run optimiser on a 1-D function for n steps
# ─────────────────────────────────────────────────────────────────────────────
def _run_1d(opt, grad_fn, loss_fn, theta0: float, n: int):
    params = np.array([theta0], dtype=np.float64)
    opt.history["params"].append(params.copy())
    losses, thetas = [], []
    for _ in range(n):
        g = grad_fn(params)
        params = opt.step(params, g)
        opt.record(loss_fn(params), params, g)
        losses.append(float(loss_fn(params)))
        thetas.append(float(params[0]))
    return np.array(losses), np.array(thetas)


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Sparse / Infrequent Gradients
# ═════════════════════════════════════════════════════════════════════════════
def scenario_sparse_gradients(n_steps: int = 400):
    """
    Problem: 2-D function where coordinate x₁ receives a gradient only
    once every K steps (simulating a rare feature in NLP/sparse data).

    Adam normalizes each coordinate by its recent squared-gradient scale.
    On a sparse coordinate that normalization can prevent updates from
    shrinking proportionally to a small raw gradient near the optimum.

    Mathematical formulation:
        f(x) = x₀² + x₁²
        g₀ = 2x₀  (every step)
        g₁ = 2x₁  (every K steps only, else 0)
    """
    K      = 20          # gradient for x₁ fires once every K steps
    lr     = 0.1
    x0_gt  = np.array([3.0, 3.0])   # start far from origin

    results = {}
    for name, opt_cls, kwargs in [
        ("Adam",             Adam,             {"learning_rate": lr, "beta2": 0.999}),
        ("Gradient Descent", GradientDescent,  {"learning_rate": lr}),
        ("SGD",              SGD,              {"learning_rate": lr, "momentum": 0.9}),
    ]:
        opt    = opt_cls(**kwargs)
        params = x0_gt.copy()
        opt.history["params"].append(params.copy())
        losses, traj = [], [params.copy()]
        for t in range(n_steps):
            g = np.array([2.0 * params[0], 0.0])   # x₁ gradient = 0
            if t % K == 0:
                g[1] = 2.0 * params[1]              # sparse: fire every K steps
            g = g.astype(np.float64)
            params = opt.step(params, g)
            loss = params[0] ** 2 + params[1] ** 2
            opt.record(loss, params, g)
            losses.append(loss)
            traj.append(params.copy())
        results[name] = {
            "losses": np.array(losses),
            "traj":   np.array(traj),
        }

    return {
        "title":       "Scenario 1 — Sparse Gradients",
        "subtitle":    f"θ₁ receives a gradient only once every {K} steps",
        "why_fails":   (
            "Adam divides each coordinate by the square root of its recent squared gradients.\n"
            "For a rarely active coordinate this normalization can keep the update close to α,\n"
            "even after the raw gradient has become small. Sparse corrections may therefore\n"
            "oscillate around the optimum instead of shrinking smoothly.\n\n"
            "In this run Adam converges well, but momentum SGD reaches a smaller residual error.\n"
            "The issue is not an infinite step: ε and bias correction keep the update finite."
        ),
        "results":     results,
        "optimum":     np.array([0.0, 0.0]),
        "x_range":     (-4, 4),
        "y_range":     (-4, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Non-Stationary Objective (sudden landscape change)
# ═════════════════════════════════════════════════════════════════════════════
def scenario_nonstationary(n_steps: int = 600):
    """
    Problem: 1-D loss function whose minimum SHIFTS at step n_steps//2.

        Phase 1 (t < T):   f(θ) = (θ - 5)²     minimum at θ=5
        Phase 2 (t ≥ T):   f(θ) = (θ + 5)²     minimum shifts to θ=-5

    Why Adam fails here:
        Adam accumulates v with a long β₂=0.999 memory of PHASE-1 gradients.
        When the landscape flips, m initially retains the old signed direction,
        while v retains the old gradient scale and can reduce new updates.

        GD / SGD adapt immediately: no gradient memory.

    This represents: distributional shift, curriculum changes, online learning.
    """
    T   = n_steps // 2
    lr  = 0.05

    def loss_fn(theta, t):
        target = 5.0 if t < T else -5.0
        return (theta - target) ** 2

    def grad_fn(theta, t):
        target = 5.0 if t < T else -5.0
        return np.array([2.0 * (theta - target)])

    results = {}
    for name, opt_cls, kwargs in [
        ("Adam",             Adam,            {"learning_rate": lr, "beta2": 0.999}),
        ("Adam β₂=0.9",     Adam,            {"learning_rate": lr, "beta2": 0.9}),
        ("Gradient Descent", GradientDescent, {"learning_rate": lr}),
    ]:
        opt    = opt_cls(**kwargs)
        params = np.array([0.0])
        opt.history["params"].append(params.copy())
        losses, thetas, phase_marker = [], [], []
        for t in range(n_steps):
            g      = grad_fn(params[0], t)
            params = opt.step(params, g)
            loss   = loss_fn(params[0], t)
            opt.record(loss, params, g)
            losses.append(loss)
            thetas.append(float(params[0]))
            phase_marker.append(0 if t < T else 1)
        results[name] = {
            "losses": np.array(losses),
            "thetas": np.array(thetas),
            "phase":  phase_marker,
        }

    return {
        "title":      "Scenario 2 — Non-Stationary Objective",
        "subtitle":   f"Minimum shifts from +5 to -5 at step {T}",
        "why_fails":  (
            "After the minimum moves, Adam carries stale optimizer state from phase 1.\n"
            "The first moment m initially retains the old signed direction, while the second\n"
            "moment v retains old gradient magnitudes and can reduce the effective step.\n\n"
            "With β₂=0.9 the scale estimate forgets the past faster than with β₂=0.999.\n"
            "GD follows the new gradient immediately because it stores no moment estimates."
        ),
        "results":    results,
        "phase_step": T,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Sharp vs Flat Minima (generalisation gap)
# ═════════════════════════════════════════════════════════════════════════════
def scenario_sharp_vs_flat(n_steps: int = 800):
    """
    A tilted double-well with two exact minima:
        sharp/lower training minimum at θ=-1, curvature f''(-1)=3
        flat/higher training minimum at θ=+1, curvature f''(+1)=1

    Starting at θ=-2, Adam and GD settle in the sharp basin. SGD momentum
    crosses the barrier and settles in the flatter basin. This demonstrates
    optimizer-dependent implicit bias; it does not itself measure test error.
    """
    def loss_fn(theta):
        t = float(theta[0]) if hasattr(theta, '__len__') else float(theta)
        return t**4 / 4 - t**3 / 6 - t**2 / 2 + 0.5 * t + 0.65

    def grad_fn(theta):
        t = float(theta[0])
        return np.array([(t + 1) * (t - 0.5) * (t - 1)])

    lr = 0.05
    theta_start = -2.0

    results = {}
    for name, opt_cls, kwargs in [
        ("Adam",   Adam, {"learning_rate": lr}),
        ("SGD",    SGD,  {"learning_rate": lr, "momentum": 0.85}),
        ("GD",     GradientDescent, {"learning_rate": lr}),
    ]:
        losses, thetas = _run_1d(opt_cls(**kwargs), grad_fn, loss_fn, theta_start, n_steps)
        results[name] = {"losses": losses, "thetas": thetas}

    # Loss landscape for plotting
    t_grid = np.linspace(-2.2, 2.2, 500)
    l_grid = np.array([loss_fn(t) for t in t_grid])
    minima = [
        {"theta": -1.0, "loss": loss_fn(-1.0), "curvature": 3.0, "label": "Sharp minimum"},
        {"theta": 1.0, "loss": loss_fn(1.0), "curvature": 1.0, "label": "Flat minimum"},
    ]

    return {
        "title":      "Scenario 3 — Sharp vs Flat Minima",
        "subtitle":   "Same landscape, different optimizer bias",
        "why_fails":  (
            "The marked points are real stationary minima: θ=-1 is three times sharper\n"
            "than θ=+1 (curvature 3 versus 1). Adam and GD remain in the sharp, lower-loss\n"
            "basin; SGD momentum crosses the barrier and ends in the flatter basin.\n\n"
            "A flatter minimum can be more robust to parameter perturbations, but this toy\n"
            "plot does not prove better test accuracy. It shows optimizer-dependent bias.\n\n"
            "Related discussion: Wilson et al., 2017, 'The Marginal Value of Adaptive Gradient Methods'."
        ),
        "results":    results,
        "t_grid":     t_grid,
        "l_grid":     l_grid,
        "minima":     minima,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 4 — Exploding Effective Step (β₁ too large)
# ═════════════════════════════════════════════════════════════════════════════
def scenario_bad_hyperparams(n_steps: int = 300):
    """
    Problem: simple 1-D quadratic f(θ) = θ².

    Demonstrate slow or oscillatory convergence when β₁ is set too high
    or the learning rate is much too large.

    Why diverges:
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        m̂_t = m_t / (1 - β₁ᵗ)

        For β₁=0.999, signed-gradient memory decays very slowly, so m can keep
        pointing in a stale direction after the gradient changes sign.

        For large α with β₁=0.9 (default), updates make large transient
        oscillations around the minimum before converging.
    """
    loss_fn = lambda p: float(p[0] ** 2)
    grad_fn = lambda p: np.array([2.0 * p[0]])

    theta_start = 3.0
    configs = [
        ("Adam (β₁=0.9, α=0.1)",    Adam, {"learning_rate": 0.1,  "beta1": 0.9,   "beta2": 0.999}),
        ("Adam (β₁=0.999, α=0.1)",  Adam, {"learning_rate": 0.1,  "beta1": 0.999, "beta2": 0.999}),
        ("Adam (β₁=0.9, α=10)",     Adam, {"learning_rate": 10.0, "beta1": 0.9,   "beta2": 0.999}),
        ("GD  (α=0.1)",              GradientDescent, {"learning_rate": 0.1}),
    ]

    results = {}
    for name, opt_cls, kwargs in configs:
        losses, thetas = _run_1d(opt_cls(**kwargs), grad_fn, loss_fn, theta_start, n_steps)
        # Clip for display
        losses = np.clip(losses, 0, 50)
        results[name] = {"losses": losses, "thetas": np.clip(thetas, -20, 20)}

    return {
        "title":      "Scenario 4 — Hyperparameter Sensitivity",
        "subtitle":   "Long momentum memory and an oversized learning rate",
        "why_fails":  (
            "Bias correction does not amplify the first gradient 1000×: the (1-β₁) factor\n"
            "in m cancels that denominator. The real issue with β₁=0.999 is very long\n"
            "signed-gradient memory, which delays reaction when the gradient changes sign.\n\n"
            "With α=10 Adam makes large oscillatory transients around θ=0 before settling.\n"
            "GD with α=0.1 contracts predictably on this quadratic.\n\n"
            "Lesson: default Adam parameters are robust, but extreme β₁ or α can make\n"
            "convergence unnecessarily slow or oscillatory."
        ),
        "results":    results,
    }
