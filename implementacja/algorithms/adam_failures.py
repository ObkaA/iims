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

    Why Adam fails here:
        For x₁, v₁ ≈ 0 between sparse updates. When the gradient
        finally fires, v̂₁ is still near zero → effective step
        α/√v̂₁ is HUGE → overshoot past the minimum.

    GD is unaffected: step = α·g, so rare-but-large g → small step α·g.

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
        "subtitle":    f"x₁ gradient fires only once every {K} steps",
        "why_fails":   (
            "Adam's v₂ accumulates near zero between sparse updates.\n"
            f"When gradient fires at step t={K}, v̂₂≈0  →  step = α/√v̂₂ >> α.\n"
            "Result: massive overshoot on the rarely-updated dimension.\n\n"
            "GD is immune: step = α·g regardless of history."
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
        When the landscape flips, v still encodes the old direction for
        ~1/(1-0.999)=1000 steps → tiny effective steps in the new direction.

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
            "Adam's β₂=0.999 means v has ~1000-step memory.\n"
            "After the landscape flips, v still encodes the OLD gradient direction.\n"
            "→ Adam takes ~1000 steps to 'forget' the old landscape.\n\n"
            "Lower β₂=0.9 recovers faster (shorter memory, ~10 steps).\n"
            "GD adapts in 1 step — no memory of past gradients."
        ),
        "results":    results,
        "phase_step": T,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Sharp vs Flat Minima (generalisation gap)
# ═════════════════════════════════════════════════════════════════════════════
def scenario_sharp_vs_flat(n_steps: int = 800):
    """
    Problem: 1-D loss with one SHARP minimum (good train loss, bad test)
    and one FLAT minimum (slightly worse train loss, good test).

        f(θ) = 0.4·sin(8θ)·exp(-0.3·θ²)  +  0.05·θ²

    Sharp minimum near θ ≈ -0.4 (low loss, high curvature)
    Flat minimum near  θ ≈  0   (higher loss, low curvature)

    Why Adam converges to sharp minimum:
        Adam's adaptive step is α/√v̂ — it SHRINKS the step in noisy/oscillating
        regions, helping it squeeze into sharp minima. SGD's momentum makes it
        'roll through' sharp minima and settle in flat ones.

    Wilson et al. (2017) showed this explains Adam's generalisation gap on
    deep learning tasks: sharp minima found by Adam overfit more than flat
    minima found by SGD.
    """
    def loss_fn(theta):
        t = float(theta[0]) if hasattr(theta, '__len__') else float(theta)
        return 0.4 * np.sin(8 * t) * np.exp(-0.3 * t**2) + 0.05 * t**2

    def grad_fn(theta):
        t = float(theta[0])
        dl = (0.4 * 8 * np.cos(8*t) * np.exp(-0.3*t**2)
              + 0.4 * np.sin(8*t) * (-0.6*t) * np.exp(-0.3*t**2)
              + 0.1 * t)
        return np.array([dl])

    lr = 0.02
    theta_start = -1.5

    results = {}
    for name, opt_cls, kwargs in [
        ("Adam",   Adam, {"learning_rate": lr}),
        ("SGD",    SGD,  {"learning_rate": lr, "momentum": 0.85}),
        ("GD",     GradientDescent, {"learning_rate": lr}),
    ]:
        losses, thetas = _run_1d(opt_cls(**kwargs), grad_fn, loss_fn, theta_start, n_steps)
        results[name] = {"losses": losses, "thetas": thetas}

    # Loss landscape for plotting
    t_grid = np.linspace(-2.5, 2.5, 500)
    l_grid = np.array([loss_fn(t) for t in t_grid])

    return {
        "title":      "Scenario 3 — Sharp vs Flat Minima",
        "subtitle":   "Adam finds sharp minima → worse generalisation",
        "why_fails":  (
            "Adam's adaptive step α/√v̂ shrinks near oscillations → converges to SHARP minima.\n"
            "Sharp minima: low train loss, high curvature → small perturbation → high test loss.\n\n"
            "SGD with momentum 'rolls through' sharp valleys and settles in FLAT minima.\n"
            "Flat minima: slightly higher train loss, but robust to weight perturbations.\n\n"
            "Reference: Wilson et al. 'The Marginal Value of Momentum for SGD' (2017)"
        ),
        "results":    results,
        "t_grid":     t_grid,
        "l_grid":     l_grid,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario 4 — Exploding Effective Step (β₁ too large)
# ═════════════════════════════════════════════════════════════════════════════
def scenario_bad_hyperparams(n_steps: int = 300):
    """
    Problem: simple 1-D quadratic f(θ) = θ².

    Demonstrate how Adam diverges when β₁ is set too high (≥ 0.999)
    or learning rate is too large.

    Why diverges:
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        m̂_t = m_t / (1 - β₁ᵗ)

        For β₁=0.999, the bias correction 1/(1-0.999^t) is huge for small t
        (correction factor ≈ 1000 at t=1!). With a constant gradient, this
        inflates m̂ massively in early steps → huge parameter jump.

        For large α with β₁=0.9 (default): the effective step can oscillate
        around the minimum without converging.
    """
    loss_fn = lambda p: float(p[0] ** 2)
    grad_fn = lambda p: np.array([2.0 * p[0]])

    theta_start = 3.0
    configs = [
        ("Adam (β₁=0.9, α=0.1)",    Adam, {"learning_rate": 0.1,  "beta1": 0.9,   "beta2": 0.999}),
        ("Adam (β₁=0.999, α=0.1)",  Adam, {"learning_rate": 0.1,  "beta1": 0.999, "beta2": 0.999}),
        ("Adam (β₁=0.9, α=1.0)",    Adam, {"learning_rate": 1.0,  "beta1": 0.9,   "beta2": 0.999}),
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
        "subtitle":   "Adam diverges with β₁≥0.999 or α too large",
        "why_fails":  (
            "β₁=0.999: bias correction 1/(1-β₁ᵗ) ≈ 1000 at t=1.\n"
            "Early m̂ is 1000× amplified → parameter jumps wildly.\n\n"
            "Large α=1.0: effective step α/√v̂ >> 1 in early steps before\n"
            "v̂ accumulates enough history to act as denominator.\n\n"
            "GD with α=0.1: always stable, step = α·g → bounded update.\n\n"
            "Lesson: Adam is more sensitive to hyperparameter choice than GD."
        ),
        "results":    results,
    }
