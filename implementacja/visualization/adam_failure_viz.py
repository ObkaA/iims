"""
Visualizations for the Adam Failure Scenarios tab.
Pure Matplotlib — no external scientific libraries.
"""
from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure

COLORS = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "grid":    "#21262d",
    "text":    "#e6edf3",
    "Adam":             "#a8ff78",
    "Adam β₂=0.9":     "#78ffd6",
    "Adam (β₁=0.9, α=0.1)":   "#a8ff78",
    "Adam (β₁=0.999, α=0.1)": "#ff6b6b",
    "Adam (β₁=0.9, α=1.0)":   "#ffd700",
    "Gradient Descent": "#00d4ff",
    "GD":               "#00d4ff",
    "GD  (α=0.1)":      "#00d4ff",
    "SGD":              "#ff6b6b",
    "Newton Method":    "#c084fc",
    "warn":    "#d29922",
    "accent":  "#f78166",
}
_LINE = {"linewidth": 1.8, "alpha": 0.92}


def _style(fig: Figure, axes=None):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in (axes or fig.axes):
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS["grid"])


def _c(name: str) -> str:
    for k, v in COLORS.items():
        if k == name:
            return v
    # fallback: first color match substring
    for k, v in COLORS.items():
        if name.startswith(k):
            return v
    return "#ffffff"


def plot_scenario_sparse(data: dict, fig: Figure) -> Figure:
    """Scenario 1: sparse gradients — loss curves + 2D trajectory."""
    fig.clear()
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax_loss = fig.add_subplot(gs[0])
    ax_traj = fig.add_subplot(gs[1])

    results = data["results"]
    for name, d in results.items():
        c = _c(name)
        ax_loss.semilogy(d["losses"], color=c, label=name, **_LINE)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("Loss Convergence", fontweight="bold")
    ax_loss.legend(fontsize=8, framealpha=0.7)

    # 2D trajectory
    opt_range = np.linspace(-4, 4, 80)
    X0, X1 = np.meshgrid(opt_range, opt_range)
    Z = X0 ** 2 + X1 ** 2
    ax_traj.contourf(X0, X1, Z, levels=20, cmap="Blues", alpha=0.25)
    ax_traj.contour(X0, X1, Z, levels=8, colors=COLORS["grid"], alpha=0.4, linewidths=0.5)

    for name, d in results.items():
        traj = d["traj"]
        c    = _c(name)
        ax_traj.plot(traj[::5, 0], traj[::5, 1], color=c, **_LINE, label=name)
        ax_traj.scatter(traj[-1, 0], traj[-1, 1], color=c, s=60, zorder=5)

    ax_traj.scatter(0, 0, color="#ffd700", s=120, marker="*", zorder=6, label="Optimum")
    ax_traj.set_xlim(-4.2, 4.2)
    ax_traj.set_ylim(-4.2, 4.2)
    ax_traj.set_xlabel("θ₀ (dense updates)")
    ax_traj.set_ylabel("θ₁ (sparse updates)")
    ax_traj.set_title("Optimisation Trajectory", fontweight="bold")
    ax_traj.legend(fontsize=7, framealpha=0.7)

    fig.suptitle(data["title"], color=COLORS["text"], fontsize=11, fontweight="bold", y=1.01)
    _style(fig)
    return fig


def plot_scenario_nonstationary(data: dict, fig: Figure) -> Figure:
    """Scenario 2: non-stationary — θ over time with phase marker."""
    fig.clear()
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax_theta = fig.add_subplot(gs[0])
    ax_loss  = fig.add_subplot(gs[1])

    T = data["phase_step"]
    for name, d in data["results"].items():
        c = _c(name)
        ax_theta.plot(d["thetas"], color=c, label=name, **_LINE)
        ax_loss.semilogy(np.maximum(d["losses"], 1e-6), color=c, label=name, **_LINE)

    # Phase boundary
    for ax in (ax_theta, ax_loss):
        ax.axvline(T, color=COLORS["warn"], lw=1.5, ls="--", alpha=0.8)
        ax.text(T + 5, ax.get_ylim()[1] * 0.9 if ax == ax_loss else ax.get_ylim()[1] * 0.8,
                "← shift →", color=COLORS["warn"], fontsize=8)

    ax_theta.axhline(5, color="#ffffff", lw=0.6, ls=":", alpha=0.4)
    ax_theta.axhline(-5, color="#ffffff", lw=0.6, ls=":", alpha=0.4)
    ax_theta.set_xlabel("Step");  ax_theta.set_ylabel("θ")
    ax_theta.set_title("Parameter Value vs Time", fontweight="bold")
    ax_theta.legend(fontsize=8, framealpha=0.7)

    ax_loss.set_xlabel("Step");  ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_title("Loss After Landscape Shift", fontweight="bold")
    ax_loss.legend(fontsize=8, framealpha=0.7)

    fig.suptitle(data["title"], color=COLORS["text"], fontsize=11, fontweight="bold", y=1.01)
    _style(fig)
    return fig


def plot_scenario_sharp_flat(data: dict, fig: Figure) -> Figure:
    """Scenario 3: sharp vs flat minima."""
    fig.clear()
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax_land  = fig.add_subplot(gs[0])
    ax_loss  = fig.add_subplot(gs[1])

    t_grid = data["t_grid"]
    l_grid = data["l_grid"]
    ax_land.plot(t_grid, l_grid, color="#58a6ff", lw=2, label="f(θ)")
    ax_land.fill_between(t_grid, l_grid, alpha=0.1, color="#58a6ff")

    for name, d in data["results"].items():
        c       = _c(name)
        thetas  = d["thetas"]
        # Plot where each optimiser ended up
        final_t = thetas[-1]
        final_l = float(np.interp(final_t, t_grid, l_grid))
        ax_land.scatter(final_t, final_l, color=c, s=80, zorder=5, label=f"{name} end")
        # Trajectory on loss landscape (last 200 steps)
        trail = thetas[-200:]
        l_trail = np.interp(trail, t_grid, l_grid)
        ax_land.plot(trail, l_trail, color=c, alpha=0.5, lw=0.8)

        ax_loss.semilogy(np.maximum(d["losses"], 1e-8), color=c, label=name, **_LINE)

    ax_land.set_xlabel("θ");  ax_land.set_ylabel("f(θ)")
    ax_land.set_title("Loss Landscape + Final Position", fontweight="bold")
    ax_land.legend(fontsize=7, framealpha=0.7)
    ax_loss.set_xlabel("Step");  ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_title("Convergence Curves", fontweight="bold")
    ax_loss.legend(fontsize=8, framealpha=0.7)

    fig.suptitle(data["title"], color=COLORS["text"], fontsize=11, fontweight="bold", y=1.01)
    _style(fig)
    return fig


def plot_scenario_hyperparams(data: dict, fig: Figure) -> Figure:
    """Scenario 4: bad hyperparameters."""
    fig.clear()
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax_loss  = fig.add_subplot(gs[0])
    ax_theta = fig.add_subplot(gs[1])

    for name, d in data["results"].items():
        c = _c(name)
        ax_loss.plot(d["losses"], color=c, label=name, **_LINE)
        ax_theta.plot(d["thetas"], color=c, label=name, **_LINE)

    ax_loss.set_xlabel("Step");  ax_loss.set_ylabel("Loss (clipped at 50)")
    ax_loss.set_title("Loss — Divergence Visible", fontweight="bold")
    ax_loss.legend(fontsize=7, framealpha=0.7)
    ax_theta.axhline(0, color="#ffd700", lw=1, ls="--", alpha=0.6, label="Optimum")
    ax_theta.set_xlabel("Step");  ax_theta.set_ylabel("θ (clipped)")
    ax_theta.set_title("Parameter Trajectory", fontweight="bold")
    ax_theta.legend(fontsize=7, framealpha=0.7)

    fig.suptitle(data["title"], color=COLORS["text"], fontsize=11, fontweight="bold", y=1.01)
    _style(fig)
    return fig


SCENARIO_PLOTS = {
    0: plot_scenario_sparse,
    1: plot_scenario_nonstationary,
    2: plot_scenario_sharp_flat,
    3: plot_scenario_hyperparams,
}
