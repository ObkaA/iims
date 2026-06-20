"""Visualization helpers — all return Matplotlib Figure objects."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "Gradient Descent": "#00d4ff",
    "SGD":              "#ff6b6b",
    "Adam":             "#a8ff78",
    "bg":               "#0d1117",
    "surface":          "#161b22",
    "grid":             "#21262d",
    "text":             "#e6edf3",
    "accent":           "#f78166",
}

OPT_COLORS = [COLORS["Gradient Descent"], COLORS["SGD"], COLORS["Adam"]]

_DARK = {
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["surface"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "legend.facecolor": "#21262d",
    "legend.edgecolor": "#30363d",
}


def _apply_dark(fig: Figure):
    with plt.rc_context(_DARK):
        pass
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in fig.axes:
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str = "Confusion Matrix",
    fig: Figure | None = None,
) -> Figure:
    """Render a binary confusion matrix with counts and TN/FP/FN/TP labels."""
    matrix = np.asarray(matrix)
    if matrix.shape != (2, 2):
        raise ValueError("Binary confusion matrix must have shape (2, 2).")
    if fig is None:
        fig = Figure(figsize=(6, 5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list(
        "confusion", [COLORS["surface"], "#1f6feb", "#58a6ff"]
    )
    image = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    labels = (("TN", "FP"), ("FN", "TP"))
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                f"{labels[row][column]}\n{value}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#ffffff" if value > threshold else COLORS["text"],
            )
    ax.set_xticks([0, 1], labels=["Class 0", "Class 1"])
    ax.set_yticks([0, 1], labels=["Class 0", "Class 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _apply_dark(fig)
    return fig


# ── Loss curve ────────────────────────────────────────────────────────────────
def plot_loss_curves(histories: dict[str, list[float]], fig: Figure | None = None) -> Figure:
    """histories = {"Optimizer Name": [loss_t0, loss_t1, …]}"""
    if fig is None:
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)
    for name, losses in histories.items():
        color = COLORS.get(name, "#ffffff")
        ax.plot(losses, label=name, color=color, linewidth=2, alpha=0.9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.8)
    _apply_dark(fig)
    return fig


# ── Regression fit ────────────────────────────────────────────────────────────
def plot_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    models_params: dict[str, np.ndarray],
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], y, s=15, color="#58a6ff", alpha=0.5, label="Data", zorder=2)
    x_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
    X_line = np.tile(np.median(X, axis=0), (len(x_line), 1))
    X_line[:, 0] = x_line
    X_line_aug = np.c_[np.ones(len(x_line)), X_line]
    for name, params in models_params.items():
        y_line = X_line_aug @ params
        ax.plot(x_line, y_line, color=COLORS.get(name, "#fff"), linewidth=2, label=name)
    xlabel = "Feature 1" if X.shape[1] == 1 else "Feature 1 (others fixed at median)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("y")
    ax.set_title("Regression Fit", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.8)
    _apply_dark(fig)
    return fig


# ── Decision boundary ─────────────────────────────────────────────────────────
def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model,
    optimizer_name: str = "Model",
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)

    h = 0.05
    if X.shape[1] == 1:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_line = np.linspace(x_min, x_max, 500)
        probabilities = model.predict(x_line[:, None])
        ax.plot(
            x_line,
            probabilities,
            color=COLORS.get(optimizer_name, "#fff"),
            linewidth=2,
            label="P(class 1)",
        )
        ax.scatter(X[:, 0], y, s=20, c=y, cmap="coolwarm", alpha=0.7, zorder=3)
        ax.axhline(0.5, color=COLORS["grid"], linestyle="--", linewidth=1)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Probability / class")
        ax.set_ylim(-0.08, 1.08)
        ax.set_title(f"Classification Fit — {optimizer_name}", fontsize=11, fontweight="bold")
        ax.legend(framealpha=0.8)
        _apply_dark(fig)
        return fig

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.tile(np.median(X, axis=0), (xx.size, 1))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()

    probs = model.predict(grid).reshape(xx.shape)
    cmap_bg = LinearSegmentedColormap.from_list("bg", ["#1a3a5c", "#3a1a2a"])
    ax.contourf(xx, yy, probs, levels=20, cmap=cmap_bg, alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors=[COLORS.get(optimizer_name, "#fff")], linewidths=2)

    mask0 = y == 0
    mask1 = y == 1
    ax.scatter(X[mask0, 0], X[mask0, 1], s=20, color="#58a6ff", alpha=0.8, label="Class 0", zorder=3)
    ax.scatter(X[mask1, 0], X[mask1, 1], s=20, color="#ff6b6b", alpha=0.8, label="Class 1", zorder=3)
    ax.set_title(f"Decision Boundary — {optimizer_name}", fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.8)
    _apply_dark(fig)
    return fig


# ── Comparison bar chart ──────────────────────────────────────────────────────
def plot_comparison(stats: dict[str, dict], fig: Figure | None = None) -> Figure:
    """stats = {"Adam": {"final_loss": 0.1, "iterations": 200, "time": 1.2}, …}"""
    if fig is None:
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
    fig.clear()
    names = list(stats.keys())
    metrics = ["final_loss", "time"]
    titles = ["Final Loss", "Time (s)"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(1, 2, i + 1)
        vals = [stats[n].get(metric, 0) for n in names]
        colors = [COLORS.get(n, "#888") for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor=COLORS["grid"], linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8, color=COLORS["text"]
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, fontsize=8)
    _apply_dark(fig)
    return fig


# ── 3-D loss surface ──────────────────────────────────────────────────────────
def build_loss_surface(model, X, y, resolution: int = 40):
    """Compute 2-D loss surface around current params (first 2 dims)."""
    p = model.params.copy()
    b_range = np.linspace(p[0] - 3, p[0] + 3, resolution)
    w_range = np.linspace(p[1] - 3, p[1] + 3, resolution)
    B, W = np.meshgrid(b_range, w_range)
    Z = np.zeros_like(B)
    for i in range(resolution):
        for j in range(resolution):
            tmp = p.copy()
            tmp[0] = B[i, j]
            tmp[1] = W[i, j]
            Z[i, j] = model.loss(tmp, X, y)
    return B, W, Z


def plot_loss_surface_3d(
    B, W, Z,
    trajectories: dict[str, list] | None = None,
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(6, 4.5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(B, W, Z, cmap="plasma", alpha=0.6, linewidth=0, antialiased=True)
    ax.set_xlabel("Bias (b)", labelpad=6)
    ax.set_ylabel("Weight (w₁)", labelpad=6)
    ax.set_zlabel("Loss", labelpad=6)
    ax.set_title("Loss Surface", fontsize=11, fontweight="bold")

    if trajectories:
        for name, pts in trajectories.items():
            if len(pts) < 2:
                continue
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=COLORS.get(name, "#fff"), linewidth=2, zorder=5, label=name)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                       color=COLORS.get(name, "#fff"), s=60, zorder=6)
        ax.legend(loc="upper right", fontsize=8)

    ax.set_facecolor(COLORS["surface"])
    fig.patch.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.zaxis.label.set_color(COLORS["text"])  # type: ignore
    ax.title.set_color(COLORS["text"])
    return fig
