"""Visualizations for the Music Recommendation module. Pure Matplotlib."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "bg": "#0d1117", "surface": "#161b22", "grid": "#21262d",
    "text": "#e6edf3", "user": "#58a6ff", "item": "#f78166",
    "accent": "#a8ff78", "warn": "#d29922", "you": "#ffd700",
}


def _dark(fig: Figure):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in fig.axes:
        try:
            ax.set_facecolor(COLORS["surface"])
            ax.tick_params(colors=COLORS["text"], labelsize=8)
            ax.xaxis.label.set_color(COLORS["text"])
            ax.yaxis.label.set_color(COLORS["text"])
            ax.title.set_color(COLORS["text"])
            ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.6)
            for sp in ax.spines.values():
                sp.set_edgecolor(COLORS["grid"])
        except Exception:
            pass


def plot_mf_loss(loss_history: list[float], fig: Figure | None = None) -> Figure:
    if fig is None:
        fig = Figure(figsize=(6, 3), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)
    xs = list(range(1, len(loss_history) + 1))
    ax.plot(xs, loss_history, color=COLORS["accent"], lw=2.5)
    ax.fill_between(xs, loss_history, alpha=0.15, color=COLORS["accent"])
    ax.set_xlabel("ALS Epoch"); ax.set_ylabel("RMSE")
    ax.set_title("ALS Convergence — RMSE over Epochs", fontweight="bold")
    _dark(fig)
    return fig


def plot_latent_heatmap(
    U: np.ndarray,
    labels: list[str] | None = None,
    title: str = "User Latent Factors",
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(7, 4), tight_layout=True)
    fig.clear()
    ax      = fig.add_subplot(111)
    n_show  = min(30, U.shape[0])
    data    = U[:n_show]
    lbs     = (labels or [str(i) for i in range(U.shape[0])])[:n_show]
    im      = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_yticks(range(len(lbs)))
    ax.set_yticklabels(lbs, fontsize=7)
    ax.set_xlabel("Latent Factor Dimension")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    _dark(fig)
    return fig


def plot_embeddings_2d(
    user_2d:         np.ndarray,
    item_2d:         np.ndarray,
    user_labels:     list[str],
    item_labels:     list[str],
    highlight_user:  str | None          = None,
    highlight_items: list[str] | None    = None,
    you_point:       np.ndarray | None   = None,   # (1,2) or None
    fig:             Figure | None       = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(7, 5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)

    ax.scatter(item_2d[:, 0], item_2d[:, 1],
               s=25, color=COLORS["item"], alpha=0.6, label="Artists", zorder=2)
    ax.scatter(user_2d[:, 0], user_2d[:, 1],
               s=30, color=COLORS["user"], alpha=0.45, label="Users",
               marker="^", zorder=2)

    if len(item_labels) <= 35:
        for (x, y), lbl in zip(item_2d, item_labels):
            ax.text(x + 0.01, y + 0.01, lbl[:18], fontsize=6,
                    color=COLORS["item"], alpha=0.85)

    if highlight_user and highlight_user in user_labels:
        idx = user_labels.index(highlight_user)
        ax.scatter(user_2d[idx, 0], user_2d[idx, 1],
                   s=180, color="#ffd700", marker="*", zorder=5,
                   label=f"▶ {highlight_user}")

    if highlight_items:
        for lbl in highlight_items:
            if lbl in item_labels:
                j = item_labels.index(lbl)
                ax.scatter(item_2d[j, 0], item_2d[j, 1],
                           s=110, color=COLORS["accent"], marker="D", zorder=4)

    if you_point is not None and you_point.shape[0] > 0:
        ax.scatter(you_point[0, 0], you_point[0, 1],
                   s=280, color=COLORS["you"], marker="*", zorder=7, label="★ YOU")
        ax.annotate("YOU", (you_point[0, 0], you_point[0, 1]),
                    fontsize=9, fontweight="bold", color=COLORS["you"],
                    xytext=(6, 6), textcoords="offset points")

    ax.set_title("2-D Latent Space (PCA / t-SNE projection)", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    _dark(fig)
    return fig


def plot_rating_matrix(
    R: np.ndarray, mask: np.ndarray,
    user_labels: list[str], item_labels: list[str],
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(8, 5), tight_layout=True)
    fig.clear()
    ax = fig.add_subplot(111)
    n_u = min(25, R.shape[0])
    n_i = min(30, R.shape[1])
    disp = np.where(mask[:n_u, :n_i] > 0, R[:n_u, :n_i], np.nan)
    im = ax.imshow(disp, aspect="auto", cmap="YlOrRd",
                   vmin=1, vmax=10, interpolation="nearest")
    ax.set_yticks(range(n_u))
    ax.set_yticklabels(user_labels[:n_u], fontsize=7)
    ax.set_xticks(range(n_i))
    ax.set_xticklabels(item_labels[:n_i], fontsize=6, rotation=45, ha="right")
    ax.set_title("Interaction Matrix (observed ratings)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score [1–10]", pad=0.01)
    _dark(fig)
    return fig


def plot_recommendations(
    recs: list[tuple],
    title: str = "Top Recommendations",
    fig: Figure | None = None,
) -> Figure:
    if fig is None:
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
    fig.clear()
    if not recs:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No results", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["text"])
        _dark(fig); return fig
    ax     = fig.add_subplot(111)
    items  = [r[0][:24] for r in recs]
    scores = [float(r[1]) for r in recs]
    s_arr  = np.array(scores)
    norm   = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-10)
    colors = plt.cm.YlOrRd(0.3 + 0.7 * norm)
    bars   = ax.barh(range(len(items)), scores, color=colors,
                     edgecolor=COLORS["grid"], linewidth=0.6)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(title, fontweight="bold")
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{sc:.2f}", va="center", fontsize=8, color=COLORS["text"])
    _dark(fig)
    return fig


def plot_latent_snapshot(
    U_list: list[np.ndarray],
    epoch_idx: int,
    labels: list[str],
    fig: Figure | None = None,
) -> Figure:
    U = U_list[min(epoch_idx, len(U_list) - 1)]
    return plot_latent_heatmap(
        U, labels,
        title=f"User Latent Factors — Snapshot {epoch_idx + 1}/{len(U_list)}",
        fig=fig,
    )
