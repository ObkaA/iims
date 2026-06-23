"""Diagnostics for a real Adam run from the Optimization tab."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


PALETTE = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "grid": "#21262d",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "raw": "#58a6ff",
    "trend": "#a8ff78",
    "warning": "#d29922",
    "failure": "#f85149",
    "ok": "#3fb950",
}
OPTIMIZER_COLORS = {
    "Adam": "#a8ff78",
    "SGD": "#ff6b6b",
    "Gradient Descent": "#00d4ff",
}


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _norm_rows(values) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    return np.array([np.linalg.norm(value) for value in arrays], dtype=float)


def _project_parameter_trajectories(histories: dict[str, dict]) -> dict:
    """Project all optimizer paths into one comparable 2-D coordinate system."""
    trajectories = {}
    for name, history in histories.items():
        arrays = [
            np.asarray(value, dtype=float).reshape(-1)
            for value in history.get("params", [])
        ]
        if len(arrays) >= 2 and len({array.size for array in arrays}) == 1:
            trajectories[name] = np.vstack(arrays)

    if not trajectories:
        return {"paths": {}, "mode": "unavailable", "x_label": "", "y_label": ""}

    dimensions = {path.shape[1] for path in trajectories.values()}
    if len(dimensions) != 1:
        return {"paths": {}, "mode": "unavailable", "x_label": "", "y_label": ""}

    dimension = dimensions.pop()
    if dimension == 1:
        projected = {
            name: np.column_stack([path[:, 0], np.zeros(path.shape[0])])
            for name, path in trajectories.items()
        }
        return {
            "paths": projected,
            "mode": "direct",
            "x_label": "θ₀",
            "y_label": "constant axis",
        }
    if dimension == 2:
        return {
            "paths": trajectories,
            "mode": "direct",
            "x_label": "θ₀",
            "y_label": "θ₁",
        }

    combined = np.vstack(list(trajectories.values()))
    center = np.mean(combined, axis=0)
    _, _, components = np.linalg.svd(combined - center, full_matrices=False)
    basis = components[:2].T
    projected = {
        name: (path - center) @ basis
        for name, path in trajectories.items()
    }
    return {
        "paths": projected,
        "mode": "PCA",
        "x_label": "PC1",
        "y_label": "PC2",
    }


def analyze_adam_history(history: dict) -> dict:
    """Return plot-ready series and a conservative Adam health verdict."""
    losses = np.asarray(history.get("loss", []), dtype=float)
    loss_source = "mini-batch training"
    gradients = history.get("gradients", [])
    params = history.get("params", [])
    if losses.size < 3 or len(gradients) != losses.size or len(params) < losses.size:
        return {
            "verdict": "NO DATA",
            "color": PALETTE["muted"],
            "reasons": ["Uruchom trening z optimizerem Adam, aby zebrać historię diagnostyczną."],
            "metrics": {},
            "losses": losses,
            "smooth_loss": losses.copy(),
            "gradient_norms": np.array([]),
            "step_norms": np.array([]),
            "improvement": np.array([]),
            "loss_source": loss_source,
        }

    gradient_norms = _norm_rows(gradients)
    parameter_arrays = [np.asarray(value, dtype=float).reshape(-1) for value in params]
    step_norms = np.array([
        np.linalg.norm(current - previous)
        for previous, current in zip(parameter_arrays, parameter_arrays[1:])
    ], dtype=float)
    if step_norms.size > losses.size:
        step_norms = step_norms[:losses.size]
    elif step_norms.size < losses.size:
        step_norms = np.pad(step_norms, (0, losses.size - step_norms.size), mode="edge")

    finite = (
        np.isfinite(losses).all()
        and np.isfinite(gradient_norms).all()
        and np.isfinite(step_norms).all()
    )
    if not finite:
        return {
            "verdict": "FAILURE",
            "color": PALETTE["failure"],
            "reasons": ["Historia zawiera NaN lub nieskończoność — trening jest numerycznie niestabilny."],
            "metrics": {"iterations": int(losses.size)},
            "losses": losses,
            "smooth_loss": losses.copy(),
            "gradient_norms": gradient_norms,
            "step_norms": step_norms,
            "improvement": np.array([]),
            "loss_source": loss_source,
        }

    span = max(3, min(21, losses.size // 12 or 3))
    smooth_loss = _ema(losses, span)
    epsilon = np.finfo(float).eps
    improvement = np.zeros_like(smooth_loss)
    improvement[1:] = (
        smooth_loss[:-1] - smooth_loss[1:]
    ) / np.maximum(np.abs(smooth_loss[:-1]), epsilon)

    window = max(3, min(25, losses.size // 5))
    initial_loss = float(np.median(smooth_loss[:window]))
    final_loss = float(np.median(smooth_loss[-window:]))
    best_index = int(np.argmin(smooth_loss))
    best_loss = float(smooth_loss[best_index])
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), epsilon)
    regression_from_best = (final_loss - best_loss) / max(abs(best_loss), epsilon)
    increase_fraction = float(np.mean(np.diff(smooth_loss) > 1e-10))

    head_gradient = float(np.median(gradient_norms[:window]))
    tail_gradient = float(np.median(gradient_norms[-window:]))
    gradient_ratio = tail_gradient / max(head_gradient, epsilon)
    head_step = float(np.median(step_norms[:window]))
    tail_step = float(np.median(step_norms[-window:]))
    step_ratio = tail_step / max(head_step, epsilon)

    tail = smooth_loss[-window:]
    relative_tail_slope = float(
        np.polyfit(np.arange(window), tail, 1)[0]
        / max(abs(float(np.mean(tail))), epsilon)
    )
    peak_ratio = float(np.max(np.abs(smooth_loss)) / max(abs(initial_loss), epsilon))

    metrics = {
        "iterations": int(losses.size),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_iteration": best_index,
        "loss_reduction": reduction,
        "regression_from_best": regression_from_best,
        "increase_fraction": increase_fraction,
        "gradient_ratio": gradient_ratio,
        "step_ratio": step_ratio,
        "tail_slope": relative_tail_slope,
        "peak_ratio": peak_ratio,
    }

    failure_reasons = []
    if final_loss > initial_loss * 1.20 and regression_from_best > 0.20:
        failure_reasons.append("Końcowy trend loss jest wyraźnie gorszy niż na początku.")
    if peak_ratio > 100.0 and final_loss > initial_loss:
        failure_reasons.append("Loss eksplodował i nie wrócił do poziomu początkowego.")

    warning_reasons = []
    if reduction < 0.05 and gradient_ratio > 0.80:
        warning_reasons.append(
            "Postęp jest bardzo wolny, a gradient prawie nie maleje — zwiększ liczbę iteracji "
            "lub sprawdź learning rate."
        )
    if regression_from_best > 0.25 and best_index < int(0.8 * losses.size):
        warning_reasons.append("Końcówka pogorszyła się względem najlepszego punktu treningu.")
    if increase_fraction > 0.45 and reduction < 0.10:
        warning_reasons.append("Wygładzony loss często rośnie — przebieg jest niestabilny.")
    if relative_tail_slope > 0.002:
        warning_reasons.append("Końcowa część krzywej ma trend rosnący.")

    if failure_reasons:
        verdict = "FAILURE"
        color = PALETTE["failure"]
        reasons = failure_reasons
    elif warning_reasons:
        verdict = "WARNING"
        color = PALETTE["warning"]
        reasons = warning_reasons
    else:
        verdict = "OK"
        color = PALETTE["ok"]
        reasons = [
            f"Wygładzony loss zmalał o {max(reduction, 0.0) * 100:.1f}%.",
            "Nie wykryto eksplozji, trwałego pogorszenia ani numerycznej niestabilności.",
        ]
        if gradient_ratio < 0.5:
            reasons.append("Norma gradientu wyraźnie zmalała.")
        if step_ratio < 0.5:
            reasons.append("Kroki optymalizatora zmniejszają się przy zbliżaniu do minimum.")

    return {
        "verdict": verdict,
        "color": color,
        "reasons": reasons,
        "metrics": metrics,
        "losses": losses,
        "smooth_loss": smooth_loss,
        "gradient_norms": gradient_norms,
        "step_norms": step_norms,
        "improvement": improvement,
        "loss_source": loss_source,
    }


def analyze_optimizer_comparison(histories: dict[str, dict]) -> dict:
    """Report Adam health separately from the fixed-config benchmark result."""
    runs = {
        name: analyze_adam_history(history)
        for name, history in histories.items()
        if name in OPTIMIZER_COLORS
    }
    trajectory = _project_parameter_trajectories(histories)
    for name, run in runs.items():
        evaluation_losses = np.asarray(
            histories[name].get("eval_loss", []), dtype=float
        )
        if evaluation_losses.size >= 3:
            span = max(3, min(21, evaluation_losses.size // 12 or 3))
            evaluation_iterations = np.asarray(
                histories[name].get("eval_iteration", []), dtype=int
            )
            if evaluation_iterations.size != evaluation_losses.size:
                evaluation_iterations = np.arange(evaluation_losses.size)
            run["comparison_losses"] = evaluation_losses
            run["smooth_comparison_loss"] = _ema(evaluation_losses, span)
            run["comparison_iterations"] = evaluation_iterations
            run["comparison_loss_source"] = "evaluation"
        else:
            run["comparison_losses"] = run["losses"]
            run["smooth_comparison_loss"] = run["smooth_loss"]
            run["comparison_iterations"] = np.arange(run["losses"].size)
            run["comparison_loss_source"] = "training fallback"
    adam = runs.get("Adam")
    if adam is None or adam["verdict"] == "NO DATA":
        return {
            "verdict": "NO DATA",
            "color": PALETTE["muted"],
            "reasons": ["Uruchom porównanie Adam vs SGD vs Gradient Descent."],
            "metrics": {},
            "runs": runs,
            "comparison_summary": "No optimizer comparison available.",
            "trajectory": trajectory,
        }

    verdict = adam["verdict"]
    color = adam["color"]
    reasons = list(adam["reasons"])
    metrics = dict(adam["metrics"])
    usable_runs = {
        name: result for name, result in runs.items()
        if result["verdict"] != "NO DATA"
    }
    comparison_summary = "Run SGD and Gradient Descent to compare optimizers."

    if len(usable_runs) > 1:
        final_losses = {}
        for name, result in usable_runs.items():
            trend = result["smooth_comparison_loss"]
            window = max(3, min(25, trend.size // 5))
            final_losses[name] = float(np.median(trend[-window:]))
        ranking = sorted(final_losses, key=final_losses.get)
        winner = ranking[0]
        winner_loss = final_losses[winner]
        adam_loss = final_losses["Adam"]
        denominator = max(abs(winner_loss), np.finfo(float).eps)
        adam_gap = (adam_loss - winner_loss) / denominator
        metrics.update({
            "adam_final_loss": adam_loss,
            "comparison_winner": winner,
            "winner_loss": winner_loss,
            "adam_rank": ranking.index("Adam") + 1,
            "ranking": ranking,
            "adam_baseline_gap": adam_gap,
            "final_losses": final_losses,
        })
        if winner == "Adam":
            comparison_summary = "Adam uzyskał najniższy końcowy loss w tym benchmarku."
        else:
            comparison_summary = (
                f"Najniższy końcowy loss uzyskał {winner}. Adam zajął "
                f"miejsce {metrics['adam_rank']}/{len(ranking)}. Nie zmienia to "
                "werdyktu Adam Health."
            )

    return {
        "verdict": verdict,
        "color": color,
        "reasons": reasons,
        "metrics": metrics,
        "runs": runs,
        "comparison_summary": comparison_summary,
        "trajectory": trajectory,
    }


def plot_optimizer_comparison(comparison: dict, fig: Figure) -> Figure:
    """Draw Adam, SGD and GD diagnostics on common axes."""
    fig.clear()
    axes = fig.subplots(2, 3)
    ax_loss, ax_trajectory, ax_final, ax_gradient, ax_step, ax_summary = axes.flat
    runs = comparison.get("runs", {})

    usable = {
        name: run for name, run in runs.items()
        if run["verdict"] != "NO DATA" and run["losses"].size >= 3
    }
    if not usable:
        ax_loss.text(
            0.5, 0.5, "Run Adam vs SGD vs Gradient Descent",
            ha="center", va="center", transform=ax_loss.transAxes,
            color=PALETTE["text"], fontsize=12,
        )
        for axis in (ax_trajectory, ax_gradient, ax_step, ax_final, ax_summary):
            axis.set_visible(False)
    else:
        for name, run in usable.items():
            color = OPTIMIZER_COLORS[name]
            comparison_loss = run["smooth_comparison_loss"]
            comparison_iterations = run["comparison_iterations"]
            ax_loss.semilogy(
                comparison_iterations,
                np.maximum(comparison_loss, np.finfo(float).tiny),
                color=color, lw=2.0, label=name,
            )
            iterations = np.arange(run["gradient_norms"].size)
            ax_gradient.semilogy(
                iterations,
                np.maximum(run["gradient_norms"], np.finfo(float).tiny),
                color=color, lw=1.6, label=name,
            )
            ax_step.semilogy(
                iterations,
                np.maximum(run["step_norms"], np.finfo(float).tiny),
                color=color, lw=1.6, label=name,
            )

        ax_loss.set_title("Smoothed evaluation loss — lower is better", fontweight="bold")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss (log)")
        ax_gradient.set_title("Gradient norm", fontweight="bold")
        ax_gradient.set_xlabel("Iteration")
        ax_gradient.set_ylabel("||gradient|| (log)")
        ax_step.set_title("Parameter-step norm", fontweight="bold")
        ax_step.set_xlabel("Iteration")
        ax_step.set_ylabel("||Δθ|| (log)")

        final_loss_map = comparison.get("metrics", {}).get("final_losses", {})
        names = [name for name in usable if name in final_loss_map]
        final_losses = [final_loss_map[name] for name in names]
        bars = ax_final.bar(
            names, final_losses,
            color=[OPTIMIZER_COLORS[name] for name in names], alpha=0.85,
        )
        ax_final.bar_label(bars, fmt="%.4g", color=PALETTE["text"], fontsize=8)
        ax_final.set_title("Final evaluation-loss trend", fontweight="bold")
        ax_final.set_ylabel("Loss")
        ax_final.tick_params(axis="x", labelrotation=12)

        trajectory = comparison.get("trajectory", {})
        paths = trajectory.get("paths", {})
        winner = comparison.get("metrics", {}).get("comparison_winner")
        for name, path in paths.items():
            if name not in OPTIMIZER_COLORS or path.size == 0:
                continue
            color = OPTIMIZER_COLORS[name]
            point_count = min(300, len(path))
            indices = np.linspace(0, len(path) - 1, point_count, dtype=int)
            shown = path[indices]
            ax_trajectory.plot(
                shown[:, 0], shown[:, 1], color=color, lw=1.7,
                alpha=0.9, label=name,
            )
            ax_trajectory.scatter(
                shown[0, 0], shown[0, 1], color=color, marker="o",
                s=28, edgecolor=PALETTE["text"], linewidth=0.5, zorder=4,
            )
            ax_trajectory.scatter(
                shown[-1, 0], shown[-1, 1], color=color,
                marker="*" if name == winner else "X",
                s=100 if name == winner else 55, zorder=5,
            )
        mode = trajectory.get("mode", "unavailable")
        ax_trajectory.set_title(
            "Parameter trajectory" + (" (PCA)" if mode == "PCA" else ""),
            fontweight="bold",
        )
        ax_trajectory.set_xlabel(trajectory.get("x_label", ""))
        ax_trajectory.set_ylabel(trajectory.get("y_label", ""))
        if paths:
            trajectory_legend = ax_trajectory.legend(
                fontsize=7, facecolor=PALETTE["surface"],
                edgecolor=PALETTE["grid"], framealpha=0.9,
            )
            for label in trajectory_legend.get_texts():
                label.set_color(PALETTE["text"])

        ax_summary.axis("off")
        ax_summary.text(
            0.02, 0.95,
            "How to read the trajectory\n\n"
            "○ start\nX final position\n★ benchmark winner\n\n"
            "PCA is used only when the model\nhas more than two parameters.",
            transform=ax_summary.transAxes, va="top", ha="left",
            color=PALETTE["muted"], fontsize=9, linespacing=1.45,
        )

    for axis in fig.axes:
        axis.set_facecolor(PALETTE["surface"])
        axis.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.55)
        axis.tick_params(colors=PALETTE["text"], labelsize=8)
        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick_label.set_color(PALETTE["text"])
        axis.xaxis.get_offset_text().set_color(PALETTE["text"])
        axis.yaxis.get_offset_text().set_color(PALETTE["text"])
        axis.xaxis.label.set_color(PALETTE["text"])
        axis.yaxis.label.set_color(PALETTE["text"])
        axis.title.set_color(PALETTE["text"])
        for spine in axis.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

    if usable:
        legend = ax_loss.legend(
            fontsize=8, facecolor=PALETTE["surface"],
            edgecolor=PALETTE["grid"], framealpha=0.9,
        )
        for label in legend.get_texts():
            label.set_color(PALETTE["text"])

    fig.patch.set_facecolor(PALETTE["bg"])
    winner = comparison.get("metrics", {}).get("comparison_winner", "—")
    fig.suptitle(
        f"Adam health: {comparison['verdict']}  |  Benchmark winner: {winner}",
        color=comparison["color"], fontsize=12, fontweight="bold",
    )
    if usable:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        ax_loss.set_position([0.10, 0.12, 0.82, 0.76])
    return fig


def plot_adam_diagnostics(diagnostic: dict, fig: Figure) -> Figure:
    """Draw four diagnostics using the same history as the Optimization tab."""
    fig.clear()
    axes = fig.subplots(2, 2)
    ax_loss, ax_gradient, ax_step, ax_change = axes.flat
    losses = diagnostic["losses"]
    smooth = diagnostic["smooth_loss"]
    gradients = diagnostic["gradient_norms"]
    steps = diagnostic["step_norms"]
    improvement = diagnostic["improvement"]

    if losses.size < 3:
        ax_loss.text(
            0.5, 0.5, "Run Adam on the current experiment",
            ha="center", va="center", transform=ax_loss.transAxes,
            color=PALETTE["text"], fontsize=12,
        )
        for axis in (ax_gradient, ax_step, ax_change):
            axis.set_visible(False)
    else:
        iterations = np.arange(losses.size)
        positive_loss = np.all(losses > 0)
        if positive_loss:
            ax_loss.semilogy(iterations, losses, color=PALETTE["raw"], alpha=0.35, label="Raw loss")
            ax_loss.semilogy(iterations, smooth, color=PALETTE["trend"], lw=2.0, label="EMA trend")
        else:
            ax_loss.plot(iterations, losses, color=PALETTE["raw"], alpha=0.35, label="Raw loss")
            ax_loss.plot(iterations, smooth, color=PALETTE["trend"], lw=2.0, label="EMA trend")
        best = diagnostic["metrics"]["best_iteration"]
        ax_loss.scatter(best, smooth[best], color="#ffd700", marker="*", s=80, zorder=5, label="Best")
        ax_loss.set_title("Loss: raw data and stable trend", fontweight="bold")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss" + (" (log)" if positive_loss else ""))

        ax_gradient.semilogy(
            iterations, np.maximum(gradients, np.finfo(float).tiny),
            color="#c084fc", lw=1.7,
        )
        ax_gradient.set_title("Gradient norm", fontweight="bold")
        ax_gradient.set_xlabel("Iteration")
        ax_gradient.set_ylabel("||gradient|| (log)")

        ax_step.semilogy(
            iterations, np.maximum(steps, np.finfo(float).tiny),
            color="#00d4ff", lw=1.7,
        )
        ax_step.set_title("Adam parameter-step norm", fontweight="bold")
        ax_step.set_xlabel("Iteration")
        ax_step.set_ylabel("||Δθ|| (log)")

        colors = np.where(improvement >= 0, PALETTE["ok"], PALETTE["failure"])
        ax_change.bar(iterations[1:], improvement[1:] * 100.0, color=colors[1:], width=1.0, alpha=0.75)
        ax_change.axhline(0, color=PALETTE["muted"], lw=0.8)
        ax_change.set_title("Loss improvement per iteration", fontweight="bold")
        ax_change.set_xlabel("Iteration")
        ax_change.set_ylabel("Improvement [%]")

    for axis in fig.axes:
        axis.set_facecolor(PALETTE["surface"])
        axis.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.55)
        axis.tick_params(colors=PALETTE["text"], labelsize=8)
        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick_label.set_color(PALETTE["text"])
        axis.xaxis.get_offset_text().set_color(PALETTE["text"])
        axis.yaxis.get_offset_text().set_color(PALETTE["text"])
        axis.xaxis.label.set_color(PALETTE["text"])
        axis.yaxis.label.set_color(PALETTE["text"])
        axis.title.set_color(PALETTE["text"])
        for spine in axis.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
    if losses.size >= 3:
        legend = ax_loss.legend(
            fontsize=8, facecolor=PALETTE["surface"],
            edgecolor=PALETTE["grid"], framealpha=0.9,
        )
        for label in legend.get_texts():
            label.set_color(PALETTE["text"])

    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"Adam diagnostic: {diagnostic['verdict']}",
        color=diagnostic["color"], fontsize=12, fontweight="bold",
    )
    if losses.size >= 3:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        ax_loss.set_position([0.10, 0.12, 0.82, 0.76])
    return fig
