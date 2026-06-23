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
TIE_RELATIVE_TOLERANCE = 0.01
STEP_NORM_FLOOR = 1e-12


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


def build_parameter_loss_contours(
    model,
    X: np.ndarray,
    y: np.ndarray,
    histories: dict[str, dict],
    resolution: int = 45,
) -> dict | None:
    """Build an explicitly approximate loss surface for a 2-parameter model."""
    paths = []
    for history in histories.values():
        for value in history.get("params", []):
            params = np.asarray(value, dtype=float).reshape(-1)
            if params.size != 2:
                return None
            paths.append(params)
    if not paths:
        return None

    points = np.vstack(paths)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 0.5)
    lower = lower - 0.12 * span
    upper = upper + 0.12 * span
    theta0 = np.linspace(lower[0], upper[0], resolution)
    theta1 = np.linspace(lower[1], upper[1], resolution)
    grid0, grid1 = np.meshgrid(theta0, theta1)
    loss = np.empty_like(grid0)
    for row in range(resolution):
        for column in range(resolution):
            params = np.array([grid0[row, column], grid1[row, column]])
            loss[row, column] = model.loss(params, X, y)
    minimum_index = np.unravel_index(np.argmin(loss), loss.shape)
    return {
        "theta0": grid0,
        "theta1": grid1,
        "loss": loss,
        "minimum": np.array([
            grid0[minimum_index], grid1[minimum_index]
        ]),
        "minimum_loss": float(loss[minimum_index]),
        "label": "estimated grid minimum",
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
    samples_per_step = max(1, int(history.get("samples_per_step", 1)))
    train_size = max(1, int(history.get("train_size", samples_per_step)))
    epoch_per_step = samples_per_step / train_size

    metrics = {
        "iterations": int(losses.size),
        "epochs": float(losses.size * epoch_per_step),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_iteration": best_index,
        "best_epoch": float((best_index + 1) * epoch_per_step),
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
        history = histories[name]
        samples_per_step = max(1, int(history.get("samples_per_step", 1)))
        train_size = max(1, int(history.get("train_size", samples_per_step)))
        epoch_per_step = samples_per_step / train_size
        evaluation_losses = np.asarray(
            history.get("eval_loss", []), dtype=float
        )
        if evaluation_losses.size >= 3:
            span = max(3, min(21, evaluation_losses.size // 12 or 3))
            evaluation_iterations = np.asarray(
                history.get("eval_iteration", []), dtype=int
            )
            if evaluation_iterations.size != evaluation_losses.size:
                evaluation_iterations = np.arange(evaluation_losses.size)
            run["comparison_losses"] = evaluation_losses
            run["smooth_comparison_loss"] = _ema(evaluation_losses, span)
            run["comparison_iterations"] = evaluation_iterations
            run["comparison_epochs"] = (
                evaluation_iterations + 1
            ) * epoch_per_step
            run["comparison_loss_source"] = "evaluation"
        else:
            run["comparison_losses"] = run["losses"]
            run["smooth_comparison_loss"] = run["smooth_loss"]
            run["comparison_iterations"] = np.arange(run["losses"].size)
            run["comparison_epochs"] = (
                np.arange(run["losses"].size) + 1
            ) * epoch_per_step
            run["comparison_loss_source"] = "training fallback"
        diagnostic_gradient_norms = np.asarray(
            history.get("diagnostic_gradient_norm", []), dtype=float
        )
        if diagnostic_gradient_norms.size == run["comparison_iterations"].size:
            run["comparison_gradient_norms"] = diagnostic_gradient_norms
            run["comparison_gradient_epochs"] = run["comparison_epochs"]
            run["gradient_source"] = "shared training sample"
        else:
            run["comparison_gradient_norms"] = run["gradient_norms"]
            run["comparison_gradient_epochs"] = (
                np.arange(run["gradient_norms"].size) + 1
            ) * epoch_per_step
            run["gradient_source"] = "mini-batch fallback"
        run["step_epochs"] = (
            np.arange(run["step_norms"].size) + 1
        ) * epoch_per_step
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
        shared_budget = min(
            float(result["comparison_epochs"][-1])
            for result in usable_runs.values()
        )
        budget_losses = {}
        for name, result in usable_runs.items():
            budget_losses[name] = float(np.interp(
                shared_budget,
                result["comparison_epochs"],
                result["smooth_comparison_loss"],
            ))
        ranking = sorted(budget_losses, key=budget_losses.get)
        best_name = ranking[0]
        best_loss = budget_losses[best_name]
        tie_margin = max(abs(best_loss) * TIE_RELATIVE_TOLERANCE, 1e-12)
        tied_names = [
            name for name in ranking
            if budget_losses[name] - best_loss <= tie_margin
        ]
        comparison_label = (
            f"TIE: {' / '.join(tied_names)}"
            if len(tied_names) > 1 else best_name
        )
        comparison_winner = best_name if len(tied_names) == 1 else None
        adam_loss = budget_losses["Adam"]
        denominator = max(abs(best_loss), np.finfo(float).eps)
        adam_gap = (adam_loss - best_loss) / denominator
        adam_rank = 1 + sum(
            loss < adam_loss - tie_margin
            for loss in budget_losses.values()
        )
        metrics.update({
            "adam_final_loss": adam_loss,
            "comparison_winner": comparison_winner,
            "comparison_winners": tied_names,
            "comparison_label": comparison_label,
            "winner_loss": best_loss,
            "adam_rank": adam_rank,
            "ranking": ranking,
            "adam_baseline_gap": adam_gap,
            "final_losses": budget_losses,
            "shared_budget_epochs": shared_budget,
            "tie_tolerance": TIE_RELATIVE_TOLERANCE,
        })
        if len(tied_names) > 1:
            comparison_summary = (
                f"Remis w tolerancji {TIE_RELATIVE_TOLERANCE * 100:.0f}% przy "
                f"wspólnym budżecie {shared_budget:.2f} epok: "
                f"{', '.join(tied_names)}."
            )
        elif comparison_winner == "Adam":
            comparison_summary = (
                f"Adam uzyskał najniższy loss przy wspólnym budżecie "
                f"{shared_budget:.2f} epok."
            )
        else:
            comparison_summary = (
                f"Najniższy loss przy {shared_budget:.2f} epok uzyskał "
                f"{comparison_winner}. Adam zajął "
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
    ax_loss, ax_trajectory, ax_final = fig.subplots(1, 3)
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
        for axis in (ax_trajectory, ax_final):
            axis.set_visible(False)
    else:
        for name, run in usable.items():
            color = OPTIMIZER_COLORS[name]
            comparison_loss = run["smooth_comparison_loss"]
            comparison_epochs = run["comparison_epochs"]
            ax_loss.semilogy(
                comparison_epochs,
                np.maximum(comparison_loss, np.finfo(float).tiny),
                color=color, lw=2.0, label=name,
            )

        ax_loss.set_title(
            "Smoothed evaluation loss\n(lower is better)",
            fontsize=10, fontweight="bold",
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss (log)")

        final_loss_map = comparison.get("metrics", {}).get("final_losses", {})
        names = [name for name in usable if name in final_loss_map]
        final_losses = [final_loss_map[name] for name in names]
        bars = ax_final.bar(
            names, final_losses,
            color=[OPTIMIZER_COLORS[name] for name in names], alpha=0.85,
        )
        ax_final.bar_label(bars, fmt="%.4g", color=PALETTE["text"], fontsize=8)
        shared_budget = comparison.get("metrics", {}).get("shared_budget_epochs")
        budget_label = "shared budget" if shared_budget is None else f"{shared_budget:.2f} epochs"
        ax_final.set_title(
            f"Evaluation loss at {budget_label}",
            fontsize=10, fontweight="bold",
        )
        ax_final.set_ylabel("Loss")
        ax_final.tick_params(axis="x", labelrotation=12)
        if shared_budget is not None:
            ax_loss.axvline(
                shared_budget, color=PALETTE["warning"], lw=0.9,
                ls="--", alpha=0.75,
            )

        trajectory = comparison.get("trajectory", {})
        paths = trajectory.get("paths", {})
        loss_surface = comparison.get("loss_surface")
        if loss_surface and trajectory.get("mode") == "direct":
            ax_trajectory.contourf(
                loss_surface["theta0"], loss_surface["theta1"],
                loss_surface["loss"], levels=18, cmap="Blues", alpha=0.22,
            )
            ax_trajectory.contour(
                loss_surface["theta0"], loss_surface["theta1"],
                loss_surface["loss"], levels=9,
                colors=PALETTE["grid"], linewidths=0.55, alpha=0.8,
            )
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
                marker="X", s=55, zorder=5,
            )
        if loss_surface and trajectory.get("mode") == "direct":
            minimum = loss_surface["minimum"]
            ax_trajectory.scatter(
                minimum[0], minimum[1], color="#ffd700", marker="*",
                edgecolor=PALETTE["text"], linewidth=0.6, s=120,
                zorder=7, label=loss_surface["label"],
            )
        mode = trajectory.get("mode", "unavailable")
        ax_trajectory.set_title(
            "Parameter trajectory" + (" (PCA)" if mode == "PCA" else ""),
            fontsize=10, fontweight="bold",
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
        marker_help = "○ start   X final"
        if loss_surface and trajectory.get("mode") == "direct":
            marker_help += "   ★ estimated minimum"
        else:
            marker_help += "   (shared PCA projection)"
        ax_trajectory.text(
            0.02, 0.02, marker_help,
            transform=ax_trajectory.transAxes, ha="left", va="bottom",
            color=PALETTE["muted"], fontsize=7,
            bbox={"facecolor": PALETTE["surface"], "alpha": 0.75, "edgecolor": "none"},
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
    comparison_label = comparison.get("metrics", {}).get("comparison_label", "—")
    fig.suptitle(
        f"Adam health (heuristic): {comparison['verdict']}  |  Benchmark: {comparison_label}",
        color=comparison["color"], fontsize=12, fontweight="bold",
    )
    if usable:
        fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.2, w_pad=2.0, h_pad=2.0)
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
