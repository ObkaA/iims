"""Training engine — runs the optimisation loop in a background QThread."""
from __future__ import annotations
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from metrics import regression_scores

MAX_DIAGNOSTIC_EVAL_SAMPLES = 512
MAX_DIAGNOSTIC_POINTS = 300


def optimizer_steps_for_epochs(epochs: float, train_size: int, batch_size: int) -> int:
    """Convert a data budget in epochs into optimizer update steps."""
    if epochs <= 0 or train_size <= 0 or batch_size <= 0:
        raise ValueError("Epochs, train size and batch size must be positive.")
    effective_batch = min(batch_size, train_size)
    return max(1, int(np.ceil(epochs * train_size / effective_batch)))


class TrainingWorker(QThread):
    """Runs one optimiser × model combination and emits progress signals."""

    step_done = pyqtSignal(dict)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, model, optimizer, X, y, X_eval=None, y_eval=None,
                 n_iterations=200, batch_size=32, emit_every=1, random_seed=42):
        super().__init__()
        self.model        = model
        self.optimizer    = optimizer
        self.X            = X
        self.y            = y
        self.X_eval       = X if X_eval is None else X_eval
        self.y_eval       = y if y_eval is None else y_eval
        self.n_iterations = n_iterations
        self.batch_size   = batch_size
        self.emit_every   = emit_every
        self.random_seed  = random_seed
        self._paused      = False
        self._stopped     = False
        self.start_time   = None

        self._X_diagnostic, self._y_diagnostic = self._diagnostic_sample(
            self.X_eval, self.y_eval
        )
        self._X_gradient_diagnostic, self._y_gradient_diagnostic = (
            self._diagnostic_sample(self.X, self.y)
        )

    @staticmethod
    def _diagnostic_sample(X, y):
        size = len(y)
        if size <= MAX_DIAGNOSTIC_EVAL_SAMPLES:
            return X, y
        indices = np.linspace(
            0, size - 1, MAX_DIAGNOSTIC_EVAL_SAMPLES, dtype=int
        )
        return X[indices], y[indices]

    def pause(self):  self._paused = not self._paused
    def stop(self):   self._stopped = True; self._paused = False

    def run(self):
        try:    self._run()
        except Exception as exc: self.error.emit(str(exc))

    def _run(self):
        model, opt, X, y = self.model, self.optimizer, self.X, self.y
        n = len(y)
        self.start_time = time.time()
        use_newton = opt.name == "Newton Method"
        rng = np.random.default_rng(self.random_seed)
        diagnostic_every = max(
            1, int(np.ceil(self.n_iterations / MAX_DIAGNOSTIC_POINTS))
        )
        last_evaluation_loss = None
        opt.history["samples_per_step"] = min(self.batch_size, n)
        opt.history["train_size"] = n

        for i in range(self.n_iterations):
            while self._paused and not self._stopped:
                time.sleep(0.05)
            if self._stopped:
                break

            # Mini-batch
            if self.batch_size < n:
                idx = rng.choice(n, self.batch_size, replace=False)
                X_b, y_b = X[idx], y[idx]
            else:
                X_b, y_b = X, y

            params = opt.history["params"][-1] if opt.history["params"] else model.params.copy()

            # Newton needs Hessian if available
            if use_newton and hasattr(model, "loss_gradient_hessian"):
                loss, grad, hess = model.loss_gradient_hessian(params, X_b, y_b)
                new_params = opt.step(params, grad, hessian=hess)
            else:
                loss, grad = model.loss_gradient(params, X_b, y_b)
                new_params = opt.step(params, grad)

            opt.record(loss, new_params, grad)
            should_diagnose = (
                i % diagnostic_every == 0 or i == self.n_iterations - 1
            )
            if should_diagnose:
                # All optimizers use the same deterministic held-out sample.
                # Sampling and limiting the number of points keeps large CSV
                # experiments responsive without influencing the updates.
                last_evaluation_loss = float(model.loss(
                    new_params, self._X_diagnostic, self._y_diagnostic
                ))
                opt.history["eval_loss"].append(last_evaluation_loss)
                opt.history["eval_iteration"].append(i)
                diagnostic_gradient = model.gradient(
                    new_params,
                    self._X_gradient_diagnostic,
                    self._y_gradient_diagnostic,
                )
                opt.history["diagnostic_gradient_norm"].append(
                    float(np.linalg.norm(diagnostic_gradient))
                )
            model.params = new_params

            if i % self.emit_every == 0 or i == self.n_iterations - 1:
                self.step_done.emit(self._payload(
                    loss, new_params, i, evaluation_loss=last_evaluation_loss
                ))
                time.sleep(0.01)

        elapsed = time.time() - self.start_time
        self.finished.emit(self._payload(
            opt.history["loss"][-1] if opt.history["loss"] else float("inf"),
            model.params,
            len(opt.history["loss"]) - 1,
            elapsed=elapsed,
            evaluation_loss=last_evaluation_loss,
        ))

    def _payload(self, loss, params, iteration, elapsed=None, evaluation_loss=None):
        p = {
            "loss":      float(loss),
            "params":    params.copy(),
            "iteration": iteration,
            "step":      iteration + 1,
            "epoch":     (iteration + 1) * min(self.batch_size, len(self.y)) / len(self.y),
            "elapsed":   elapsed or (time.time() - self.start_time),
            "optimizer": self.optimizer.name,
        }
        if evaluation_loss is not None:
            p["evaluation_loss"] = float(evaluation_loss)
        if hasattr(self.model, "accuracy"):
            p["accuracy"] = self.model.accuracy(self.X_eval, self.y_eval)
        elif self.model.task == "regression":
            predictions = self.model.predict(self.X_eval)
            p["r2"], p["rmse"] = regression_scores(self.y_eval, predictions)
        return p
