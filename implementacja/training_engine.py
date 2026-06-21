"""Training engine — runs the optimisation loop in a background QThread."""
from __future__ import annotations
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from metrics import regression_scores


class TrainingWorker(QThread):
    """Runs one optimiser × model combination and emits progress signals."""

    step_done = pyqtSignal(dict)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, model, optimizer, X, y, X_eval=None, y_eval=None,
                 n_iterations=200, batch_size=32, emit_every=1):
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
        self._paused      = False
        self._stopped     = False
        self.start_time   = None

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

        for i in range(self.n_iterations):
            while self._paused and not self._stopped:
                time.sleep(0.05)
            if self._stopped:
                break

            # Mini-batch
            if self.batch_size < n:
                idx = np.random.choice(n, self.batch_size, replace=False)
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
            model.params = new_params

            if i % self.emit_every == 0 or i == self.n_iterations - 1:
                self.step_done.emit(self._payload(loss, new_params, i))
                time.sleep(0.01)

        elapsed = time.time() - self.start_time
        self.finished.emit(self._payload(
            opt.history["loss"][-1] if opt.history["loss"] else float("inf"),
            model.params,
            len(opt.history["loss"]) - 1,
            elapsed=elapsed,
        ))

    def _payload(self, loss, params, iteration, elapsed=None):
        p = {
            "loss":      float(loss),
            "params":    params.copy(),
            "iteration": iteration,
            "elapsed":   elapsed or (time.time() - self.start_time),
            "optimizer": self.optimizer.name,
        }
        if hasattr(self.model, "accuracy"):
            p["accuracy"] = self.model.accuracy(self.X_eval, self.y_eval)
        elif self.model.task == "regression":
            predictions = self.model.predict(self.X_eval)
            p["r2"], p["rmse"] = regression_scores(self.y_eval, predictions)
        return p
