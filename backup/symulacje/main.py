import sys
import time
from pathlib import Path

# ============================================================
# ŚCIEŻKI PROJEKTU
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT /  "wyniki"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORTY
# ============================================================

from backup.data.generate_data import (
    make_linear,
    make_classification,
    generate_moons,
    generate_circles,
    train_test_split,
    standardize
)

from backup.models import linear_regression as lr_mod
from backup.models import logistic_regression as log_mod

from backup.optimizers.gradient_descent import GradientDescent
from backup.optimizers.newton import NewtonMethod
from backup.optimizers.sgd_adam import Adam

from backup.symulacje.visualization.plots import (
    plot_convergence,
    plot_decision_boundary,
    plot_confusion_matrices,
    plot_time_vs_accuracy,
    plot_linear_regression_fit
)

# ============================================================
# GENEROWANIE DANYCH
# ============================================================

USE_DATASET = "classification"
# "linear"
# "classification"
# "moons"
# "circles"

if USE_DATASET == "linear":

    generated_data = make_linear(
        n=5000,
        p=1,
        noise=2,
        seed=42
    )

    X_data, y_data, beta_true = generated_data

elif USE_DATASET == "classification":

    X_data, y_data = make_classification(
        n=5000,
        p=2,
        sep=1,
        seed=42
    )

elif USE_DATASET == "moons":

    X_data, y_data = generate_moons(
        n=5000,
        noise=0.2,
        seed=42
    )

elif USE_DATASET == "circles":

    X_data, y_data = generate_circles(
        n=5000,
        noise=0.1,
        factor=0.4,
        seed=42
    )

else:
    raise ValueError("Nieznany dataset")

# ============================================================
# CZĘŚĆ 1 — REGRESJA LINIOWA
# ============================================================

print("=" * 60)
print("CZĘŚĆ 1: Regresja liniowa")
print("=" * 60)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_data,
    y_data,
    seed=42
)

X_train_r, X_test_r = standardize(X_train_r, X_test_r)

X_train_r = lr_mod.add_bias(X_train_r)
X_test_r = lr_mod.add_bias(X_test_r)

# ------------------------------------------------------------
# GD
# ------------------------------------------------------------

gd_lr = GradientDescent(
    lr=0.05,
    max_iter=500,
    tol=1e-8,
    line_search=True
)

t0 = time.perf_counter()

gd_lr.fit(
    X_train_r,
    y_train_r,
    lr_mod.mse_loss,
    lr_mod.mse_grad
)

t_gd_lr = time.perf_counter() - t0

rmse_gd = lr_mod.rmse(
    y_test_r,
    lr_mod.predict(X_test_r, gd_lr.theta)
)

# ------------------------------------------------------------
# ADAM
# ------------------------------------------------------------

adam_lr = Adam(
    lr=0.001,
    max_iter=300,
    batch_size=32,
    tol=1e-8,
    seed=42
)

t0 = time.perf_counter()

adam_lr.fit(
    X_train_r,
    y_train_r,
    lr_mod.mse_loss,
    lr_mod.mse_grad
)

t_adam_lr = time.perf_counter() - t0

rmse_adam = lr_mod.rmse(
    y_test_r,
    lr_mod.predict(X_test_r, adam_lr.theta)
)

# ------------------------------------------------------------
# NEWTON
# ------------------------------------------------------------

newton_lr = NewtonMethod(
    max_iter=50,
    tol=1e-10,
    reg=1e-6
)

t0 = time.perf_counter()

newton_lr.fit(
    X_train_r,
    y_train_r,
    lr_mod.mse_loss,
    lr_mod.mse_grad,
    lr_mod.mse_hessian
)

t_newton_lr = time.perf_counter() - t0

rmse_newton = lr_mod.rmse(
    y_test_r,
    lr_mod.predict(X_test_r, newton_lr.theta)
)

# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print(f"  GD      — RMSE: {rmse_gd:.4f} | iteracje: {len(gd_lr.loss_history)} | czas: {t_gd_lr:.3f}s")
print(f"  Adam    — RMSE: {rmse_adam:.4f} | iteracje: {len(adam_lr.loss_history)} | czas: {t_adam_lr:.3f}s")
print(f"  Newton  — RMSE: {rmse_newton:.4f} | iteracje: {len(newton_lr.loss_history)} | czas: {t_newton_lr:.3f}s")

# ------------------------------------------------------------
# PLOTY REGRESJI LINIOWEJ
# ------------------------------------------------------------

plot_convergence(
    {
        "GD": gd_lr.loss_history,
        "Newton": newton_lr.loss_history,
        "Adam": adam_lr.loss_history
    },
    title="Zbieżność — regresja liniowa (MSE)",
    ylabel="MSE",
    save_path=OUT_DIR / "convergence_linear.png"
)

# wykres dopasowania tylko dla 1D
if USE_DATASET == "linear":

    plot_linear_regression_fit(
        models={
            "GD": gd_lr.theta,
            "Newton": newton_lr.theta,
            "Adam": adam_lr.theta
        },

        X=X_test_r,
        y=y_test_r,

        title="Dopasowanie regresji liniowej",

        save_path=OUT_DIR / "linear_fit.png"
    )

# ============================================================
# CZĘŚĆ 2 — REGRESJA LOGISTYCZNA
# ============================================================

if USE_DATASET != "linear":

    print()
    print("=" * 60)
    print("CZĘŚĆ 2: Regresja logistyczna")
    print("=" * 60)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_data,
        y_data,
        seed=42
    )

    X_train_c, X_test_c = standardize(X_train_c, X_test_c)

    X_train_vis = X_train_c.copy()
    X_test_vis = X_test_c.copy()

    X_train_c = log_mod.add_bias(X_train_c)
    X_test_c = log_mod.add_bias(X_test_c)

    def _loss(X, y, t):
        return log_mod.log_loss(X, y, t)

    def _grad(X, y, t):
        return log_mod.log_grad(X, y, t)

    def _hess(X, y, t):
        return log_mod.log_hessian(X, y, t)

    # ------------------------------------------------------------
    # GD
    # ------------------------------------------------------------

    gd_cls = GradientDescent(
        lr=0.3,
        max_iter=1000,
        tol=1e-8,
        line_search=True
    )

    t0 = time.perf_counter()

    gd_cls.fit(
        X_train_c,
        y_train_c,
        _loss,
        _grad
    )

    t_gd = time.perf_counter() - t0

    acc_gd = log_mod.accuracy(
        y_test_c,
        log_mod.predict(X_test_c, gd_cls.theta)
    )

    # ------------------------------------------------------------
    # NEWTON
    # ------------------------------------------------------------

    newton_cls = NewtonMethod(
        max_iter=50,
        tol=1e-10,
        reg=1e-6
    )

    t0 = time.perf_counter()

    newton_cls.fit(
        X_train_c,
        y_train_c,
        _loss,
        _grad,
        _hess
    )

    t_newton = time.perf_counter() - t0

    acc_newton = log_mod.accuracy(
        y_test_c,
        log_mod.predict(X_test_c, newton_cls.theta)
    )

    # ------------------------------------------------------------
    # ADAM
    # ------------------------------------------------------------

    adam_cls = Adam(
        lr=0.001,
        max_iter=200,
        batch_size=32,
        tol=1e-8,
        seed=42
    )

    t0 = time.perf_counter()

    adam_cls.fit(
        X_train_c,
        y_train_c,
        _loss,
        _grad
    )

    t_adam = time.perf_counter() - t0

    acc_adam = log_mod.accuracy(
        y_test_c,
        log_mod.predict(X_test_c, adam_cls.theta)
    )

    # ------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------

    print(f"  GD      — Accuracy: {acc_gd:.4f} | iteracje: {len(gd_cls.loss_history)} | czas: {t_gd:.4f}s")
    print(f"  Newton  — Accuracy: {acc_newton:.4f} | iteracje: {len(newton_cls.loss_history)} | czas: {t_newton:.4f}s")
    print(f"  Adam    — Accuracy: {acc_adam:.4f} | iteracje: {len(adam_cls.loss_history)} | czas: {t_adam:.4f}s")

    # ------------------------------------------------------------
    # PLOTY
    # ------------------------------------------------------------

    plot_convergence(
        {
            "GD": gd_cls.loss_history,
            "Newton": newton_cls.loss_history,
            "Adam": adam_cls.loss_history
        },
        title="Zbieżność — regresja logistyczna (log-loss)",
        ylabel="Log-loss",
        save_path=OUT_DIR / "convergence_logistic.png"
    )

    plot_decision_boundary(
        models={
            "GD": gd_cls.theta,
            "Newton": newton_cls.theta,
            "Adam": adam_cls.theta
        },
        X=X_test_vis,
        y=y_test_c,
        title="Granica decyzyjna — regresja logistyczna",
        save_path=OUT_DIR / "decision_boundary.png"
    )

    cms = {
        "GD": log_mod.confusion_matrix(
            y_test_c,
            log_mod.predict(X_test_c, gd_cls.theta)
        ),

        "Newton": log_mod.confusion_matrix(
            y_test_c,
            log_mod.predict(X_test_c, newton_cls.theta)
        ),

        "Adam": log_mod.confusion_matrix(
            y_test_c,
            log_mod.predict(X_test_c, adam_cls.theta)
        ),
    }

    plot_confusion_matrices(
        cms,
        title="Macierz pomyłek — regresja logistyczna",
        save_path=OUT_DIR / "confusion_matrix.png"
    )

    plot_time_vs_accuracy(
        {
            "GD": {
                "time": t_gd,
                "accuracy": acc_gd
            },

            "Newton": {
                "time": t_newton,
                "accuracy": acc_newton
            },

            "Adam": {
                "time": t_adam,
                "accuracy": acc_adam
            },
        },

        title="Czas treningu vs. Dokładność — regresja logistyczna",
        save_path=OUT_DIR / "time_vs_accuracy.png"
    )

else:

    print()
    print("=" * 60)
    print("POMINIĘTO REGRESJĘ LOGISTYCZNĄ")
    print("Powód: dataset typu linear")
    print("=" * 60)