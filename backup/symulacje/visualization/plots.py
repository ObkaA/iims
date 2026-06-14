import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

COLORS = {"GD": "#2196F3", "Newton": "#F44336", "Adam": "#4CAF50"}


def plot_convergence(histories: dict, title="Zbieżność algorytmów",
                     ylabel="Wartość funkcji straty", save_path=None):
    """
    Krzywe zbieznosci: wartosc funkcji straty vs. iteracja.

    histories : dict { nazwa_metody : lista_wartosci_straty }
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        ax.plot(hist, label=name, color=COLORS.get(name), linewidth=2)
    ax.set_xlabel("Iteracja / epoka", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_decision_boundary(models: dict, X, y,
                           title="Granica decyzyjna — regresja logistyczna",
                           save_path=None):
    """
    Granica decyzyjna dla danych 2D.

    models : dict { nazwa_metody : theta (p,) } — theta BEZ biasu lub z biasem
             X musi miec 2 cechy (bez kolumny jedynek)
    """
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300)
    )
    grid = np.c_[np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, theta) in zip(axes, models.items()):
        from backup.models.logistic_regression import sigmoid
        zz = sigmoid(grid @ theta).reshape(xx1.shape)
        ax.contourf(xx1, xx2, zz, levels=[0, 0.5, 1],
                    colors=["#BBDEFB", "#FFCDD2"], alpha=0.6)
        ax.contour(xx1, xx2, zz, levels=[0.5],
                   colors=[COLORS.get(name, "black")], linewidths=2)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#2196F3",
                   edgecolors="k", s=20, label="Klasa 0", alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#F44336",
                   edgecolors="k", s=20, label="Klasa 1", alpha=0.7)
        ax.set_title(name, fontsize=12, fontweight="bold",
                     color=COLORS.get(name, "black"))
        ax.set_xlabel("x₁", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("x₂", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_confusion_matrices(cms: dict, title="Macierz pomyłek",
                            save_path=None):
    """
    cms : dict { nazwa_metody : ndarray [[TN, FP],[FN, TP]] }
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    labels = ["Negatywna", "Pozytywna"]
    for ax, (name, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], labels, fontsize=10)
        ax.set_yticks([0, 1], labels, fontsize=10)
        ax.set_xlabel("Predykcja", fontsize=10)
        ax.set_ylabel("Rzeczywistość", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold",
                     color=COLORS.get(name, "black"))
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_time_vs_accuracy(results: dict, title="Czas CPU vs. Dokładność",
                          save_path=None):
    """
    results : dict { nazwa_metody : {"time": float [s], "accuracy": float} }
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        ax.scatter(r["time"], r["accuracy"],
                   s=160, color=COLORS.get(name), zorder=5,
                   edgecolors="k", linewidths=0.8)
        ax.annotate(name, (r["time"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=11, color=COLORS.get(name, "black"))

    ax.set_xlabel("Czas treningu [s]", fontsize=12)
    ax.set_ylabel("Dokładność (accuracy)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig

def plot_linear_regression_fit(models, X, y,
                               title="Dopasowanie regresji liniowej",
                               save_path=None):
    """
    Wykres:
    - punkty danych
    - proste regresji dla różnych optimizerów

    models : dict { nazwa : theta }
    X : dane Z BIASem
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    # usuwamy bias
    x = X[:, 1]

    # scatter danych
    ax.scatter(
        x,
        y,
        color="black",
        alpha=0.35,
        s=15,
        label="Dane"
    )

    # zakres do rysowania prostych
    x_line = np.linspace(x.min(), x.max(), 300)

    for name, theta in models.items():

        # theta[0] = bias
        # theta[1] = współczynnik kierunkowy
        y_line = theta[0] + theta[1] * x_line

        ax.plot(
            x_line,
            y_line,
            linewidth=2.5,
            color=COLORS.get(name),
            label=name
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(fontsize=11)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig
