"""Main application window — extended with Newton Method, ALS, and Music Recommendation."""
from __future__ import annotations
import sys, os
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QFileDialog, QFrame, QGridLayout,
    QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms import ALGORITHMS
from models import MODELS
from data import DATASETS, load_csv
from training_engine import TrainingWorker
from visualization import (
    plot_loss_curves, plot_regression_fit, plot_decision_boundary,
    plot_comparison, build_loss_surface, plot_loss_surface_3d, COLORS,
)
from ui.rec_panel import MusicRecommendationPanel

# ── Stylesheet ─────────────────────────────────────────────────────────────────
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
}
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: bold;
    color: #7d8590;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #58a6ff;
}
QComboBox {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
    font-size: 12px;
    min-height: 28px;
}
QComboBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    selection-background-color: #1f6feb;
}
QSpinBox, QDoubleSpinBox {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 8px;
    color: #e6edf3;
    font-size: 12px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #58a6ff; }
QSlider::groove:horizontal {
    height: 4px;
    background: #21262d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #58a6ff;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }
QPushButton {
    border-radius: 7px;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: bold;
    border: none;
    min-height: 34px;
}
QPushButton#btn_start {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1f6feb,stop:1 #388bfd);
    color: #ffffff;
}
QPushButton#btn_start:hover { background: #388bfd; }
QPushButton#btn_pause {
    background: #161b22;
    color: #d29922;
    border: 1px solid #d29922;
}
QPushButton#btn_pause:hover { background: #2d2208; }
QPushButton#btn_reset {
    background: #161b22;
    color: #f85149;
    border: 1px solid #f85149;
}
QPushButton#btn_reset:hover { background: #2d0e0e; }
QPushButton#btn_csv {
    background: #161b22;
    color: #7d8590;
    border: 1px solid #30363d;
    font-size: 11px;
    padding: 5px 12px;
}
QPushButton#btn_csv:hover { border-color: #58a6ff; color: #58a6ff; }
QTabWidget::pane {
    border: 1px solid #21262d;
    border-radius: 6px;
    background: #0d1117;
}
QTabBar::tab {
    background: #161b22;
    border: 1px solid #21262d;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 14px;
    color: #7d8590;
    font-size: 11px;
}
QTabBar::tab:selected { background: #1f6feb; color: #ffffff; }
QTabBar::tab:hover { color: #e6edf3; }
QLabel#stat_value {
    font-size: 22px;
    font-weight: bold;
    color: #58a6ff;
}
QLabel#stat_label {
    font-size: 10px;
    color: #7d8590;
    letter-spacing: 1px;
}
QScrollArea { border: none; }
"""

# ── Reusable components ────────────────────────────────────────────────────────
class StatCard(QWidget):
    def __init__(self, label: str, unit: str = ""):
        super().__init__()
        self.unit = unit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        frame = QFrame(self)
        frame.setStyleSheet(
            "QFrame { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }"
        )
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(12, 10, 12, 10)
        self.value_label = QLabel("—")
        self.value_label.setObjectName("stat_value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl = QLabel(label.upper())
        self.lbl.setObjectName("stat_label")
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fl.addWidget(self.value_label)
        fl.addWidget(self.lbl)
        layout.addWidget(frame)

    def set_value(self, v):
        if isinstance(v, float):
            self.value_label.setText(f"{v:.4f}{self.unit}")
        else:
            self.value_label.setText(f"{v}{self.unit}")


class MplCanvas(FigureCanvas):
    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        fig.patch.set_facecolor(COLORS["bg"])


# ── Algorithm info panel ──────────────────────────────────────────────────────
ALG_INFO = {
    "Gradient Descent":
        "Full-batch GD. Uses all data each step.\nθ ← θ − α·∇L\nSlow but stable. Guaranteed convergence for convex L.",
    "SGD":
        "Mini-batch SGD with momentum.\nv ← β·v + α·∇L  |  θ ← θ − v\nFaster, noisier. Good for large datasets.",
    "Adam":
        "Adaptive Moment Estimation.\nCombines momentum + RMSProp.\nFast convergence, less tuning required.",
    "Newton Method":
        "Second-order method. Uses Hessian H.\nθ ← θ − α·H⁻¹·∇L\nQuadratic convergence near optimum.\nExpensive but fewer iterations.",
    "ALS":
        "Alternating Least Squares.\nClosed-form update, not gradient-based.\nDesigned for Matrix Factorization.\nAlternates: fix V→solve U, fix U→solve V.",
}


# ── Main Window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ConvexML — Optimization Explorer v2")
        self.setMinimumSize(1340, 860)
        self._workers: list[TrainingWorker] = []
        self._histories: dict[str, list[float]] = {}
        self._stats: dict[str, dict] = {}
        self._surface_trajectories: dict[str, list] = {}
        self._X = None
        self._y = None
        self._current_model = None

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._load_dataset()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Top-level tabs (Optimization | Music Recommendation)
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.mode_tabs.setStyleSheet(
            "QTabBar::tab { font-size: 13px; padding: 10px 20px; min-width: 180px; }"
            "QTabBar::tab:selected { background: #0d419d; color: #fff; }"
        )

        # Page 1: Optimization
        opt_page = QWidget()
        opt_lay = QHBoxLayout(opt_page)
        opt_lay.setSpacing(0)
        opt_lay.setContentsMargins(0, 0, 0, 0)
        sidebar = self._build_sidebar()
        opt_lay.addWidget(sidebar)
        content = self._build_content()
        opt_lay.addWidget(content, stretch=1)
        self.mode_tabs.addTab(opt_page, "⚙  Optimization Algorithms")

        # Page 2: Music Recommendation
        self.rec_panel = MusicRecommendationPanel()
        self.mode_tabs.addTab(self.rec_panel, "♫  Music Recommendation")

        root.addWidget(self.mode_tabs)
        # Safe to call now — both sidebar and content widgets exist
        self._on_alg_changed()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(290)
        sidebar.setStyleSheet("background: #010409; border-right: 1px solid #21262d;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(14)

        logo = QLabel("◈ ConvexML")
        logo.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; letter-spacing: 2px;")
        subtitle = QLabel("Optimization Explorer v2")
        subtitle.setStyleSheet("font-size: 10px; color: #7d8590; letter-spacing: 1px;")
        layout.addWidget(logo)
        layout.addWidget(subtitle)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #21262d;"); layout.addWidget(sep)

        # Algorithm selection
        grp_alg = QGroupBox("Algorithm")
        alg_layout = QVBoxLayout(grp_alg)
        self.cb_algorithm = QComboBox()
        self.cb_algorithm.addItems(list(ALGORITHMS.keys()))
        self.cb_algorithm.currentIndexChanged.connect(self._on_alg_changed)
        alg_layout.addWidget(self.cb_algorithm)

        self.cb_compare = QComboBox()
        self.cb_compare.addItem("— Compare with —")
        self.cb_compare.addItems(list(ALGORITHMS.keys()))
        alg_layout.addWidget(self.cb_compare)

        # Algorithm info box
        self.lbl_alg_info = QLabel()
        self.lbl_alg_info.setStyleSheet(
            "font-size: 10px; color: #7d8590; background: #0d1117;"
            "border: 1px solid #21262d; border-radius: 5px; padding: 6px;"
        )
        self.lbl_alg_info.setWordWrap(True)
        alg_layout.addWidget(self.lbl_alg_info)
        layout.addWidget(grp_alg)
        # NOTE: _on_alg_changed() is called after _build_content() in _build_ui()

        # Model / Dataset
        grp_data = QGroupBox("Model & Dataset")
        data_layout = QVBoxLayout(grp_data)
        self.cb_model = QComboBox()
        self.cb_model.addItems(list(MODELS.keys()))
        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        data_layout.addWidget(self.cb_model)
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(list(DATASETS.keys()))
        self.cb_dataset.currentIndexChanged.connect(self._load_dataset)
        data_layout.addWidget(self.cb_dataset)
        btn_csv = QPushButton("⬆  Load CSV")
        btn_csv.setObjectName("btn_csv")
        btn_csv.clicked.connect(self._load_csv)
        data_layout.addWidget(btn_csv)
        layout.addWidget(grp_data)

        # Hyperparameters
        grp_hp = QGroupBox("Hyperparameters")
        hp_layout = QGridLayout(grp_hp)
        hp_layout.setHorizontalSpacing(8)

        hp_layout.addWidget(QLabel("Learning rate"), 0, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-5, 10.0); self.spin_lr.setValue(0.01)
        self.spin_lr.setSingleStep(0.005); self.spin_lr.setDecimals(5)
        hp_layout.addWidget(self.spin_lr, 0, 1)

        hp_layout.addWidget(QLabel("Iterations"), 1, 0)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000); self.spin_iter.setValue(300)
        self.spin_iter.setSingleStep(50)
        hp_layout.addWidget(self.spin_iter, 1, 1)

        hp_layout.addWidget(QLabel("Batch size"), 2, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 2000); self.spin_batch.setValue(32)
        hp_layout.addWidget(self.spin_batch, 2, 1)

        hp_layout.addWidget(QLabel("Emit every"), 3, 0)
        self.spin_emit = QSpinBox()
        self.spin_emit.setRange(1, 50); self.spin_emit.setValue(2)
        hp_layout.addWidget(self.spin_emit, 3, 1)

        layout.addWidget(grp_hp)

        # Controls
        grp_ctrl = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(grp_ctrl)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)
        ctrl_layout.addWidget(self.btn_start)
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        ctrl_layout.addWidget(self.btn_pause)
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self._reset)
        ctrl_layout.addWidget(self.btn_reset)
        layout.addWidget(grp_ctrl)

        layout.addStretch()
        ver = QLabel("v2.0.0  ·  Python + PyQt6")
        ver.setStyleSheet("font-size: 9px; color: #3d444d;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)
        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Stats row
        stats_row = QWidget()
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        self.stat_loss = StatCard("Loss")
        self.stat_acc  = StatCard("Accuracy", "%")
        self.stat_iter = StatCard("Iteration")
        self.stat_time = StatCard("Time", "s")
        for card in (self.stat_loss, self.stat_acc, self.stat_iter, self.stat_time):
            stats_layout.addWidget(card)
        layout.addWidget(stats_row)

        # Chart tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self.fig_loss = Figure(tight_layout=True)
        self.canvas_loss = MplCanvas(self.fig_loss)
        self.tabs.addTab(self.canvas_loss, "📉 Loss Curve")

        self.fig_fit = Figure(tight_layout=True)
        self.canvas_fit = MplCanvas(self.fig_fit)
        self.tabs.addTab(self.canvas_fit, "🎯 Model Fit")

        self.fig_3d = Figure(tight_layout=True)
        self.canvas_3d = MplCanvas(self.fig_3d)
        self.tabs.addTab(self.canvas_3d, "🌐 Loss Surface 3D")

        self.fig_compare = Figure(tight_layout=True)
        self.canvas_compare = MplCanvas(self.fig_compare)
        self.tabs.addTab(self.canvas_compare, "⚖ Comparison")

        self.status_label = QLabel("Ready. Configure and press ▶ Start.")
        self.status_label.setStyleSheet(
            "font-size: 11px; color: #7d8590; padding: 4px 8px;"
            "border-top: 1px solid #21262d; background: #010409;"
        )
        layout.addWidget(self.status_label)
        return content

    # ── Algorithm info ─────────────────────────────────────────────────────────
    def _on_alg_changed(self, _=None):
        name = self.cb_algorithm.currentText()
        info = ALG_INFO.get(name, "")
        self.lbl_alg_info.setText(info)
        # Newton works best on small datasets with Logistic Regression
        if name == "Newton Method":
            self.spin_iter.setValue(30)
            self.spin_lr.setValue(1.0)
            self.spin_batch.setValue(1000)
        elif name == "Adam":
            self.spin_lr.setValue(0.001)
        elif name == "Gradient Descent":
            self.spin_lr.setValue(0.01)
            self.spin_iter.setValue(300)

    # ── Dataset helpers ────────────────────────────────────────────────────────
    def _load_dataset(self):
        name = self.cb_dataset.currentText()
        if name not in DATASETS:
            return
        self._X, self._y = DATASETS[name]()
        self._on_model_changed()
        self._reset()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if path:
            try:
                self._X, self._y = load_csv(path)
                self.status_label.setText(f"Loaded: {Path(path).name}  ({len(self._y)} rows)")
                self._reset()
            except Exception as e:
                self.status_label.setText(f"Error loading CSV: {e}")

    def _on_model_changed(self):
        model_name = self.cb_model.currentText()
        model_cls = MODELS[model_name]
        self._current_model = model_cls()
        if self._X is not None and self._y is not None:
            self._current_model.fit_data(self._X, self._y)
        is_cls = self._current_model.task == "classification"
        self.stat_acc.value_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {'#3fb950' if is_cls else '#3d444d'};"
        )

    # ── Training ───────────────────────────────────────────────────────────────
    def _start(self):
        if self._X is None:
            self._load_dataset()
        self._reset_workers()
        self._histories.clear()
        self._stats.clear()
        self._surface_trajectories.clear()

        algorithms_to_run = [self.cb_algorithm.currentText()]
        compare_name = self.cb_compare.currentText()
        if compare_name and compare_name != "— Compare with —":
            algorithms_to_run.append(compare_name)

        for alg_name in algorithms_to_run:
            if alg_name == "ALS":
                self.status_label.setText(
                    "ALS is for Matrix Factorization — use Music Recommendation tab."
                )
                continue
            self._launch_worker(alg_name)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def _launch_worker(self, alg_name: str):
        model_name = self.cb_model.currentText()
        model = MODELS[model_name]()
        model.fit_data(self._X, self._y)

        lr = self.spin_lr.value()
        optimizer = ALGORITHMS[alg_name](learning_rate=lr)
        optimizer.history["params"].append(model.params.copy())

        worker = TrainingWorker(
            model=model, optimizer=optimizer,
            X=self._X, y=self._y,
            n_iterations=self.spin_iter.value(),
            batch_size=self.spin_batch.value(),
            emit_every=self.spin_emit.value(),
        )
        worker.step_done.connect(lambda payload, n=alg_name: self._on_step(payload, n))
        worker.finished.connect(lambda payload, n=alg_name: self._on_finished(payload, n))
        worker.error.connect(lambda msg: self.status_label.setText(f"Error: {msg}"))
        worker.start()
        self._workers.append(worker)

    @pyqtSlot(dict)
    def _on_step(self, payload: dict, name: str):
        if name not in self._histories:
            self._histories[name] = []
        self._histories[name].append(payload["loss"])
        if name == self.cb_algorithm.currentText():
            self.stat_loss.set_value(payload["loss"])
            self.stat_iter.set_value(payload["iteration"])
            self.stat_time.set_value(round(payload["elapsed"], 2))
            if "accuracy" in payload:
                self.stat_acc.set_value(round(payload["accuracy"] * 100, 1))
        params = payload["params"]
        if name not in self._surface_trajectories:
            self._surface_trajectories[name] = []
        if len(params) >= 2:
            self._surface_trajectories[name].append((params[0], params[1], payload["loss"]))
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._update_loss_tab()
        elif idx == 2:
            self._update_3d_tab()

    @pyqtSlot(dict)
    def _on_finished(self, payload: dict, name: str):
        self._stats[name] = {
            "final_loss": payload["loss"],
            "time": payload["elapsed"],
            "iterations": payload["iteration"],
            "accuracy": payload.get("accuracy"),
        }
        self.status_label.setText(
            f"✔ {name} finished — loss: {payload['loss']:.5f}  "
            f"| {payload['iteration']} iters  | {payload['elapsed']:.2f}s"
        )
        if len(self._stats) >= len(self._workers):
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self._refresh_all_tabs()

    def _refresh_all_tabs(self):
        self._update_loss_tab()
        self._update_fit_tab()
        self._update_3d_tab()
        self._update_compare_tab()

    def _update_loss_tab(self):
        if not self._histories:
            return
        plot_loss_curves(self._histories, fig=self.fig_loss)
        self.canvas_loss.draw_idle()

    def _update_fit_tab(self):
        if self._X is None or not self._workers:
            return
        self.fig_fit.clear()
        if self._workers[0].model.task == "regression":
            models_params = {w.optimizer.name: w.model.params.copy() for w in self._workers}
            plot_regression_fit(self._X, self._y, models_params, fig=self.fig_fit)
        else:
            alg_name = self.cb_algorithm.currentText()
            plot_decision_boundary(
                self._X, self._y, self._workers[0].model,
                optimizer_name=alg_name, fig=self.fig_fit
            )
        self.canvas_fit.draw_idle()

    def _update_3d_tab(self):
        if self._X is None or not self._workers:
            return
        w = self._workers[0]
        try:
            B, W, Z = build_loss_surface(w.model, self._X, self._y, resolution=35)
            plot_loss_surface_3d(B, W, Z, trajectories=self._surface_trajectories, fig=self.fig_3d)
            self.canvas_3d.draw_idle()
        except Exception:
            pass

    def _update_compare_tab(self):
        if len(self._stats) < 1:
            return
        plot_comparison(self._stats, fig=self.fig_compare)
        self.canvas_compare.draw_idle()

    # ── Controls ───────────────────────────────────────────────────────────────
    def _pause(self):
        for w in self._workers:
            w.pause()
        paused = any(w._paused for w in self._workers)
        self.btn_pause.setText("▶  Resume" if paused else "⏸  Pause")

    def _reset(self):
        self._reset_workers()
        self._histories.clear()
        self._stats.clear()
        self._surface_trajectories.clear()
        for fig, canvas in [
            (self.fig_loss, self.canvas_loss), (self.fig_fit, self.canvas_fit),
            (self.fig_3d, self.canvas_3d), (self.fig_compare, self.canvas_compare),
        ]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(COLORS["surface"])
            ax.text(0.5, 0.5, "Run an experiment to see results",
                    transform=ax.transAxes, ha="center", va="center",
                    color=COLORS["text"], fontsize=11, alpha=0.5)
            fig.patch.set_facecolor(COLORS["bg"])
            canvas.draw_idle()
        for card in (self.stat_loss, self.stat_acc, self.stat_iter, self.stat_time):
            card.set_value("—")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.status_label.setText("Reset. Configure and press ▶ Start.")

    def _reset_workers(self):
        for w in self._workers:
            w.stop(); w.wait(500)
        self._workers.clear()

    def closeEvent(self, event):
        self._reset_workers()
        self.rec_panel.closeEvent(event)
        super().closeEvent(event)

    def _setup_tab_refresh(self):
        self.tabs.currentChanged.connect(
            lambda _: (self._update_loss_tab() if self.tabs.currentIndex() == 0
                       else self._update_3d_tab() if self.tabs.currentIndex() == 2
                       else None)
        )
