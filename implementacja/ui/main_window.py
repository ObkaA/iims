"""Main application window — ConvexML v2."""
from __future__ import annotations
import sys, os
import re
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QFileDialog, QFrame, QGridLayout,
    QSizePolicy, QScrollArea, QTextEdit, QInputDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms import ALGORITHMS
from models import MODELS
from data import (
    DATASETS, DATASET_TASKS, CSVDataError, inspect_csv, load_csv,
    standardize_from_training, train_test_split,
)
from metrics import binary_confusion_matrix
from training_engine import TrainingWorker
from visualization import (
    plot_loss_curves, plot_regression_fit, plot_decision_boundary,
    plot_comparison, plot_confusion_matrix, build_loss_surface,
    plot_loss_surface_3d, COLORS,
)
from visualization.adam_diagnostics import (
    analyze_optimizer_comparison,
    plot_optimizer_comparison,
)
from ui.rec_panel import MusicRecommendationPanel

OPTIMIZATION_ALGORITHMS = {
    name: optimizer for name, optimizer in ALGORITHMS.items() if name != "ALS"
}
COMPARE_PLACEHOLDER = "— Compare with —"

# ── Stylesheet ─────────────────────────────────────────────────────────────────
SS = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    font-size: 10px;
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
    padding: 5px 10px;
    color: #e6edf3;
    min-height: 26px;
}
QComboBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    selection-background-color: #1f6feb;
    padding: 2px;
}
QSpinBox, QDoubleSpinBox {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e6edf3;
    min-height: 24px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #58a6ff; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #21262d; border: none; width: 16px;
}
QSlider::groove:horizontal {
    height: 4px; background: #21262d; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #58a6ff; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }
QPushButton {
    border-radius: 7px; padding: 7px 16px;
    font-size: 12px; font-weight: bold;
    border: none; min-height: 32px;
}
QPushButton#btn_start {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1f6feb,stop:1 #388bfd);
    color: #fff;
}
QPushButton#btn_start:hover  { background: #388bfd; }
QPushButton#btn_start:disabled { background: #21262d; color: #7d8590; }
QPushButton#btn_pause {
    background: #161b22; color: #d29922; border: 1px solid #d29922;
}
QPushButton#btn_pause:hover { background: #2d2208; }
QPushButton#btn_reset {
    background: #161b22; color: #f85149; border: 1px solid #f85149;
}
QPushButton#btn_reset:hover { background: #2d0e0e; }
QPushButton#btn_csv {
    background: #161b22; color: #7d8590;
    border: 1px solid #30363d; font-size: 11px; padding: 4px 10px;
}
QPushButton#btn_csv:hover { border-color: #58a6ff; color: #58a6ff; }
QPushButton#btn_generate {
    background: #0d419d; color: #fff;
    border: 1px solid #1f6feb; font-size: 11px; padding: 4px 10px;
}
QPushButton#btn_generate:hover { background: #1f6feb; }
QPushButton#btn_generate:disabled {
    background: #161b22; color: #484f58; border-color: #21262d;
}
QTabWidget::pane {
    border: 1px solid #21262d; border-radius: 6px; background: #0d1117;
}
QTabBar::tab {
    background: #161b22; border: 1px solid #21262d;
    border-bottom: none; border-radius: 4px 4px 0 0;
    padding: 6px 12px; color: #7d8590; font-size: 11px;
}
QTabBar::tab:selected { background: #1f6feb; color: #fff; }
QTabBar::tab:hover    { color: #e6edf3; }
QLabel#stat_value {
    font-size: 20px; font-weight: bold; color: #58a6ff;
}
QLabel#stat_label {
    font-size: 9px; color: #7d8590; letter-spacing: 1px;
}
QScrollArea  { border: none; }
QScrollBar:vertical {
    background: #010409;
    width: 9px;
    margin: 2px 0;
}
QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #58a6ff; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
QTextEdit {
    background: #0d1117; color: #c9d1d9;
    border: 1px solid #21262d; border-radius: 6px;
    font-size: 11px; padding: 6px;
    selection-background-color: #1f6feb;
}
"""


# ── Reusable widgets ──────────────────────────────────────────────────────────
class StatCard(QWidget):
    def __init__(self, label: str, unit: str = ""):
        super().__init__()
        self.unit = unit
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#161b22;border:1px solid #21262d;border-radius:8px;}"
        )
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(10, 8, 10, 8)
        self.val = QLabel("—")
        self.val.setObjectName("stat_value")
        self.val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl = QLabel(label.upper())
        self.lbl.setObjectName("stat_label")
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fl.addWidget(self.val)
        fl.addWidget(self.lbl)
        lay.addWidget(frame)

    def set_value(self, v):
        if isinstance(v, float):
            self.val.setText(f"{v:.4f}{self.unit}")
        else:
            self.val.setText(f"{v}{self.unit}")

    def set_metric(self, label: str, unit: str = ""):
        self.lbl.setText(label.upper())
        self.unit = unit


class MplCanvas(FigureCanvas):
    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setMinimumHeight(180)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        fig.patch.set_facecolor(COLORS["bg"])


# ── ALG descriptions ──────────────────────────────────────────────────────────
ALG_INFO = {
    "Gradient Descent":
        "θ ← θ − α·∇L\nFull-batch. Stable, O(1/t) convergence.\nNo memory of past gradients.",
    "SGD":
        "v ← β·v + α·∇L  |  θ ← θ − v\nMini-batch + Polyak momentum β.\nFaster than GD; momentum dampens oscillations.",
    "Adam":
        "m ← β₁m+(1-β₁)g  |  v ← β₂v+(1-β₂)g²\nθ ← θ − α·m̂/√v̂\nAdaptive per-param step. Fast but see ⚠ tab!",
    "Newton Method":
        "(H+λI)·Δθ = ∇L  |  θ ← θ − α·Δθ\nUses Hessian. Quadratic convergence.\n~10 iters vs ~300 for GD on logistic regression.",
    "ALS":
        "uᵢ = (VᵢᵀVᵢ+λI)⁻¹Vᵢᵀrᵢ\nClosed-form, not gradient-based.\nFor Matrix Factorization — use ♫ tab.",
}

# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ConvexML — Optimization Explorer v2")
        self.setMinimumSize(1360, 860)
        self._workers:   list[TrainingWorker] = []
        self._histories: dict = {}
        self._stats:     dict = {}
        self._surface_trajectories: dict = {}
        self._X = self._y = None
        self._X_model = self._X_train = self._X_test = None
        self._y_train = self._y_test = None
        self._csv_dataset_name: str | None = None
        self._csv_task: str | None = None
        self._csv_X = self._csv_y = None
        self._finished_workers = 0
        self._current_model = None
        self._adam_diagnostic: dict | None = None

        self.setStyleSheet(SS)
        self._build_ui()
        self._load_dataset()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.mode_tabs.setStyleSheet(
            "QTabBar::tab{font-size:13px;padding:10px 22px;min-width:200px;}"
            "QTabBar::tab:selected{background:#0d419d;color:#fff;}"
        )

        # Page 1: Optimization
        opt_page = QWidget()
        opt_lay  = QHBoxLayout(opt_page)
        opt_lay.setSpacing(0)
        opt_lay.setContentsMargins(0, 0, 0, 0)
        opt_lay.addWidget(self._build_sidebar())
        opt_lay.addWidget(self._build_content(), stretch=1)
        self.mode_tabs.addTab(opt_page, "⚙  Optimization")

        # Page 2: controlled Adam vs SGD vs Gradient Descent comparison
        self.mode_tabs.addTab(self._build_adam_diagnostics_tab(), "✓  Adam Check")

        # Page 3: Music Recommendation
        self.rec_panel = MusicRecommendationPanel()
        self.mode_tabs.addTab(self.rec_panel, "♫  Music Recommendation")

        root.addWidget(self.mode_tabs)
        self._on_alg_changed()  # safe — all widgets exist now

    # ── Sidebar ────────────────────────────────────────────────────────────────
    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setFixedWidth(300)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        sb = QWidget()
        sb.setMinimumWidth(280)
        sb.setStyleSheet("background:#010409;border-right:1px solid #21262d;")
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(14, 18, 14, 14)
        lay.setSpacing(12)

        # Logo
        logo = QLabel("◈ ConvexML")
        logo.setStyleSheet(
            "font-size:17px;font-weight:bold;color:#58a6ff;letter-spacing:2px;"
        )
        sub = QLabel("Optimization Explorer v2")
        sub.setStyleSheet("font-size:9px;color:#7d8590;letter-spacing:1px;")
        lay.addWidget(logo)
        lay.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#21262d;")
        lay.addWidget(sep)

        # Algorithm group
        grp_alg = QGroupBox("Algorithm")
        al = QVBoxLayout(grp_alg)
        self.cb_algorithm = QComboBox()
        self.cb_algorithm.addItems(list(OPTIMIZATION_ALGORITHMS.keys()))
        self.cb_algorithm.currentIndexChanged.connect(self._on_alg_changed)
        al.addWidget(self.cb_algorithm)

        self.cb_compare = QComboBox()
        self.cb_compare.addItem(COMPARE_PLACEHOLDER)
        self.cb_compare.addItems(list(OPTIMIZATION_ALGORITHMS.keys()))
        al.addWidget(self.cb_compare)

        self.lbl_alg_info = QLabel()
        self.lbl_alg_info.setStyleSheet(
            "font-size:10px;color:#8b949e;background:#0d1117;"
            "border:1px solid #21262d;border-radius:5px;padding:6px;"
        )
        self.lbl_alg_info.setWordWrap(True)
        self.lbl_alg_info.setMinimumHeight(60)
        al.addWidget(self.lbl_alg_info)
        lay.addWidget(grp_alg)

        # Model / Dataset group
        grp_data = QGroupBox("Model & Dataset")
        dl = QVBoxLayout(grp_data)
        self.cb_model = QComboBox()
        self.cb_model.addItems(list(MODELS.keys()))
        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        dl.addWidget(QLabel("Model:"))
        dl.addWidget(self.cb_model)
        self.ridge_lambda_row = QWidget()
        ridge_lambda_layout = QHBoxLayout(self.ridge_lambda_row)
        ridge_lambda_layout.setContentsMargins(0, 0, 0, 0)
        ridge_lambda_layout.setSpacing(8)
        self.lbl_ridge_lambda = QLabel("Regularization λ")
        self.spin_ridge_lambda = QDoubleSpinBox()
        self.spin_ridge_lambda.setRange(0.0, 1000.0)
        self.spin_ridge_lambda.setValue(1.0)
        self.spin_ridge_lambda.setSingleStep(0.1)
        self.spin_ridge_lambda.setDecimals(4)
        self.spin_ridge_lambda.valueChanged.connect(self._on_model_changed)
        ridge_lambda_layout.addWidget(self.lbl_ridge_lambda)
        ridge_lambda_layout.addWidget(self.spin_ridge_lambda, stretch=1)
        dl.addWidget(self.ridge_lambda_row)
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems([
            name for name in DATASETS if DATASET_TASKS[name] == "regression"
        ])
        self.cb_dataset.currentIndexChanged.connect(self._load_dataset)
        dl.addWidget(QLabel("Dataset:"))
        dl.addWidget(self.cb_dataset)
        dataset_size_row = QWidget()
        dataset_size_layout = QHBoxLayout(dataset_size_row)
        dataset_size_layout.setContentsMargins(0, 0, 0, 0)
        dataset_size_layout.setSpacing(8)
        dataset_size_layout.addWidget(QLabel("Samples"))
        self.spin_dataset_size = QSpinBox()
        self.spin_dataset_size.setRange(20, 100000)
        self.spin_dataset_size.setValue(200)
        self.spin_dataset_size.setSingleStep(100)
        self.spin_dataset_size.setGroupSeparatorShown(True)
        dataset_size_layout.addWidget(self.spin_dataset_size, stretch=1)
        dl.addWidget(dataset_size_row)
        self.btn_generate_dataset = QPushButton("↻  Generate dataset")
        self.btn_generate_dataset.setObjectName("btn_generate")
        self.btn_generate_dataset.clicked.connect(self._load_dataset)
        dl.addWidget(self.btn_generate_dataset)
        self.btn_csv = QPushButton("⬆  Load CSV")
        self.btn_csv.setObjectName("btn_csv")
        self.btn_csv.clicked.connect(self._load_csv)
        dl.addWidget(self.btn_csv)
        lay.addWidget(grp_data)

        # Hyperparameters
        grp_hp = QGroupBox("Hyperparameters")
        hl = QGridLayout(grp_hp)
        hl.setColumnStretch(1, 1)
        hl.setHorizontalSpacing(8)
        hl.setVerticalSpacing(6)

        hl.addWidget(QLabel("Learning rate α"), 0, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-5, 10.0)
        self.spin_lr.setValue(0.01)
        self.spin_lr.setSingleStep(0.005)
        self.spin_lr.setDecimals(5)
        hl.addWidget(self.spin_lr, 0, 1)

        hl.addWidget(QLabel("Iterations"), 1, 0)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000)
        self.spin_iter.setValue(300)
        self.spin_iter.setSingleStep(50)
        hl.addWidget(self.spin_iter, 1, 1)

        hl.addWidget(QLabel("Batch size"), 2, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4000)
        self.spin_batch.setValue(32)
        hl.addWidget(self.spin_batch, 2, 1)

        hl.addWidget(QLabel("Emit every"), 3, 0)
        self.spin_emit = QSpinBox()
        self.spin_emit.setRange(1, 50)
        self.spin_emit.setValue(2)
        hl.addWidget(self.spin_emit, 3, 1)

        hl.addWidget(QLabel("Test split %"), 4, 0)
        self.spin_test_split = QSpinBox()
        self.spin_test_split.setRange(10, 50)
        self.spin_test_split.setValue(20)
        self.spin_test_split.setSuffix(" %")
        hl.addWidget(self.spin_test_split, 4, 1)

        lay.addWidget(grp_hp)

        # Controls
        grp_ctrl = QGroupBox("Controls")
        cl = QVBoxLayout(grp_ctrl)
        cl.setSpacing(6)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)
        cl.addWidget(self.btn_start)
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        cl.addWidget(self.btn_pause)
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self._reset)
        cl.addWidget(self.btn_reset)
        lay.addWidget(grp_ctrl)

        lay.addStretch()
        ver = QLabel("v2.1 · PyQt6 · NumPy · from scratch")
        ver.setStyleSheet("font-size:8px;color:#3d444d;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ver)
        sb.setMinimumHeight(sb.sizeHint().height())
        scroll.setWidget(sb)
        return scroll

    # ── Main content (optimization tab) ───────────────────────────────────────
    def _build_content(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        # Stat cards row
        row = QWidget()
        row.setMaximumHeight(90)
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.stat_loss = StatCard("Loss")
        self.stat_acc  = StatCard("Accuracy", "%")
        self.stat_iter = StatCard("Iteration")
        self.stat_time = StatCard("Time", "s")
        for c in (self.stat_loss, self.stat_acc, self.stat_iter, self.stat_time):
            rl.addWidget(c)
        lay.addWidget(row)

        # Chart tabs
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, stretch=1)

        for attr, title in [
            ("loss",    "📉 Loss Curve"),
            ("fit",     "🎯 Model Fit"),
            ("3d",      "🌐 Loss Surface 3D"),
            ("compare", "⚖ Comparison"),
        ]:
            fig    = Figure(tight_layout=True)
            canvas = MplCanvas(fig)
            setattr(self, f"fig_{attr}",    fig)
            setattr(self, f"canvas_{attr}", canvas)
            self.tabs.addTab(canvas, title)

        # Status bar
        self.status_label = QLabel("Ready — configure and press ▶ Start.")
        self.status_label.setStyleSheet(
            "font-size:10px;color:#7d8590;padding:4px 8px;"
            "border-top:1px solid #21262d;background:#010409;"
        )
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(34)
        lay.addWidget(self.status_label)
        return w

    # ── Adam vs baseline optimizers on the current experiment ────────────────
    def _build_adam_diagnostics_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background:#010409;border-right:1px solid #21262d;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 20, 18, 16)
        side_layout.setSpacing(12)

        title = QLabel("✓  ADAM CHECK")
        title.setStyleSheet(
            "font-size:15px;font-weight:bold;color:#58a6ff;letter-spacing:1px;"
        )
        side_layout.addWidget(title)

        intro = QLabel(
            "Porównanie Adam vs SGD vs Gradient Descent na tych samych danych, "
            "modelu, podziale train/test i learning rate."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size:10px;color:#8b949e;line-height:1.4;")
        side_layout.addWidget(intro)

        self.btn_run_adam_check = QPushButton("▶  Compare Adam vs SGD vs GD")
        self.btn_run_adam_check.setObjectName("btn_start")
        self.btn_run_adam_check.clicked.connect(self._run_adam_diagnosis)
        side_layout.addWidget(self.btn_run_adam_check)

        verdict_box = QGroupBox("Automatic verdict")
        verdict_layout = QVBoxLayout(verdict_box)
        self.lbl_adam_verdict = QLabel("NO DATA")
        self.lbl_adam_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_adam_verdict.setStyleSheet(
            "font-size:25px;font-weight:bold;color:#7d8590;padding:10px;"
            "background:#161b22;border:1px solid #30363d;border-radius:8px;"
        )
        verdict_layout.addWidget(self.lbl_adam_verdict)
        side_layout.addWidget(verdict_box)

        self.lbl_adam_context = QLabel("No Adam experiment yet.")
        self.lbl_adam_context.setWordWrap(True)
        self.lbl_adam_context.setStyleSheet(
            "font-size:10px;color:#8b949e;background:#0d1117;"
            "border:1px solid #21262d;border-radius:6px;padding:8px;"
        )
        side_layout.addWidget(self.lbl_adam_context)

        reasons_box = QGroupBox("What the plots say")
        reasons_layout = QVBoxLayout(reasons_box)
        self.txt_adam_reasons = QTextEdit()
        self.txt_adam_reasons.setReadOnly(True)
        self.txt_adam_reasons.setPlainText(
            "Run the comparison to check Adam against SGD and Gradient Descent."
        )
        reasons_layout.addWidget(self.txt_adam_reasons)
        side_layout.addWidget(reasons_box, stretch=1)

        layout.addWidget(sidebar)

        chart_panel = QWidget()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        self.fig_adam_diag = Figure()
        self.canvas_adam_diag = MplCanvas(self.fig_adam_diag)
        chart_layout.addWidget(self.canvas_adam_diag, stretch=1)
        layout.addWidget(chart_panel, stretch=1)

        self._set_adam_diagnostic_placeholder(
            "Uruchom porównanie na bieżącym modelu i datasecie."
        )
        return page

    def _set_adam_diagnostic_placeholder(self, message: str):
        diagnostic = analyze_optimizer_comparison({})
        diagnostic["reasons"] = [message]
        self._adam_diagnostic = diagnostic
        if hasattr(self, "fig_adam_diag"):
            plot_optimizer_comparison(diagnostic, self.fig_adam_diag)
            self.canvas_adam_diag.draw_idle()
        if hasattr(self, "lbl_adam_verdict"):
            self.lbl_adam_verdict.setText("NO DATA")
            self.lbl_adam_verdict.setStyleSheet(
                "font-size:25px;font-weight:bold;color:#7d8590;padding:10px;"
                "background:#161b22;border:1px solid #30363d;border-radius:8px;"
            )
            self.txt_adam_reasons.setPlainText(message)

    def _run_adam_diagnosis(self):
        if any(worker.isRunning() for worker in self._workers):
            self.txt_adam_reasons.setPlainText(
                "Poczekaj na zakończenie bieżącego eksperymentu."
            )
            return
        current_lr = self.spin_lr.value()
        self.cb_algorithm.setCurrentText("Adam")
        self.spin_lr.setValue(current_lr)
        self.cb_compare.setCurrentText(COMPARE_PLACEHOLDER)
        self.lbl_adam_verdict.setText("RUNNING…")
        self.lbl_adam_verdict.setStyleSheet(
            "font-size:25px;font-weight:bold;color:#58a6ff;padding:10px;"
            "background:#161b22;border:1px solid #58a6ff;border-radius:8px;"
        )
        self.txt_adam_reasons.setPlainText(
            "Trwa kontrolowane porównanie Adam, SGD i Gradient Descent."
        )
        self.btn_run_adam_check.setEnabled(False)
        self._start(forced_algorithms=["Adam", "SGD", "Gradient Descent"])
        if not any(worker.isRunning() for worker in self._workers):
            self.btn_run_adam_check.setEnabled(True)

    def _update_adam_diagnostics(self):
        histories = {
            item.optimizer.name: item.optimizer.history
            for item in self._workers
            if item.optimizer.name in {"Adam", "SGD", "Gradient Descent"}
            and item.optimizer.history["loss"]
        }
        if "Adam" not in histories:
            self._set_adam_diagnostic_placeholder(
                "Ostatni eksperyment nie zawierał Adama. Użyj przycisku porównania."
            )
            return

        diagnostic = analyze_optimizer_comparison(histories)
        self._adam_diagnostic = diagnostic
        plot_optimizer_comparison(diagnostic, self.fig_adam_diag)
        self.canvas_adam_diag.draw_idle()

        verdict = diagnostic["verdict"]
        color = diagnostic["color"]
        self.lbl_adam_verdict.setText(verdict)
        self.lbl_adam_verdict.setStyleSheet(
            f"font-size:25px;font-weight:bold;color:{color};padding:10px;"
            f"background:#161b22;border:1px solid {color};border-radius:8px;"
        )

        metrics = diagnostic.get("metrics", {})
        reduction = metrics.get("loss_reduction")
        reduction_text = "—" if reduction is None else f"{reduction * 100:.1f}%"
        methods = ", ".join(histories)
        self.lbl_adam_context.setText(
            f"Model: {self.cb_model.currentText()}\n"
            f"Dataset: {self.cb_dataset.currentText()}\n"
            f"Methods: {methods}\n"
            f"Shared learning rate: {self.spin_lr.value():g}\n"
            f"Iterations: {metrics.get('iterations', 0)}\n"
            f"Adam loss reduction: {reduction_text}"
        )
        reason_lines = [f"- {reason}" for reason in diagnostic["reasons"]]
        if metrics:
            reason_lines.extend([
                "",
                f"Best loss: {metrics['best_loss']:.6g} (iteration {metrics['best_iteration']})",
                f"Final trend loss: {metrics['final_loss']:.6g}",
                f"Gradient ratio (end/start): {metrics['gradient_ratio']:.3g}",
                f"Step ratio (end/start): {metrics['step_ratio']:.3g}",
            ])
            final_losses = metrics.get("final_losses", {})
            if final_losses:
                reason_lines.extend([
                    "",
                    "Final trend loss:",
                    *[f"  {name}: {loss:.6g}" for name, loss in final_losses.items()],
                ])
        self.txt_adam_reasons.setPlainText("\n".join(reason_lines))
        self.btn_run_adam_check.setEnabled(True)

    # ══════════════════════════════════════════════════════════════════════════
    # Optimization tab logic
    # ══════════════════════════════════════════════════════════════════════════
    def _on_alg_changed(self, _=None):
        name = self.cb_algorithm.currentText()
        self.lbl_alg_info.setText(ALG_INFO.get(name, ""))
        previous_compare = self.cb_compare.currentText()
        compare_options = [
            candidate for candidate in OPTIMIZATION_ALGORITHMS if candidate != name
        ]
        self.cb_compare.blockSignals(True)
        self.cb_compare.clear()
        self.cb_compare.addItem(COMPARE_PLACEHOLDER)
        self.cb_compare.addItems(compare_options)
        if previous_compare in compare_options:
            self.cb_compare.setCurrentText(previous_compare)
        self.cb_compare.blockSignals(False)
        # Sensible defaults per algorithm
        if name == "Newton Method":
            self.spin_iter.setValue(30)
            self.spin_lr.setValue(1.0)
            self.spin_batch.setValue(2000)
        elif name == "Adam":
            self.spin_lr.setValue(0.001)
            self.spin_iter.setValue(300)
        elif name == "Gradient Descent":
            self.spin_lr.setValue(0.01)
            self.spin_iter.setValue(300)
        elif name == "SGD":
            self.spin_lr.setValue(0.01)
            self.spin_iter.setValue(300)

    def _refresh_dataset_choices(self, task: str, preferred: str | None = None) -> bool:
        previous = self.cb_dataset.currentText()
        options = [name for name in DATASETS if DATASET_TASKS[name] == task]
        if self._csv_dataset_name and self._csv_task == task:
            options.append(self._csv_dataset_name)
        selected = preferred or previous
        if selected not in options:
            selected = options[0]
        self.cb_dataset.blockSignals(True)
        self.cb_dataset.clear()
        self.cb_dataset.addItems(options)
        self.cb_dataset.setCurrentText(selected)
        self.cb_dataset.blockSignals(False)
        return selected != previous

    def _load_dataset(self, _=None):
        name = self.cb_dataset.currentText()
        if name == self._csv_dataset_name and self._csv_X is not None:
            self._X, self._y = self._csv_X.copy(), self._csv_y.copy()
            self._X_model = self._X_train = self._X_test = None
            self._y_train = self._y_test = None
            self.spin_dataset_size.setEnabled(False)
            self.btn_generate_dataset.setEnabled(False)
            self._on_model_changed()
            self._reset()
            self.status_label.setText(
                f"Restored {name}: {len(self._y):,} samples × {self._X.shape[1]} features"
            )
            return
        if name not in DATASETS:
            return
        model_task = self._create_model().task
        if DATASET_TASKS[name] != model_task:
            self._refresh_dataset_choices(model_task)
            name = self.cb_dataset.currentText()
        self.spin_dataset_size.setEnabled(True)
        self.btn_generate_dataset.setEnabled(True)
        requested_size = self.spin_dataset_size.value()
        self._X, self._y = DATASETS[name](n_samples=requested_size)
        self._X_model = self._X_train = self._X_test = None
        self._y_train = self._y_test = None
        self._on_model_changed()
        self._reset()
        self.status_label.setText(
            f"Generated {name}: {len(self._y):,} samples × {self._X.shape[1]} features"
        )

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            "",
            "Data files (*.csv *.tsv *.txt);;All files (*)",
        )
        if not path:
            return

        try:
            inspection = inspect_csv(path)
            default_index = inspection.columns.index(inspection.suggested_target)
            target_column, accepted = QInputDialog.getItem(
                self,
                "Select target column",
                "Which column should the model predict?",
                inspection.columns,
                default_index,
                False,
            )
            if not accepted:
                return

            model = self._create_model()
            X, y, info = load_csv(
                path,
                target_column=target_column,
                task=model.task,
                standardize=False,
                return_info=True,
            )

            self._X, self._y = X, y
            dataset_name = f"CSV: {Path(path).name}"
            self._X_model = self._X_train = self._X_test = None
            self._y_train = self._y_test = None
            self._csv_dataset_name = dataset_name
            self._csv_task = model.task
            self._csv_X, self._csv_y = X.copy(), y.copy()
            self._refresh_dataset_choices(model.task, preferred=dataset_name)
            self.spin_dataset_size.setEnabled(False)
            self.btn_generate_dataset.setEnabled(False)

            self._on_model_changed()
            self._reset()
            notes = []
            if info.rows_dropped:
                notes.append(f"dropped {info.rows_dropped} rows without target")
            if info.imputed_values:
                notes.append(f"imputed {info.imputed_values} missing values")
            if info.categorical_columns:
                notes.append(f"encoded {len(info.categorical_columns)} categorical columns")
            if info.class_mapping:
                mapping = ", ".join(
                    f"{label}→{int(value)}" for label, value in info.class_mapping.items()
                )
                notes.append(f"classes: {mapping}")
            suffix = f"; {'; '.join(notes)}" if notes else ""
            self.status_label.setText(
                f"Loaded {Path(path).name}: {X.shape[0]} rows × {X.shape[1]} features; "
                f"target: {info.target_name}{suffix}"
            )
        except CSVDataError as exc:
            self.status_label.setText(f"CSV error: {exc}")
            QMessageBox.warning(self, "Cannot load dataset", str(exc))
        except Exception as exc:
            self.status_label.setText(f"Unexpected CSV error: {exc}")
            QMessageBox.critical(
                self,
                "Unexpected CSV error",
                f"The file could not be prepared:\n{exc}",
            )

    def _on_model_changed(self):
        model_name = self.cb_model.currentText()
        is_ridge = model_name == "Ridge Regression"
        self.ridge_lambda_row.setVisible(is_ridge)
        self._current_model = self._create_model()
        dataset_changed = self._refresh_dataset_choices(self._current_model.task)
        if dataset_changed and self.cb_dataset.currentText() in DATASETS:
            self._load_dataset()
            return
        if self._X is not None and self._y is not None:
            self._current_model.fit_data(self._X, self._y)
        is_cls = self._current_model.task == "classification"
        self.stat_acc.set_metric("Test Accuracy" if is_cls else "Test R²", "%")
        self.stat_acc.val.setStyleSheet(
            f"font-size:20px;font-weight:bold;"
            f"color:{'#3fb950' if is_cls else '#58a6ff'};"
        )

    def _create_model(self):
        model_name = self.cb_model.currentText()
        model_class = MODELS[model_name]
        if model_name == "Ridge Regression":
            return model_class(regularization=self.spin_ridge_lambda.value())
        return model_class()

    def _start(self, _checked=False, forced_algorithms=None):
        if self._X is None:
            self._load_dataset()
        if not self._reset_workers():
            self.status_label.setText("Previous workers are still stopping. Try again shortly.")
            return
        self._histories.clear()
        self._stats.clear()
        self._surface_trajectories.clear()
        self._finished_workers = 0

        try:
            model_task = self._create_model().task
            X_train, X_test, self._y_train, self._y_test = train_test_split(
                self._X,
                self._y,
                test_fraction=self.spin_test_split.value() / 100,
                task=model_task,
                seed=42,
            )
            self._X_train, self._X_test, self._X_model = standardize_from_training(
                X_train, X_test, self._X
            )
        except ValueError as exc:
            self.status_label.setText(f"Cannot start experiment: {exc}")
            QMessageBox.warning(self, "Invalid train/test split", str(exc))
            return

        if forced_algorithms is None:
            algs = [self.cb_algorithm.currentText()]
            cmp = self.cb_compare.currentText()
            if cmp and cmp != COMPARE_PLACEHOLDER and cmp not in algs:
                algs.append(cmp)
        else:
            algs = list(dict.fromkeys(forced_algorithms))

        if "Adam" in algs:
            self.lbl_adam_verdict.setText("RUNNING…")
            self.txt_adam_reasons.setPlainText(
                "Zbieram historię loss, gradientów i kroków z bieżącego treningu Adama."
            )
            self.btn_run_adam_check.setEnabled(False)
        else:
            self._set_adam_diagnostic_placeholder(
                "Bieżący eksperyment nie używa Adama. Dodaj Adam jako główny optimizer "
                "lub wybierz go w polu porównania."
            )

        launched = 0
        for alg_name in algs:
            if alg_name == "ALS":
                self.status_label.setText(
                    "ALS is for Matrix Factorization — use ♫ Music Recommendation tab."
                )
                continue
            self._launch_worker(alg_name)
            launched += 1

        if launched > 0:
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self._set_experiment_controls_enabled(False)

    def _launch_worker(self, alg_name: str):
        model = self._create_model()
        model.fit_data(self._X_train, self._y_train)

        lr  = self.spin_lr.value()
        opt = OPTIMIZATION_ALGORITHMS[alg_name](learning_rate=lr)
        opt.history["params"].append(model.params.copy())

        batch_size = (
            len(self._y_train)
            if alg_name == "Gradient Descent"
            else self.spin_batch.value()
        )
        w = TrainingWorker(
            model=model, optimizer=opt,
            X=self._X_train, y=self._y_train,
            X_eval=self._X_test, y_eval=self._y_test,
            n_iterations=self.spin_iter.value(),
            batch_size=batch_size,
            emit_every=self.spin_emit.value(),
        )
        w.step_done.connect(lambda p, n=alg_name: self._on_step(p, n))
        w.finished.connect( lambda p, n=alg_name: self._on_finished(p, n))
        w.error.connect(lambda msg, n=alg_name: self._on_worker_error(msg, n))
        self._workers.append(w)
        w.start()

    @pyqtSlot(dict)
    def _on_step(self, payload: dict, name: str):
        self._histories.setdefault(name, []).append(payload["loss"])

        if name == self.cb_algorithm.currentText():
            self.stat_loss.set_value(payload["loss"])
            self.stat_iter.set_value(payload["iteration"])
            self.stat_time.set_value(round(payload["elapsed"], 2))
            if "accuracy" in payload:
                self.stat_acc.set_value(round(payload["accuracy"] * 100, 1))
            elif "r2" in payload:
                self.stat_acc.set_value(round(payload["r2"] * 100, 1))

        params = payload["params"]
        if len(params) >= 2:
            self._surface_trajectories.setdefault(name, []).append(
                (params[0], params[1], payload["loss"])
            )

        idx = self.tabs.currentIndex()
        if idx == 0:
            self._update_loss_tab()
        elif idx == 2:
            self._update_3d_tab()

    @pyqtSlot(dict)
    def _on_finished(self, payload: dict, name: str):
        self._stats[name] = {
            "final_loss": payload["loss"],
            "time":       payload["elapsed"],
            "iterations": payload["iteration"],
            "accuracy":   payload.get("accuracy"),
            "r2":         payload.get("r2"),
            "rmse":       payload.get("rmse"),
        }
        if name == self.cb_algorithm.currentText():
            if "accuracy" in payload:
                self.stat_acc.set_value(round(payload["accuracy"] * 100, 1))
            elif "r2" in payload:
                self.stat_acc.set_value(round(payload["r2"] * 100, 1))
        try:
            saved_path = self._save_confusion_matrix(name)
            saved_message = f"  |  PNG: {saved_path.name}" if saved_path else ""
        except Exception as exc:
            saved_message = f"  |  PNG save failed: {exc}"
        self.status_label.setText(
            f"✔ {name}  loss={payload['loss']:.5f}"
            f"  iters={payload['iteration']}  t={payload['elapsed']:.2f}s"
            f"{saved_message}"
        )
        self._finished_workers += 1
        self._finish_experiment_if_ready()

    def _on_worker_error(self, message: str, name: str):
        self._finished_workers += 1
        self.status_label.setText(f"{name} failed: {message}")
        if name == "Adam":
            self.lbl_adam_verdict.setText("FAILURE")
            self.lbl_adam_verdict.setStyleSheet(
                "font-size:25px;font-weight:bold;color:#f85149;padding:10px;"
                "background:#161b22;border:1px solid #f85149;border-radius:8px;"
            )
            self.txt_adam_reasons.setPlainText(
                f"Trening zakończył się wyjątkiem:\n{message}"
            )
            self.btn_run_adam_check.setEnabled(True)
        self._finish_experiment_if_ready()

    def _finish_experiment_if_ready(self):
        if self._finished_workers >= len(self._workers):
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self._set_experiment_controls_enabled(True)
            self.btn_run_adam_check.setEnabled(True)
            if self._stats:
                self._refresh_all_tabs()
            if any(worker.optimizer.name == "Adam" for worker in self._workers):
                self._update_adam_diagnostics()

    def _set_experiment_controls_enabled(self, enabled: bool):
        for widget in (
            self.cb_algorithm,
            self.cb_compare,
            self.cb_model,
            self.cb_dataset,
            self.spin_dataset_size,
            self.btn_generate_dataset,
            self.btn_csv,
            self.spin_ridge_lambda,
            self.spin_lr,
            self.spin_iter,
            self.spin_batch,
            self.spin_emit,
            self.spin_test_split,
        ):
            widget.setEnabled(enabled)
        if enabled:
            is_csv = self.cb_dataset.currentText().startswith("CSV: ")
            self.spin_dataset_size.setEnabled(not is_csv)
            self.btn_generate_dataset.setEnabled(not is_csv)
            self.spin_ridge_lambda.setEnabled(
                self.cb_model.currentText() == "Ridge Regression"
            )

    def _save_confusion_matrix(self, optimizer_name: str) -> Path | None:
        worker = next(
            (item for item in self._workers if item.optimizer.name == optimizer_name),
            None,
        )
        if worker is None or worker.model.task != "classification":
            return None

        predicted = worker.model.predict_class(self._X_test)
        matrix = binary_confusion_matrix(self._y_test, predicted)
        dataset_name = self.cb_dataset.currentText().removeprefix("CSV: ")

        def safe_name(value: str) -> str:
            cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            return cleaned or "dataset"

        output_dir = Path(__file__).resolve().parents[2] / "wyniki"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"confusion_matrix_{safe_name(dataset_name)}_"
            f"{safe_name(optimizer_name)}.png"
        )
        figure = plot_confusion_matrix(
            matrix,
            title=f"{dataset_name} — {optimizer_name} (test set)",
        )
        figure.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        figure.clear()
        return output_path

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
        if self._X_model is None or not self._workers:
            return
        self.fig_fit.clear()
        if self._workers[0].model.task == "regression":
            params_map = {w.optimizer.name: w.model.params.copy() for w in self._workers}
            plot_regression_fit(self._X_model, self._y, params_map, fig=self.fig_fit)
        else:
            plot_decision_boundary(
                self._X_model, self._y, self._workers[0].model,
                optimizer_name=self.cb_algorithm.currentText(),
                fig=self.fig_fit,
            )
        self.canvas_fit.draw_idle()

    def _update_3d_tab(self):
        if self._X is None or not self._workers:
            return
        try:
            w = self._workers[0]
            B, W, Z = build_loss_surface(w.model, w.X, w.y, resolution=35)
            plot_loss_surface_3d(
                B, W, Z, trajectories=self._surface_trajectories, fig=self.fig_3d
            )
            self.canvas_3d.draw_idle()
        except Exception as exc:
            self.status_label.setText(f"3D plot unavailable: {exc}")

    def _update_compare_tab(self):
        if not self._stats:
            return
        plot_comparison(self._stats, fig=self.fig_compare)
        self.canvas_compare.draw_idle()

    def _pause(self):
        for w in self._workers:
            w.pause()
        paused = any(w._paused for w in self._workers)
        self.btn_pause.setText("▶  Resume" if paused else "⏸  Pause")

    def _reset(self):
        if not self._reset_workers():
            self.status_label.setText("Cannot reset while a worker is still stopping.")
            return
        self._histories.clear()
        self._stats.clear()
        self._surface_trajectories.clear()
        for attr in ("loss", "fit", "3d", "compare"):
            fig    = getattr(self, f"fig_{attr}")
            canvas = getattr(self, f"canvas_{attr}")
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
        self._set_experiment_controls_enabled(True)
        self.btn_run_adam_check.setEnabled(True)
        self.lbl_adam_context.setText("No Adam experiment yet.")
        self._set_adam_diagnostic_placeholder(
            "Uruchom Adam na bieżącym modelu i datasecie."
        )
        self.status_label.setText("Reset. Configure and press ▶ Start.")

    def _reset_workers(self, timeout_ms: int = 3000) -> bool:
        workers = list(self._workers)
        for w in workers:
            w.stop()
        still_running = []
        for w in workers:
            if not w.wait(timeout_ms) and w.isRunning():
                still_running.append(w)
        self._workers = still_running
        self._finished_workers = 0
        return not still_running

    def _setup_tab_refresh(self):
        self.tabs.currentChanged.connect(lambda _: (
            self._update_loss_tab()    if self.tabs.currentIndex() == 0 else
            self._update_3d_tab()      if self.tabs.currentIndex() == 2 else
            None
        ))

    def closeEvent(self, event):
        if not self._reset_workers(timeout_ms=5000):
            self.status_label.setText("Waiting for optimization workers to stop before closing.")
            event.ignore()
            return
        self.rec_panel.closeEvent(event)
        if not event.isAccepted():
            return
        super().closeEvent(event)
