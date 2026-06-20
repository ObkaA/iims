"""
Music Recommendation Panel — Matrix Factorization via ALS.
UX: pick up to 5 favourite artists → fold-in → personalised recommendations.
All algorithms implemented from scratch (see recommendation/matrix_factorization.py).
"""
from __future__ import annotations
import os, sys
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
    QFileDialog, QListWidget, QListWidgetItem, QGridLayout,
    QFrame, QSizePolicy, QProgressBar, QSplitter, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recommendation.matrix_factorization import MatrixFactorization
from datasets import load_lastfm_csv, generate_synthetic_lastfm
from visualization.rec_viz import (
    plot_mf_loss, plot_latent_heatmap, plot_embeddings_2d,
    plot_rating_matrix, plot_recommendations, plot_latent_snapshot, COLORS,
)

MAX_PICKS = 5


# ── Training thread ────────────────────────────────────────────────────────────
class ALSWorker(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, mf, user_ids, item_ids, ratings):
        super().__init__()
        self.mf = mf
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings  = ratings
        self._stopped = False

    def stop(self): self._stopped = True

    def run(self):
        try:
            def cb(epoch, rmse):
                if self._stopped: raise InterruptedError
                self.progress.emit(epoch, rmse)
            self.mf.fit_from_triplets(
                self.user_ids, self.item_ids, self.ratings, progress_cb=cb
            )
            if not self._stopped:
                self.finished.emit()
        except InterruptedError:
            pass
        except Exception as e:
            self.error.emit(str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _canvas(fig: Figure) -> FigureCanvas:
    c = FigureCanvas(fig)
    c.setMinimumHeight(180)
    c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    fig.patch.set_facecolor(COLORS["bg"])
    return c


def _stat_frame(label: str) -> tuple[QFrame, QLabel]:
    frame = QFrame()
    frame.setStyleSheet(
        "QFrame{background:#161b22;border:1px solid #21262d;border-radius:8px;}"
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 8)
    val = QLabel("—")
    val.setStyleSheet("font-size:18px;font-weight:bold;color:#58a6ff;")
    val.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl = QLabel(label.upper())
    lbl.setStyleSheet("font-size:9px;color:#7d8590;letter-spacing:1px;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(val); lay.addWidget(lbl)
    return frame, val


# ══════════════════════════════════════════════════════════════════════════════
class MusicRecommendationPanel(QWidget):
    """
    Self-contained Music Recommendation panel.
    Flow:
      1. Load dataset (synthetic or CSV)
      2. Train ALS
      3. Pick ≤5 favourite artists from the catalogue
      4. Click "Recommend for Me" → fold-in + personalised Top-N
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mf: MatrixFactorization | None = None
        self._worker: ALSWorker | None = None
        self._user_ids: list = []
        self._item_ids: list = []
        self._ratings:  list = []
        self._you_vec:  np.ndarray | None = None
        self._anim_idx  = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)

        self._build_ui()
        self._load_synthetic()

    # ══════════════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._build_sidebar())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("QSplitter::handle{background:#21262d;height:2px;}")
        splitter.addWidget(self._build_stats_row())
        self.chart_tabs = QTabWidget()
        self._build_chart_tabs()
        splitter.addWidget(self.chart_tabs)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, stretch=1)

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(285)
        w.setStyleSheet("background:#010409;border-right:1px solid #21262d;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 16, 14, 12)
        lay.setSpacing(10)

        # Header
        logo = QLabel("♫  Music Recommender")
        logo.setStyleSheet("font-size:14px;font-weight:bold;color:#58a6ff;letter-spacing:1px;")
        sub  = QLabel("Matrix Factorization · ALS · from scratch")
        sub.setStyleSheet("font-size:9px;color:#7d8590;")
        lay.addWidget(logo); lay.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#21262d;"); lay.addWidget(sep)

        # Dataset
        grp_data = QGroupBox("Dataset")
        dl = QVBoxLayout(grp_data); dl.setSpacing(6)
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(["Synthetic Last.fm Demo", "Load CSV…"])
        self.cb_dataset.currentIndexChanged.connect(self._on_dataset_change)
        dl.addWidget(self.cb_dataset)
        self.lbl_data_info = QLabel("80 users · 25 artists · synthetic")
        self.lbl_data_info.setStyleSheet("font-size:10px;color:#7d8590;")
        self.lbl_data_info.setWordWrap(True)
        dl.addWidget(self.lbl_data_info)
        lay.addWidget(grp_data)

        # Hyperparameters
        grp_hp = QGroupBox("ALS Hyperparameters")
        hl = QGridLayout(grp_hp); hl.setSpacing(6)

        hl.addWidget(QLabel("Latent factors k"), 0, 0)
        self.spin_k = QSpinBox(); self.spin_k.setRange(2, 100); self.spin_k.setValue(15)
        hl.addWidget(self.spin_k, 0, 1)

        hl.addWidget(QLabel("Regularization λ"), 1, 0)
        self.spin_reg = QDoubleSpinBox()
        self.spin_reg.setRange(0.001, 10.0); self.spin_reg.setValue(0.1)
        self.spin_reg.setSingleStep(0.05); self.spin_reg.setDecimals(3)
        hl.addWidget(self.spin_reg, 1, 1)

        hl.addWidget(QLabel("ALS Epochs"), 2, 0)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(5, 500); self.spin_epochs.setValue(30)
        hl.addWidget(self.spin_epochs, 2, 1)

        hl.addWidget(QLabel("Embedding"), 3, 0)
        self.cb_embed = QComboBox(); self.cb_embed.addItems(["PCA", "t-SNE"])
        hl.addWidget(self.cb_embed, 3, 1)

        lay.addWidget(grp_hp)

        # Train controls
        grp_ctrl = QGroupBox("Model Training")
        cl = QVBoxLayout(grp_ctrl); cl.setSpacing(6)
        self.btn_train = QPushButton("▶  Train ALS Model")
        self.btn_train.setObjectName("btn_start")
        self.btn_train.clicked.connect(self._train)
        cl.addWidget(self.btn_train)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("btn_reset")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        cl.addWidget(self.btn_stop)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;"
            "border-radius:4px;height:8px;text-align:center;}"
            "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}"
        )
        cl.addWidget(self.progress_bar)
        lay.addWidget(grp_ctrl)

        # ── YOUR favourite artists picker ───────────────────────────────────
        grp_you = QGroupBox(f"Your Favourite Artists  (max {MAX_PICKS})")
        grp_you.setStyleSheet(
            "QGroupBox{border:1px solid #1f6feb;border-radius:8px;margin-top:12px;"
            "padding-top:8px;font-size:10px;font-weight:bold;color:#58a6ff;letter-spacing:1px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;color:#58a6ff;}"
        )
        yl = QVBoxLayout(grp_you); yl.setSpacing(6)

        hint = QLabel("Pick from catalogue, then click Recommend:")
        hint.setStyleSheet("font-size:10px;color:#7d8590;")
        yl.addWidget(hint)

        row1 = QHBoxLayout()
        self.cb_catalogue = QComboBox()
        row1.addWidget(self.cb_catalogue, stretch=1)
        btn_add = QPushButton("＋")
        btn_add.setFixedWidth(32)
        btn_add.setStyleSheet(
            "QPushButton{background:#1f6feb;color:#fff;border-radius:5px;"
            "font-size:16px;font-weight:bold;padding:0;min-height:28px;}"
            "QPushButton:hover{background:#388bfd;}"
        )
        btn_add.clicked.connect(self._add_favourite)
        row1.addWidget(btn_add)
        yl.addLayout(row1)

        self.lst_favourites = QListWidget()
        self.lst_favourites.setMaximumHeight(110)
        self.lst_favourites.setStyleSheet(
            "QListWidget{background:#0d1117;border:1px solid #30363d;"
            "border-radius:5px;color:#e6edf3;font-size:11px;}"
            "QListWidget::item:selected{background:#1f6feb;color:#fff;}"
            "QListWidget::item:hover{background:#161b22;}"
        )
        yl.addWidget(self.lst_favourites)

        btn_row = QHBoxLayout()
        btn_rem = QPushButton("✕ Remove")
        btn_rem.setStyleSheet(
            "QPushButton{background:#161b22;border:1px solid #f85149;"
            "color:#f85149;border-radius:5px;font-size:10px;padding:3px 8px;min-height:24px;}"
            "QPushButton:hover{background:#2d0e0e;}"
        )
        btn_rem.clicked.connect(self._remove_favourite)
        btn_clr = QPushButton("↺ Clear")
        btn_clr.setStyleSheet(
            "QPushButton{background:#161b22;border:1px solid #30363d;"
            "color:#7d8590;border-radius:5px;font-size:10px;padding:3px 8px;min-height:24px;}"
            "QPushButton:hover{border-color:#7d8590;color:#e6edf3;}"
        )
        btn_clr.clicked.connect(self._clear_favourites)
        btn_row.addWidget(btn_rem); btn_row.addWidget(btn_clr)
        yl.addLayout(btn_row)

        self.lbl_picks = QLabel("0 / 5 selected")
        self.lbl_picks.setStyleSheet("font-size:10px;color:#7d8590;")
        yl.addWidget(self.lbl_picks)
        lay.addWidget(grp_you)

        # Recommend for Me
        grp_rec = QGroupBox("Personalised Recommendations")
        rl = QVBoxLayout(grp_rec); rl.setSpacing(6)
        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Top-N:"))
        self.spin_topn = QSpinBox(); self.spin_topn.setRange(3, 20); self.spin_topn.setValue(8)
        row_n.addWidget(self.spin_topn)
        rl.addLayout(row_n)
        self.btn_recommend = QPushButton("🎵  Recommend For Me")
        self.btn_recommend.setObjectName("btn_start")
        self.btn_recommend.clicked.connect(self._recommend_for_me)
        self.btn_recommend.setEnabled(False)
        rl.addWidget(self.btn_recommend)
        lay.addWidget(grp_rec)

        # Animation
        grp_anim = QGroupBox("Latent Factor Animation")
        al = QVBoxLayout(grp_anim)
        self.sld_epoch = QSlider(Qt.Orientation.Horizontal)
        self.sld_epoch.setRange(0, 0)
        self.sld_epoch.valueChanged.connect(self._on_epoch_slide)
        al.addWidget(self.sld_epoch)
        row_a = QHBoxLayout()
        self.btn_anim = QPushButton("▶ Play")
        self.btn_anim.setEnabled(False)
        self.btn_anim.clicked.connect(self._toggle_anim)
        self.btn_anim.setStyleSheet(
            "QPushButton{background:#161b22;border:1px solid #30363d;"
            "border-radius:5px;color:#e6edf3;padding:4px 10px;font-size:11px;}"
        )
        row_a.addWidget(self.btn_anim)
        al.addLayout(row_a)
        lay.addWidget(grp_anim)

        lay.addStretch()
        self.status_lbl = QLabel("Load data → Train → Pick artists → Recommend.")
        self.status_lbl.setStyleSheet("font-size:10px;color:#7d8590;")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)
        return w

    def _build_stats_row(self) -> QWidget:
        w = QWidget(); w.setMaximumHeight(86)
        lay = QHBoxLayout(w); lay.setContentsMargins(12, 6, 12, 6)
        f1, self.val_rmse  = _stat_frame("RMSE")
        f2, self.val_users = _stat_frame("Users")
        f3, self.val_items = _stat_frame("Artists")
        f4, self.val_spars = _stat_frame("Sparsity %")
        for f in (f1, f2, f3, f4): lay.addWidget(f)
        return w

    def _build_chart_tabs(self):
        for attr, title in [
            ("loss",      "📉 ALS Loss"),
            ("matrix",    "🗃 Rating Matrix"),
            ("embed",     "🌐 2D Embedding"),
            ("recs",      "🎵 Your Recs"),
            ("similar",   "🔗 Similar Artists"),
            ("sim_users", "👥 Similar Users"),
            ("anim",      "🎞 Latent Anim"),
        ]:
            fig    = Figure(tight_layout=True)
            canvas = _canvas(fig)
            setattr(self, f"fig_{attr}",    fig)
            setattr(self, f"canvas_{attr}", canvas)
            self.chart_tabs.addTab(canvas, title)
        self._blank_all()

    def _blank_all(self):
        for attr in ("loss","matrix","embed","recs","similar","sim_users","anim"):
            fig    = getattr(self, f"fig_{attr}")
            canvas = getattr(self, f"canvas_{attr}")
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(COLORS["surface"])
            ax.text(0.5, 0.5, "Train the model to see results",
                    transform=ax.transAxes, ha="center", va="center",
                    color=COLORS["text"], fontsize=11, alpha=0.4)
            fig.patch.set_facecolor(COLORS["bg"])
            canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading
    # ══════════════════════════════════════════════════════════════════════════
    def _load_synthetic(self):
        self._user_ids, self._item_ids, self._ratings = generate_synthetic_lastfm()
        n_u = len(set(self._user_ids))
        n_i = len(set(self._item_ids))
        n   = len(self._ratings)
        sp  = 100.0 * (1 - n / (n_u * n_i))
        self._refresh_catalogue()
        self._update_stats(n_u, n_i, n, sp)

    def _on_dataset_change(self, idx: int):
        if idx == 0:
            self._load_synthetic()
            self.lbl_data_info.setText("80 users · 25 artists · synthetic")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Last.fm CSV", "", "CSV (*.csv)"
            )
            if path:
                try:
                    self._user_ids, self._item_ids, self._ratings = load_lastfm_csv(path)
                    n_u = len(set(self._user_ids))
                    n_i = len(set(self._item_ids))
                    n   = len(self._ratings)
                    sp  = 100.0 * (1 - n / (n_u * n_i))
                    self._refresh_catalogue()
                    self._update_stats(n_u, n_i, n, sp)
                    self.lbl_data_info.setText(
                        f"{n_u} users · {n_i} items · {n} interactions"
                    )
                    self.status_lbl.setText(f"Loaded: {Path(path).name}")
                except Exception as e:
                    self.status_lbl.setText(f"Error: {e}")
                    self.cb_dataset.setCurrentIndex(0)
                    self._load_synthetic()

    def _refresh_catalogue(self):
        items = sorted(set(self._item_ids))
        self.cb_catalogue.clear()
        self.cb_catalogue.addItems(items)
        # Drop picked items no longer in catalogue
        kept = [self._get_favourites()[i] for i in range(self.lst_favourites.count())
                if self._get_favourites()[i] in items]
        self.lst_favourites.clear()
        for fav in kept: self._add_item_to_list(fav)
        self._update_picks_label()

    def _update_stats(self, n_u, n_i, n, sp):
        self.val_users.setText(str(n_u))
        self.val_items.setText(str(n_i))
        self.val_spars.setText(f"{sp:.1f}")
        # placeholder
        self.fig_matrix.clear()
        ax = self.fig_matrix.add_subplot(111)
        ax.set_facecolor(COLORS["surface"])
        ax.text(0.5, 0.5, f"Dataset: {n_u} users × {n_i} artists\nTrain to see matrix",
                transform=ax.transAxes, ha="center", va="center",
                color=COLORS["text"], fontsize=11, alpha=0.6)
        self.fig_matrix.patch.set_facecolor(COLORS["bg"])
        self.canvas_matrix.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Favourites picker
    # ══════════════════════════════════════════════════════════════════════════
    def _get_favourites(self) -> list[str]:
        return [self.lst_favourites.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.lst_favourites.count())]

    def _add_item_to_list(self, name: str):
        item = QListWidgetItem(f"♪  {name}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setForeground(QColor("#a8ff78"))
        self.lst_favourites.addItem(item)

    def _add_favourite(self):
        artist = self.cb_catalogue.currentText()
        if not artist: return
        if artist in self._get_favourites():
            self.status_lbl.setText(f"'{artist}' already added.")
            return
        if self.lst_favourites.count() >= MAX_PICKS:
            self.status_lbl.setText(f"Max {MAX_PICKS} — remove one first.")
            return
        self._add_item_to_list(artist)
        self._update_picks_label()
        self.status_lbl.setText(f"Added: {artist}")

    def _remove_favourite(self):
        row = self.lst_favourites.currentRow()
        if row >= 0:
            item = self.lst_favourites.takeItem(row)
            self.status_lbl.setText(f"Removed: {item.data(Qt.ItemDataRole.UserRole)}")
        self._update_picks_label()

    def _clear_favourites(self):
        self.lst_favourites.clear()
        self._update_picks_label()
        self.status_lbl.setText("Cleared selection.")

    def _update_picks_label(self):
        n = self.lst_favourites.count()
        self.lbl_picks.setText(f"{n} / {MAX_PICKS} selected")
        self.lbl_picks.setStyleSheet(
            f"font-size:10px;color:{'#a8ff78' if n > 0 else '#7d8590'};"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ALS Training
    # ══════════════════════════════════════════════════════════════════════════
    def _train(self):
        self._stop()
        self._you_vec = None
        self._mf = MatrixFactorization(
            n_factors  = self.spin_k.value(),
            reg_lambda = self.spin_reg.value(),
            n_iter     = self.spin_epochs.value(),
        )
        self._worker = ALSWorker(
            self._mf, self._user_ids, self._item_ids, self._ratings
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_training_done)
        self._worker.error.connect(lambda e: self.status_lbl.setText(f"Error: {e}"))
        self._worker.start()
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_recommend.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Training ALS model…")

    @pyqtSlot(int, float)
    def _on_progress(self, epoch: int, rmse: float):
        total = self.spin_epochs.value()
        self.progress_bar.setValue(int(100 * (epoch + 1) / total))
        self.val_rmse.setText(f"{rmse:.4f}")
        self.status_lbl.setText(f"Epoch {epoch+1}/{total}  RMSE={rmse:.4f}")
        if self._mf and self._mf.loss_history:
            plot_mf_loss(self._mf.loss_history, fig=self.fig_loss)
            self.canvas_loss.draw_idle()

    @pyqtSlot()
    def _on_training_done(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_recommend.setEnabled(True)
        self.btn_anim.setEnabled(True)
        self.progress_bar.setValue(100)
        mf = self._mf
        self.status_lbl.setText(
            f"✔ Done!  RMSE={mf.loss_history[-1]:.4f}"
            f" — pick artists and press Recommend."
        )
        self.sld_epoch.setRange(0, len(mf.U_snapshots) - 1)
        self.sld_epoch.setValue(0)
        self._refresh_all_charts()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self._worker.wait(1000)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ══════════════════════════════════════════════════════════════════════════
    # Personalised recommendations via fold-in ALS
    # ══════════════════════════════════════════════════════════════════════════
    def _recommend_for_me(self):
        mf = self._mf
        if mf is None:
            self.status_lbl.setText("Train the model first.")
            return
        faves = self._get_favourites()
        if not faves:
            self.status_lbl.setText("Select at least 1 favourite artist.")
            return

        # Fold-in: u_new = (V_Ω^T V_Ω + λI)^{-1} V_Ω^T r
        # (one ALS update for a new user, V held fixed)
        self._you_vec = mf.fold_in_user(faves, [9.0] * len(faves))

        if np.all(self._you_vec == 0):
            self.status_lbl.setText(
                "None of your picks overlap with the training catalogue."
            )
            return

        top_n = self.spin_topn.value()
        recs  = mf.recommend_for_vector(
            self._you_vec, top_n=top_n, exclude_items=faves
        )
        sim_users = mf.similar_users(self._you_vec, top_n=top_n)
        sim_items = mf.similar_items(recs[0][0], top_n=top_n) if recs else []

        # Recommendations chart
        plot_recommendations(
            recs,
            title=f"🎵 Top-{top_n} Recommendations for You",
            fig=self.fig_recs,
        )
        self.canvas_recs.draw_idle()

        # Similar artists
        if sim_items:
            plot_recommendations(
                sim_items,
                title=f"Artists similar to '{recs[0][0][:20]}'",
                fig=self.fig_similar,
            )
            self.canvas_similar.draw_idle()

        # Similar users
        if sim_users:
            plot_recommendations(
                sim_users, title="Users similar to you",
                fig=self.fig_sim_users,
            )
            self.canvas_sim_users.draw_idle()

        # Embedding with YOU highlighted
        self._refresh_embedding(highlight_items=[r[0] for r in recs])

        self.chart_tabs.setCurrentIndex(3)
        picks_str = ", ".join(faves[:3]) + ("…" if len(faves) > 3 else "")
        self.status_lbl.setText(
            f"✔ {len(recs)} recommendations based on: {picks_str}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Chart helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_all_charts(self):
        mf = self._mf
        if mf is None: return
        plot_mf_loss(mf.loss_history, fig=self.fig_loss)
        self.canvas_loss.draw_idle()
        plot_rating_matrix(mf.R, mf.mask, mf.users, mf.items, fig=self.fig_matrix)
        self.canvas_matrix.draw_idle()
        self._refresh_embedding()
        self._show_anim_frame(0)

    def _refresh_embedding(self, highlight_items: list[str] | None = None):
        mf = self._mf
        if mf is None: return
        try:
            method = self.cb_embed.currentText().lower().replace("-", "")
            u2d, i2d, e2d = mf.get_2d_embeddings(
                method=method, extra_vec=self._you_vec
            )
            plot_embeddings_2d(
                u2d, i2d, mf.users, mf.items,
                highlight_items=highlight_items,
                you_point=e2d,
                fig=self.fig_embed,
            )
            self.canvas_embed.draw_idle()
        except Exception:
            pass

    # Animation
    def _on_epoch_slide(self, val: int):
        self._anim_idx = val
        self._show_anim_frame(val)

    def _show_anim_frame(self, idx: int):
        mf = self._mf
        if mf is None or not mf.U_snapshots: return
        plot_latent_snapshot(mf.U_snapshots, idx, mf.users, fig=self.fig_anim)
        self.canvas_anim.draw_idle()

    def _toggle_anim(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop(); self.btn_anim.setText("▶ Play")
        else:
            self._anim_timer.start(320); self.btn_anim.setText("⏸ Pause")

    def _anim_tick(self):
        mf = self._mf
        if mf is None or not mf.U_snapshots:
            self._anim_timer.stop(); return
        self._anim_idx = (self._anim_idx + 1) % len(mf.U_snapshots)
        self.sld_epoch.setValue(self._anim_idx)
        self._show_anim_frame(self._anim_idx)

    def closeEvent(self, event):
        self._anim_timer.stop(); self._stop(); super().closeEvent(event)
