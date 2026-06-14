"""
Music Recommendation Panel — Matrix Factorization via ALS.
This is a self-contained QWidget that plugs into the main tab system.
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
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recommendation import MatrixFactorization
from datasets import load_lastfm_csv, generate_synthetic_lastfm
from visualization.rec_viz import (
    plot_mf_loss, plot_latent_heatmap, plot_embeddings_2d,
    plot_rating_matrix, plot_recommendations, plot_latent_snapshot,
    COLORS,
)

# ── Training thread ────────────────────────────────────────────────────────────
class ALSWorker(QThread):
    progress = pyqtSignal(int, float)   # epoch, rmse
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, mf: MatrixFactorization, user_ids, item_ids, ratings):
        super().__init__()
        self.mf       = mf
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings  = ratings
        self._stopped = False

    def stop(self): self._stopped = True

    def run(self):
        try:
            def cb(epoch, rmse):
                if self._stopped:
                    raise InterruptedError
                self.progress.emit(epoch, rmse)
            self.mf.fit_from_triplets(self.user_ids, self.item_ids, self.ratings, progress_cb=cb)
            if not self._stopped:
                self.finished.emit()
        except InterruptedError:
            pass
        except Exception as e:
            self.error.emit(str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _mpl_canvas(fig: Figure) -> FigureCanvas:
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
    lay.addWidget(val)
    lay.addWidget(lbl)
    return frame, val


# ── Main Widget ────────────────────────────────────────────────────────────────
class MusicRecommendationPanel(QWidget):
    """Drop-in panel for the Music Recommendation tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mf: MatrixFactorization | None = None
        self._worker: ALSWorker | None = None
        self._user_ids = []
        self._item_ids = []
        self._ratings  = []
        self._anim_idx = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)

        self._build_ui()
        self._load_synthetic()

    # ── Build UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Left sidebar
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # Right: content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("QSplitter::handle{background:#21262d;height:2px;}")

        # Stats row
        stats_w = self._build_stats_row()
        splitter.addWidget(stats_w)

        # Chart tabs
        self.chart_tabs = QTabWidget()
        self._build_chart_tabs()
        splitter.addWidget(self.chart_tabs)
        splitter.setStretchFactor(1, 4)

        root.addWidget(splitter, stretch=1)

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(270)
        w.setStyleSheet("background:#010409;border-right:1px solid #21262d;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 18, 14, 14)
        lay.setSpacing(14)

        # Header
        logo = QLabel("♫ Music Recommender")
        logo.setStyleSheet("font-size:15px;font-weight:bold;color:#58a6ff;letter-spacing:1px;")
        sub  = QLabel("Matrix Factorization · ALS")
        sub.setStyleSheet("font-size:10px;color:#7d8590;")
        lay.addWidget(logo)
        lay.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#21262d;"); lay.addWidget(sep)

        # Dataset
        grp_data = QGroupBox("Dataset")
        dl = QVBoxLayout(grp_data)
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
        grp_hp = QGroupBox("Model Hyperparameters")
        hl = QGridLayout(grp_hp)

        hl.addWidget(QLabel("Latent factors k"), 0, 0)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 100); self.spin_k.setValue(15)
        hl.addWidget(self.spin_k, 0, 1)

        hl.addWidget(QLabel("Regularization λ"), 1, 0)
        self.spin_reg = QDoubleSpinBox()
        self.spin_reg.setRange(0.001, 10.0); self.spin_reg.setValue(0.1)
        self.spin_reg.setSingleStep(0.05); self.spin_reg.setDecimals(3)
        hl.addWidget(self.spin_reg, 1, 1)

        hl.addWidget(QLabel("ALS Epochs"), 2, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(5, 500); self.spin_epochs.setValue(30)
        hl.addWidget(self.spin_epochs, 2, 1)

        hl.addWidget(QLabel("Embedding"), 3, 0)
        self.cb_embed = QComboBox()
        self.cb_embed.addItems(["PCA", "t-SNE"])
        hl.addWidget(self.cb_embed, 3, 1)

        lay.addWidget(grp_hp)

        # Controls
        grp_ctrl = QGroupBox("Controls")
        cl = QVBoxLayout(grp_ctrl)
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
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:4px;height:8px;}"
            "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}"
        )
        cl.addWidget(self.progress_bar)
        lay.addWidget(grp_ctrl)

        # Artist selection (auto-selected)
        grp_artist = QGroupBox("Selected Artists → Recommend")
        al = QVBoxLayout(grp_artist)
        self.lbl_artists = QLabel("—")
        self.lbl_artists.setStyleSheet("font-size:9px;color:#7d8590;")
        self.lbl_artists.setWordWrap(True)
        al.addWidget(self.lbl_artists)

        self.btn_resample = QPushButton("🔄  Resample Artists")
        self.btn_resample.setObjectName("btn_reset")
        self.btn_resample.clicked.connect(self._resample_artists)
        self.btn_resample.setEnabled(False)
        al.addWidget(self.btn_resample)

        self.btn_recommend = QPushButton("🎵  Generate Recommendations")
        self.btn_recommend.setObjectName("btn_start")
        self.btn_recommend.clicked.connect(self._recommend)
        self.btn_recommend.setEnabled(False)
        al.addWidget(self.btn_recommend)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Top-N:"))
        self.spin_topn = QSpinBox(); self.spin_topn.setRange(3, 20); self.spin_topn.setValue(8)
        hl2.addWidget(self.spin_topn)
        al.addLayout(hl2)
        lay.addWidget(grp_artist)
        
        # Store selected artists
        self._selected_artists = []

        # Animation controls
        grp_anim = QGroupBox("Latent Factor Animation")
        al = QVBoxLayout(grp_anim)
        self.sld_epoch = QSlider(Qt.Orientation.Horizontal)
        self.sld_epoch.setRange(0, 0)
        self.sld_epoch.valueChanged.connect(self._on_epoch_slide)
        al.addWidget(self.sld_epoch)
        row = QHBoxLayout()
        self.btn_anim = QPushButton("▶ Play")
        self.btn_anim.clicked.connect(self._toggle_anim)
        self.btn_anim.setEnabled(False)
        self.btn_anim.setStyleSheet(
            "QPushButton{background:#161b22;border:1px solid #30363d;border-radius:5px;"
            "color:#e6edf3;padding:4px 10px;font-size:11px;}"
        )
        row.addWidget(self.btn_anim)
        al.addLayout(row)
        lay.addWidget(grp_anim)

        lay.addStretch()

        self.status_lbl = QLabel("Load data and press Train.")
        self.status_lbl.setStyleSheet("font-size:10px;color:#7d8590;")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)
        return w

    def _build_stats_row(self) -> QWidget:
        w = QWidget(); w.setMaximumHeight(90)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 6, 12, 6)
        f1, self.val_rmse  = _stat_frame("RMSE")
        f2, self.val_users = _stat_frame("Users")
        f3, self.val_items = _stat_frame("Artists")
        f4, self.val_spars = _stat_frame("Sparsity %")
        for f in (f1, f2, f3, f4):
            lay.addWidget(f)
        return w

    def _build_chart_tabs(self):
        # Tab 1: Loss convergence
        self.fig_loss = Figure(tight_layout=True)
        self.canvas_loss = _mpl_canvas(self.fig_loss)
        self.chart_tabs.addTab(self.canvas_loss, "📉 ALS Loss")

        # Tab 2: Rating matrix
        self.fig_matrix = Figure(tight_layout=True)
        self.canvas_matrix = _mpl_canvas(self.fig_matrix)
        self.chart_tabs.addTab(self.canvas_matrix, "🗃 Rating Matrix")

        # Tab 3: 2D embeddings
        self.fig_embed = Figure(tight_layout=True)
        self.canvas_embed = _mpl_canvas(self.fig_embed)
        self.chart_tabs.addTab(self.canvas_embed, "🌐 2D Embedding")

        # Tab 4: Recommendations
        self.fig_recs = Figure(tight_layout=True)
        self.canvas_recs = _mpl_canvas(self.fig_recs)
        self.chart_tabs.addTab(self.canvas_recs, "🎵 Recommendations")

        # Tab 5: Similar artists
        self.fig_similar = Figure(tight_layout=True)
        self.canvas_similar = _mpl_canvas(self.fig_similar)
        self.chart_tabs.addTab(self.canvas_similar, "🔗 Similar Artists")

        # Tab 6: Similar users
        self.fig_sim_users = Figure(tight_layout=True)
        self.canvas_sim_users = _mpl_canvas(self.fig_sim_users)
        self.chart_tabs.addTab(self.canvas_sim_users, "👥 Similar Users")

        # Tab 7: Latent heatmap animation
        self.fig_anim = Figure(tight_layout=True)
        self.canvas_anim = _mpl_canvas(self.fig_anim)
        self.chart_tabs.addTab(self.canvas_anim, "🎞 Latent Animation")

        self._blank_all()

    def _blank_all(self):
        for fig, canvas in [
            (self.fig_loss, self.canvas_loss),
            (self.fig_matrix, self.canvas_matrix),
            (self.fig_embed, self.canvas_embed),
            (self.fig_recs, self.canvas_recs),
            (self.fig_similar, self.canvas_similar),
            (self.fig_sim_users, self.canvas_sim_users),
            (self.fig_anim, self.canvas_anim),
        ]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(COLORS["surface"])
            ax.text(0.5, 0.5, "Train the model to see results",
                    transform=ax.transAxes, ha="center", va="center",
                    color=COLORS["text"], fontsize=11, alpha=0.4)
            fig.patch.set_facecolor(COLORS["bg"])
            canvas.draw_idle()

    # ── Data loading ───────────────────────────────────────────────────────────
    def _load_synthetic(self):
        self._user_ids, self._item_ids, self._ratings = generate_synthetic_lastfm()
        n_users = len(set(self._user_ids))
        n_items = len(set(self._item_ids))
        n_pairs = len(self._ratings)
        sparsity = 100 * (1 - n_pairs / (n_users * n_items))
        self._update_data_stats(n_users, n_items, n_pairs, sparsity)

    def _on_dataset_change(self, idx: int):
        if idx == 0:
            self._load_synthetic()
            self.lbl_data_info.setText("80 users · 25 artists · synthetic")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open Last.fm CSV", "", "CSV (*.csv)")
            if path:
                try:
                    self._user_ids, self._item_ids, self._ratings = load_lastfm_csv(path)
                    n_u = len(set(self._user_ids))
                    n_i = len(set(self._item_ids))
                    n   = len(self._ratings)
                    sp  = 100 * (1 - n / (n_u * n_i))
                    self._update_data_stats(n_u, n_i, n, sp)
                    self.lbl_data_info.setText(
                        f"{n_u} users · {n_i} items · {n} interactions"
                    )
                    self.status_lbl.setText(f"Loaded: {Path(path).name}")
                except Exception as e:
                    self.status_lbl.setText(f"Error: {e}")
                    self.cb_dataset.setCurrentIndex(0)
                    self._load_synthetic()

    def _update_data_stats(self, n_users, n_items, n_pairs, sparsity):
        self.val_users.setText(str(n_users))
        self.val_items.setText(str(n_items))
        self.val_spars.setText(f"{sparsity:.1f}")
        # Populate rating matrix viz
        self._show_rating_matrix_placeholder(n_users, n_items)

    def _show_rating_matrix_placeholder(self, n_u, n_i):
        # Just show a message until model is trained
        self.fig_matrix.clear()
        ax = self.fig_matrix.add_subplot(111)
        ax.set_facecolor(COLORS["surface"])
        ax.text(0.5, 0.5, f"Dataset: {n_u} users × {n_i} artists\nTrain model to see full matrix",
                transform=ax.transAxes, ha="center", va="center",
                color=COLORS["text"], fontsize=11, alpha=0.6)
        self.fig_matrix.patch.set_facecolor(COLORS["bg"])
        self.canvas_matrix.draw_idle()

    # ── Training ───────────────────────────────────────────────────────────────
    def _train(self):
        self._stop()
        self._mf = MatrixFactorization(
            n_factors  = self.spin_k.value(),
            reg_lambda = self.spin_reg.value(),
            n_iter     = self.spin_epochs.value(),
        )
        self._worker = ALSWorker(self._mf, self._user_ids, self._item_ids, self._ratings)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_training_done)
        self._worker.error.connect(lambda e: self.status_lbl.setText(f"Error: {e}"))
        self._worker.start()
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Training ALS model…")

    @pyqtSlot(int, float)
    def _on_progress(self, epoch: int, rmse: float):
        total = self.spin_epochs.value()
        pct   = int(100 * (epoch + 1) / total)
        self.progress_bar.setValue(pct)
        self.val_rmse.setText(f"{rmse:.4f}")
        self.status_lbl.setText(f"Epoch {epoch + 1}/{total}  RMSE={rmse:.4f}")
        # Live loss plot
        if self._mf and self._mf.loss_history:
            plot_mf_loss(self._mf.loss_history, fig=self.fig_loss)
            self.canvas_loss.draw_idle()

    @pyqtSlot()
    def _on_training_done(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_recommend.setEnabled(True)
        self.btn_resample.setEnabled(True)
        self.btn_anim.setEnabled(True)
        self.progress_bar.setValue(100)

        mf = self._mf
        self.status_lbl.setText(
            f"✔ Training complete!  Final RMSE={mf.loss_history[-1]:.4f}"
        )

        # Auto-select random artists
        self._resample_artists()

        # Slider for animation
        self.sld_epoch.setRange(0, len(mf.U_snapshots) - 1)
        self.sld_epoch.setValue(0)

        # Update all static charts
        self._refresh_all_charts()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(1000)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ── Chart refresh ──────────────────────────────────────────────────────────
    def _refresh_all_charts(self):
        mf = self._mf
        if mf is None:
            return

        # Loss
        plot_mf_loss(mf.loss_history, fig=self.fig_loss)
        self.canvas_loss.draw_idle()

        # Rating matrix
        plot_rating_matrix(
            mf.R, mf.mask, mf.users, mf.items, fig=self.fig_matrix
        )
        self.canvas_matrix.draw_idle()

        # 2D embedding (without user highlight since we're using artists)
        method = self.cb_embed.currentText().lower().replace("-", "")
        try:
            u2d, i2d = mf.get_2d_embeddings(method=method)
            plot_embeddings_2d(
                u2d, i2d, mf.users, mf.items,
                highlight_items=self._selected_artists,
                fig=self.fig_embed,
            )
            self.canvas_embed.draw_idle()
        except Exception:
            pass

        # Latent animation frame 0
        self._anim_idx = 0
        self._show_anim_frame(0)

    def _resample_artists(self):
        """Automatically select up to 5 random artists from the available list."""
        mf = self._mf
        if mf is None or not mf.items:
            return
        
        # Select max 5 random artists
        max_artists = min(5, len(mf.items))
        self._selected_artists = list(np.random.choice(mf.items, size=max_artists, replace=False))
        
        # Update label
        artist_text = " • ".join([a[:30] for a in self._selected_artists])  # truncate long names
        self.lbl_artists.setText(artist_text)
        self.status_lbl.setText(f"Selected {len(self._selected_artists)} artists for recommendations")
        
        # Refresh embedding to highlight selected artists
        try:
            method = self.cb_embed.currentText().lower().replace("-", "")
            u2d, i2d = mf.get_2d_embeddings(method=method)
            plot_embeddings_2d(
                u2d, i2d, mf.users, mf.items,
                highlight_items=self._selected_artists,
                fig=self.fig_embed,
            )
            self.canvas_embed.draw_idle()
        except Exception:
            pass

    def _recommend(self):
        """Generate recommendations based on selected artists."""
        mf = self._mf
        if mf is None or not self._selected_artists:
            self.status_lbl.setText("No artists selected. Please train first.")
            return
        
        top_n = self.spin_topn.value()
        
        # Collect similar artists for each selected artist
        all_recs = {}
        for artist in self._selected_artists:
            similar = mf.similar_items(artist, top_n=top_n)
            for item, score in similar:
                if item not in all_recs:
                    all_recs[item] = 0
                all_recs[item] += score
        
        if not all_recs:
            self.status_lbl.setText("No recommendations found.")
            return
        
        # Sort by aggregated score and take top N
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Plot recommendations
        artists_str = ", ".join([a[:20] for a in self._selected_artists])
        plot_recommendations(
            sorted_recs, 
            title=f"Artists similar to: {artists_str}", 
            fig=self.fig_recs
        )
        self.canvas_recs.draw_idle()

        # Similar artists to top recommendation
        if sorted_recs:
            top_item = sorted_recs[0][0]
            similar = mf.similar_items(top_item, top_n=top_n)
            plot_recommendations(
                similar,
                title=f"Artists similar to '{top_item[:25]}'",
                fig=self.fig_similar,
            )
            self.canvas_similar.draw_idle()

        # Update embedding with highlighted recs
        try:
            method = self.cb_embed.currentText().lower().replace("-", "")
            u2d, i2d = mf.get_2d_embeddings(method=method)
            rec_items = [r[0] for r in sorted_recs]
            plot_embeddings_2d(
                u2d, i2d, mf.users, mf.items,
                highlight_items=rec_items,
                fig=self.fig_embed,
            )
            self.canvas_embed.draw_idle()
        except Exception:
            pass

        self.chart_tabs.setCurrentIndex(3)  # jump to Recommendations tab
        self.status_lbl.setText(f"✔ Top-{top_n} recommendations generated for selected artists")

    # ── Animation ──────────────────────────────────────────────────────────────
    def _on_epoch_slide(self, val: int):
        self._anim_idx = val
        self._show_anim_frame(val)

    def _show_anim_frame(self, idx: int):
        mf = self._mf
        if mf is None or not mf.U_snapshots:
            return
        plot_latent_snapshot(mf.U_snapshots, idx, mf.users, fig=self.fig_anim)
        self.canvas_anim.draw_idle()

    def _toggle_anim(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop()
            self.btn_anim.setText("▶ Play")
        else:
            self._anim_timer.start(300)  # ms per frame
            self.btn_anim.setText("⏸ Pause")

    def _anim_tick(self):
        mf = self._mf
        if mf is None or not mf.U_snapshots:
            self._anim_timer.stop()
            return
        self._anim_idx = (self._anim_idx + 1) % len(mf.U_snapshots)
        self.sld_epoch.setValue(self._anim_idx)
        self._show_anim_frame(self._anim_idx)

    def closeEvent(self, event):
        self._anim_timer.stop()
        self._stop()
        super().closeEvent(event)
