"""
Matrix Factorization via ALS for music recommendation.

Model: R ≈ U · V^T
  U : (n_users,  k)  — user latent factor matrix
  V : (n_items,  k)  — item latent factor matrix

ALS update (closed form):
  u_i = (V_S^T V_S + λI)^{-1} V_S^T r_i
  v_j = (U_S^T U_S + λI)^{-1} U_S^T r_j
"""
from __future__ import annotations
import numpy as np
import time


class MatrixFactorization:
    def __init__(
        self,
        n_factors:  int   = 20,
        reg_lambda: float = 0.1,
        n_iter:     int   = 50,
        seed:       int   = 42,
    ):
        self.n_factors  = n_factors
        self.reg_lambda = reg_lambda
        self.n_iter     = n_iter
        self.seed       = seed

        # Set after fit()
        self.U:           np.ndarray | None = None   # (n_users, k)
        self.V:           np.ndarray | None = None   # (n_items, k)
        self.user_index:  dict = {}
        self.item_index:  dict = {}
        self.users:       list = []
        self.items:       list = []
        self.R:           np.ndarray | None = None   # dense rating matrix
        self.mask:        np.ndarray | None = None   # 1 where rating exists
        self.loss_history: list[float] = []
        self.U_snapshots:  list[np.ndarray] = []     # for animation
        self.V_snapshots:  list[np.ndarray] = []

    # ── Data ingestion ─────────────────────────────────────────────────────────
    def fit_from_triplets(
        self,
        user_ids: list,
        item_ids: list,
        ratings:  list[float],
        progress_cb=None,   # callable(epoch, loss) for live UI updates
    ):
        """Build matrices and run ALS training."""
        self.users = sorted(set(user_ids))
        self.items = sorted(set(item_ids))
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: j for j, it in enumerate(self.items)}

        n_u = len(self.users)
        n_i = len(self.items)

        self.R    = np.zeros((n_u, n_i), dtype=np.float32)
        self.mask = np.zeros((n_u, n_i), dtype=np.float32)

        for u, it, r in zip(user_ids, item_ids, ratings):
            i = self.user_index[u]
            j = self.item_index[it]
            self.R[i, j]    = float(r)
            self.mask[i, j] = 1.0

        self._train(progress_cb)

    def _train(self, progress_cb=None):
        rng = np.random.default_rng(self.seed)
        n_u, n_i = self.R.shape
        k = self.n_factors
        lam = self.reg_lambda
        I_k = np.eye(k)

        # Random initialisation (small values)
        self.U = rng.normal(0, 0.1, (n_u, k)).astype(np.float64)
        self.V = rng.normal(0, 0.1, (n_i, k)).astype(np.float64)
        self.loss_history.clear()
        self.U_snapshots.clear()
        self.V_snapshots.clear()

        R = self.R.astype(np.float64)
        mask = self.mask.astype(np.float64)

        for epoch in range(self.n_iter):
            # ── Update U (user factors) ──
            for i in range(n_u):
                rated = np.where(mask[i] > 0)[0]
                if len(rated) == 0:
                    continue
                V_r = self.V[rated]                    # (n_r, k)
                r_i = R[i, rated]                      # (n_r,)
                A = V_r.T @ V_r + lam * I_k
                b = V_r.T @ r_i
                self.U[i] = np.linalg.solve(A, b)

            # ── Update V (item factors) ──
            for j in range(n_i):
                rated = np.where(mask[:, j] > 0)[0]
                if len(rated) == 0:
                    continue
                U_r = self.U[rated]
                r_j = R[rated, j]
                A = U_r.T @ U_r + lam * I_k
                b = U_r.T @ r_j
                self.V[j] = np.linalg.solve(A, b)

            # RMSE loss (only observed entries)
            R_hat = self.U @ self.V.T
            diff  = (R_hat - R) * mask
            rmse  = float(np.sqrt(np.sum(diff ** 2) / np.sum(mask)))
            self.loss_history.append(rmse)

            # Save snapshots every 5 epochs for animation
            if epoch % max(1, self.n_iter // 10) == 0:
                self.U_snapshots.append(self.U.copy())
                self.V_snapshots.append(self.V.copy())

            if progress_cb is not None:
                progress_cb(epoch, rmse)

        # Final snapshot
        self.U_snapshots.append(self.U.copy())
        self.V_snapshots.append(self.V.copy())

    # ── Prediction & recommendations ──────────────────────────────────────────
    def predict(self, user_id, item_id) -> float:
        i = self.user_index.get(user_id)
        j = self.item_index.get(item_id)
        if i is None or j is None:
            return 0.0
        return float(self.U[i] @ self.V[j])

    def recommend(self, user_id, top_n: int = 10, exclude_known: bool = True) -> list[tuple]:
        """Return top-N (item, predicted_score) for a user."""
        i = self.user_index.get(user_id)
        if i is None:
            return []
        scores = self.V @ self.U[i]           # (n_items,)
        if exclude_known:
            known = np.where(self.mask[i] > 0)[0]
            scores[known] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(self.items[j], float(scores[j])) for j in top_idx]

    def similar_users(self, user_id, top_n: int = 10) -> list[tuple]:
        """Cosine similarity between users in latent space."""
        i = self.user_index.get(user_id)
        if i is None:
            return []
        u_vec = self.U[i]
        norms  = np.linalg.norm(self.U, axis=1) + 1e-10
        sims   = self.U @ u_vec / (norms * (np.linalg.norm(u_vec) + 1e-10))
        sims[i] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:top_n]
        return [(self.users[j], float(sims[j])) for j in top_idx]

    def similar_items(self, item_id, top_n: int = 10) -> list[tuple]:
        """Cosine similarity between items in latent space."""
        j = self.item_index.get(item_id)
        if j is None:
            return []
        v_vec = self.V[j]
        norms  = np.linalg.norm(self.V, axis=1) + 1e-10
        sims   = self.V @ v_vec / (norms * (np.linalg.norm(v_vec) + 1e-10))
        sims[j] = -1
        top_idx = np.argsort(sims)[::-1][:top_n]
        return [(self.items[k], float(sims[k])) for k in top_idx]

    # ── Embeddings for visualisation ──────────────────────────────────────────
    def get_2d_embeddings(self, method: str = "pca"):
        """Project U and V to 2D for scatter plot. method='pca' or 'tsne'."""
        combined = np.vstack([self.U, self.V])
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
                proj = TSNE(n_components=2, random_state=42,
                            perplexity=min(30, len(combined) - 1)).fit_transform(combined)
            except Exception:
                proj = self._pca_2d(combined)
        else:
            proj = self._pca_2d(combined)
        n_u = len(self.U)
        return proj[:n_u], proj[n_u:]  # user_2d, item_2d

    @staticmethod
    def _pca_2d(X: np.ndarray) -> np.ndarray:
        X_c = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        return X_c @ Vt[:2].T
