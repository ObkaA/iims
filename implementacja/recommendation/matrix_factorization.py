"""
Matrix Factorization via Alternating Least Squares (ALS).
Implemented entirely from scratch — pure NumPy only, no sklearn, no scipy.

════════════════════════════════════════════════════════════════════════════════
MODEL
════════════════════════════════════════════════════════════════════════════════
  R ≈ U · Vᵀ
  U ∈ ℝ^{n_users × k}  — user latent factor matrix
  V ∈ ℝ^{n_items × k}  — item latent factor matrix
  k                     — number of latent dimensions (hyperparameter)

════════════════════════════════════════════════════════════════════════════════
OBJECTIVE  (regularised, only on observed entries Ω)
════════════════════════════════════════════════════════════════════════════════
  min_{U,V}  Σ_{(i,j)∈Ω} (R_{ij} − uᵢ·vⱼ)²  +  λ(‖U‖_F² + ‖V‖_F²)

════════════════════════════════════════════════════════════════════════════════
ALS UPDATE (closed-form ridge regression per entity)
════════════════════════════════════════════════════════════════════════════════
  Fix V, update each row uᵢ:
      uᵢ = (Vᵢᵀ Vᵢ + λI)⁻¹ Vᵢᵀ rᵢ
  where Vᵢ = V rows for items rated by user i,  rᵢ = those ratings.

  Fix U, update each row vⱼ:
      vⱼ = (Uⱼᵀ Uⱼ + λI)⁻¹ Uⱼᵀ rⱼ

  Each sub-problem is standard ridge regression → unique, closed-form solution.
  Alternating between the two is guaranteed to decrease the objective each step.

════════════════════════════════════════════════════════════════════════════════
FOLD-IN (new user, V fixed)
════════════════════════════════════════════════════════════════════════════════
  u_new = (V_Ω^T V_Ω + λI)^{-1} V_Ω^T r
  One ALS step for the new user without retraining V.

════════════════════════════════════════════════════════════════════════════════
PCA  (from scratch — no sklearn)
════════════════════════════════════════════════════════════════════════════════
  1. Centre: X_c = X − mean(X, axis=0)
  2. Thin SVD: X_c = U S Vᵀ   (np.linalg.svd)
  3. Project: Z = X_c · V[:, :2]
  More stable than eigendecomposition of XᵀX; handles d > n.

════════════════════════════════════════════════════════════════════════════════
t-SNE  (from scratch — no sklearn)
════════════════════════════════════════════════════════════════════════════════
  High-dim similarity:  p_{ij} ∝ exp(−‖xᵢ−xⱼ‖²/2σ²)   Gaussian kernel
  Low-dim  similarity:  q_{ij} ∝ (1+‖yᵢ−yⱼ‖²)^{-1}     Student-t kernel
  Objective:  KL(P‖Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
  Gradient:   ∂C/∂yᵢ = 4Σⱼ(pᵢⱼ−qᵢⱼ)(yᵢ−yⱼ)(1+‖yᵢ−yⱼ‖²)^{-1}
  Optimised with gradient descent + momentum.
"""
from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PCA — from scratch
# ══════════════════════════════════════════════════════════════════════════════
def _pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Project X ∈ ℝ^{n×d} → ℝ^{n×2} via PCA.

    Algorithm:
        X_c = X − μ                 centre columns
        X_c = U S Vᵀ               thin SVD  (O(n·d·min(n,d)))
        Z   = X_c · V[:2]ᵀ         project onto top-2 principal components

    Using SVD (not eigendecomp of XᵀX) avoids squaring the condition number.
    """
    X_c = X.astype(np.float64) - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)   # Vt: (min(n,d), d)
    return X_c @ Vt[:2].T                                 # (n, 2)


# ══════════════════════════════════════════════════════════════════════════════
# t-SNE — from scratch
# ══════════════════════════════════════════════════════════════════════════════
def _compute_P(D2: np.ndarray, perplexity: float) -> np.ndarray:
    """
    Build symmetrised joint probability P via binary search on σᵢ per row.

    For each i: find βᵢ=1/(2σᵢ²) s.t. 2^{H(Pᵢ)} = perplexity
    where H(Pᵢ) = −Σⱼ pⱼ|ᵢ log₂ pⱼ|ᵢ  (Shannon entropy in bits).
    Binary search on βᵢ ∈ (1e-20, 1e6), 50 iterations per row.
    """
    n        = D2.shape[0]
    P        = np.zeros((n, n), dtype=np.float64)
    log_perp = np.log2(float(perplexity))

    for i in range(n):
        d     = D2[i].copy().astype(np.float64)
        d[i]  = np.inf
        lo, hi = 1e-20, 1e6

        for _ in range(50):
            beta   = (lo + hi) / 2.0
            exp_d  = np.exp(-d * beta)
            exp_d[i] = 0.0
            S      = exp_d.sum() + 1e-10
            p_i    = exp_d / S
            pos    = p_i > 0
            H      = -np.sum(p_i[pos] * np.log2(p_i[pos] + 1e-12))
            if H < log_perp:
                hi = beta
            else:
                lo = beta

        P[i] = p_i

    # Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / 2n
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def _tsne_2d(
    X:          np.ndarray,
    n_iter:     int   = 500,
    perplexity: float = 20.0,
    lr:         float = 150.0,
    seed:       int   = 42,
) -> np.ndarray:
    """
    t-SNE: project X ∈ ℝ^{n×d} → ℝ^{n×2}.  Pure NumPy.

    Van der Maaten & Hinton (2008).  Capped at n=120 for interactive speed;
    falls back to PCA for larger inputs.

    Gradient:
        ∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ−yⱼ)(1+‖yᵢ−yⱼ‖²)^{-1}
    Optimiser: GD with momentum annealed from 0.5 → 0.8 at iteration 250.
    """
    n = X.shape[0]
    if n > 120:
        return _pca_2d(X)   # silent fallback

    perplexity = min(float(perplexity), n - 1.0)

    # Pairwise squared distances in high-dim
    sq   = np.sum(X.astype(np.float64) ** 2, axis=1)
    D2   = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(D2, 0.0)

    P = _compute_P(D2, perplexity)

    # Initialise Y via PCA (better than random for small k)
    rng = np.random.default_rng(seed)
    Y   = _pca_2d(X) * 0.01 + rng.normal(0, 1e-4, (n, 2))
    vel = np.zeros_like(Y)
    mom = 0.5

    for t in range(1, n_iter + 1):
        if t == 250:
            mom = 0.8

        # Low-dim pairwise Student-t kernel: num_{ij} = (1+‖yᵢ−yⱼ‖²)^{-1}
        sq_y = np.sum(Y ** 2, axis=1)
        D2_y = np.maximum(sq_y[:, None] + sq_y[None, :] - 2.0 * (Y @ Y.T), 0.0)
        np.fill_diagonal(D2_y, 0.0)
        num  = 1.0 / (1.0 + D2_y)
        np.fill_diagonal(num, 0.0)
        Q    = np.maximum(num / (num.sum() + 1e-10), 1e-12)

        # Gradient: attraction (P) and repulsion (Q) forces
        PQ   = (P - Q) * num                              # (n, n)
        grad = 4.0 * (PQ.sum(axis=1, keepdims=True) * Y - PQ @ Y)

        vel  = mom * vel - lr * grad
        Y   += vel

    return Y


# ══════════════════════════════════════════════════════════════════════════════
# Matrix Factorization
# ══════════════════════════════════════════════════════════════════════════════
class MatrixFactorization:
    """
    Matrix Factorization via ALS — pure NumPy, no external ML libraries.

    Parameters
    ----------
    n_factors:  k   number of latent dimensions
    reg_lambda: λ   L2 regularisation (prevents overfitting to sparse data)
    n_iter:         number of ALS epochs (alternating passes over U and V)
    seed:           RNG seed for reproducibility
    """

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

        self.U:            np.ndarray | None = None   # (n_users, k)
        self.V:            np.ndarray | None = None   # (n_items, k)
        self.user_index:   dict[str, int]   = {}
        self.item_index:   dict[str, int]   = {}
        self.users:        list[str]        = []
        self.items:        list[str]        = []
        self.R:            np.ndarray | None = None   # (n_users, n_items) dense
        self.mask:         np.ndarray | None = None   # 1 where rating observed
        self.loss_history: list[float]      = []
        self.U_snapshots:  list[np.ndarray] = []      # for latent animation
        self.V_snapshots:  list[np.ndarray] = []

    # ── Data ingestion ─────────────────────────────────────────────────────────
    def fit_from_triplets(
        self,
        user_ids:    list,
        item_ids:    list,
        ratings:     list[float],
        progress_cb=None,    # callable(epoch: int, rmse: float)
    ):
        """Build dense rating matrix from triplets and run ALS."""
        self.users = sorted(set(user_ids))
        self.items = sorted(set(item_ids))
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: j for j, it in enumerate(self.items)}

        n_u, n_i = len(self.users), len(self.items)
        self.R    = np.zeros((n_u, n_i), dtype=np.float64)
        self.mask = np.zeros((n_u, n_i), dtype=np.float64)

        for u, it, r in zip(user_ids, item_ids, ratings):
            i = self.user_index[u]
            j = self.item_index[it]
            self.R[i, j]    = float(r)
            self.mask[i, j] = 1.0

        self._train(progress_cb)

    # ── ALS training ───────────────────────────────────────────────────────────
    def _train(self, progress_cb=None):
        """
        Full ALS loop.  Each epoch:
          1. For every user  i: solve uᵢ = (Vᵢᵀ Vᵢ + λI)^{-1} Vᵢᵀ rᵢ
          2. For every item  j: solve vⱼ = (Uⱼᵀ Uⱼ + λI)^{-1} Uⱼᵀ rⱼ
          3. Compute RMSE on observed entries only
          4. Save snapshot for animation every n_iter//10 epochs
        """
        rng  = np.random.default_rng(self.seed)
        n_u, n_i = self.R.shape
        k    = self.n_factors
        lam  = self.reg_lambda
        I_k  = np.eye(k, dtype=np.float64)

        # Small random init — breaks symmetry without large initial loss
        self.U = rng.normal(0.0, 0.1, (n_u, k)).astype(np.float64)
        self.V = rng.normal(0.0, 0.1, (n_i, k)).astype(np.float64)
        self.loss_history.clear()
        self.U_snapshots.clear()
        self.V_snapshots.clear()

        snap_every = max(1, self.n_iter // 10)

        for epoch in range(self.n_iter):

            # ── Update U: fix V, solve for each user ─────────────────────────
            # uᵢ = (Vᵢᵀ Vᵢ + λI)^{-1} Vᵢᵀ rᵢ
            for i in range(n_u):
                rated = np.where(self.mask[i] > 0)[0]
                if len(rated) == 0:
                    continue
                V_r = self.V[rated]          # (|Ωᵢ|, k)
                r_i = self.R[i, rated]       # (|Ωᵢ|,)
                A   = V_r.T @ V_r + lam * I_k
                b   = V_r.T @ r_i
                self.U[i] = np.linalg.solve(A, b)

            # ── Update V: fix U, solve for each item ─────────────────────────
            # vⱼ = (Uⱼᵀ Uⱼ + λI)^{-1} Uⱼᵀ rⱼ
            for j in range(n_i):
                rated = np.where(self.mask[:, j] > 0)[0]
                if len(rated) == 0:
                    continue
                U_r = self.U[rated]          # (|Ωⱼ|, k)
                r_j = self.R[rated, j]       # (|Ωⱼ|,)
                A   = U_r.T @ U_r + lam * I_k
                b   = U_r.T @ r_j
                self.V[j] = np.linalg.solve(A, b)

            # ── RMSE on observed entries ──────────────────────────────────────
            R_hat = self.U @ self.V.T
            diff  = (R_hat - self.R) * self.mask
            rmse  = float(np.sqrt(np.sum(diff ** 2) / (np.sum(self.mask) + 1e-10)))
            self.loss_history.append(rmse)

            if epoch % snap_every == 0:
                self.U_snapshots.append(self.U.copy())
                self.V_snapshots.append(self.V.copy())

            if progress_cb is not None:
                progress_cb(epoch, rmse)

        # Always save final snapshot
        self.U_snapshots.append(self.U.copy())
        self.V_snapshots.append(self.V.copy())

    # ── Fold-in: personalised vector for a NEW user ───────────────────────────
    def fold_in_user(
        self,
        item_ids: list[str],
        ratings:  list[float],
        reg_lambda: float | None = None,
    ) -> np.ndarray:
        """
        Compute latent vector for a new user without retraining.

        This is one ALS update treating the new user as an extra row,
        with V held fixed (the pre-trained item factors).

            u_new = (V_Ω^T V_Ω + λI)^{-1} V_Ω^T r

        Args:
            item_ids:   items the user has interacted with
            ratings:    corresponding scores
            reg_lambda: override λ (default: self.reg_lambda)

        Returns:
            u_new: np.ndarray shape (k,)
        """
        if self.V is None:
            raise RuntimeError("Model not trained yet.")

        lam  = reg_lambda if reg_lambda is not None else self.reg_lambda
        k    = self.n_factors
        I_k  = np.eye(k, dtype=np.float64)

        # Keep only items present in training catalogue
        known_idx, known_r = [], []
        for item, r in zip(item_ids, ratings):
            j = self.item_index.get(item)
            if j is not None:
                known_idx.append(j)
                known_r.append(float(r))

        if not known_idx:
            return np.zeros(k, dtype=np.float64)

        V_known = self.V[known_idx]                          # (|Ω|, k)
        r_vec   = np.array(known_r, dtype=np.float64)        # (|Ω|,)
        A = V_known.T @ V_known + lam * I_k                  # (k, k)
        b = V_known.T @ r_vec                                 # (k,)
        return np.linalg.solve(A, b)                          # (k,)

    # ── Cosine similarity (from scratch) ──────────────────────────────────────
    @staticmethod
    def _cosine(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        cos(vec, rowᵢ) = (vec · rowᵢ) / (‖vec‖ · ‖rowᵢ‖)

        Vectorised over all rows of matrix at once.
        """
        num   = matrix @ vec                                        # (n,)
        denom = (np.linalg.norm(matrix, axis=1)
                 * (np.linalg.norm(vec) + 1e-10) + 1e-10)          # (n,)
        return num / denom                                          # (n,)

    # ── Recommendations for an existing training user ─────────────────────────
    def recommend(
        self,
        user_id:       str,
        top_n:         int  = 10,
        exclude_known: bool = True,
    ) -> list[tuple[str, float]]:
        """Top-N items for a training user by dot product score."""
        i = self.user_index.get(user_id)
        if i is None:
            return []
        return self.recommend_for_vector(
            self.U[i], top_n=top_n,
            exclude_items=(
                [self.items[j] for j in np.where(self.mask[i] > 0)[0]]
                if exclude_known else []
            ),
        )

    # ── Recommendations for ANY latent vector (e.g. new user) ─────────────────
    def recommend_for_vector(
        self,
        u_vec:         np.ndarray,
        top_n:         int = 10,
        exclude_items: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Score all items: s_j = u_vec · v_j  (dot product in latent space).
        Exclude items in exclude_items, return top_n sorted descending.
        """
        scores = self.V @ u_vec.astype(np.float64)     # (n_items,)
        if exclude_items:
            for item in exclude_items:
                j = self.item_index.get(item)
                if j is not None:
                    scores[j] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(self.items[j], float(scores[j])) for j in top_idx]

    # ── Similar users (accepts latent vector, not user_id) ────────────────────
    def similar_users(
        self,
        u_vec: np.ndarray,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Cosine similarity between u_vec and every training user.

        Args:
            u_vec: (k,) query vector — can be a new user from fold_in_user()
            top_n: number of similar users to return

        Returns:
            [(user_id, cosine_similarity), ...] sorted descending
        """
        sims    = self._cosine(u_vec.astype(np.float64), self.U)  # (n_users,)
        top_idx = np.argsort(sims)[::-1][:top_n]
        return [(self.users[j], float(sims[j])) for j in top_idx]

    # ── Similar items ──────────────────────────────────────────────────────────
    def similar_items(
        self,
        item_id: str,
        top_n:   int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine similarity between items in latent space."""
        j = self.item_index.get(item_id)
        if j is None:
            return []
        v_vec    = self.V[j]
        sims     = self._cosine(v_vec, self.V)
        sims[j]  = -1.0    # exclude self
        top_idx  = np.argsort(sims)[::-1][:top_n]
        return [(self.items[k], float(sims[k])) for k in top_idx]

    # ── 2-D embeddings (PCA or t-SNE, both from scratch) ─────────────────────
    def get_2d_embeddings(
        self,
        method:     str                = "pca",
        extra_vec:  np.ndarray | None  = None,
        extra_label: str               = "YOU",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Project U and V (and optionally an extra vector) to 2D.

        Args:
            method:     'pca'  — fast, deterministic, linear
                        'tsne' — slower, non-linear, better cluster separation
            extra_vec:  (k,) latent vector to include (e.g. new user from fold-in)

        Returns:
            (user_2d, item_2d, extra_2d)
              user_2d:  (n_users, 2)
              item_2d:  (n_items, 2)
              extra_2d: (1, 2) or None
        """
        has_extra = extra_vec is not None
        parts     = [self.U, self.V]
        if has_extra:
            parts.append(extra_vec[np.newaxis])   # (1, k)
        combined  = np.vstack(parts).astype(np.float64)

        if method == "tsne":
            proj = _tsne_2d(combined)
        else:
            proj = _pca_2d(combined)

        n_u  = len(self.U)
        n_i  = len(self.V)
        u_2d = proj[:n_u]
        i_2d = proj[n_u : n_u + n_i]
        e_2d = proj[n_u + n_i : n_u + n_i + 1] if has_extra else None
        return u_2d, i_2d, e_2d
