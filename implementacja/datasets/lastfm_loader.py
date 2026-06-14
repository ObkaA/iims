"""
Last.fm dataset loader.
Supports two common Kaggle Last.fm CSV formats:

Format A — user_artists (user_id, artistID, weight):
    userId,artistId,weight   or   userID,artistID,weight

Format B — listening events (user, artist, track, timestamp):
    user,artist,track,timestamp  →  playcount aggregated per user×artist

Also generates a synthetic music dataset for demo purposes.
"""
from __future__ import annotations
import csv
import numpy as np
from pathlib import Path


# ── Synthetic demo dataset ─────────────────────────────────────────────────────
_ARTISTS = [
    "Radiohead", "Portishead", "Massive Attack", "Björk", "Aphex Twin",
    "The National", "Nick Cave", "PJ Harvey", "Boards of Canada",
    "Sigur Rós", "Four Tet", "Burial", "James Blake", "Bon Iver",
    "Sufjan Stevens", "Fleet Foxes", "Grouper", "Mount Eerie",
    "Deerhunter", "Beach House", "Animal Collective", "Grizzly Bear",
    "War on Drugs", "Sharon Van Etten", "Slowdive", "My Bloody Valentine",
    "Cocteau Twins", "Dead Can Dance", "This Mortal Coil", "Talk Talk",
]

def generate_synthetic_lastfm(
    n_users: int = 80,
    n_artists: int = 25,
    sparsity: float = 0.35,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Returns (user_ids, artist_ids, playcounts).
    Simulates realistic listening patterns: users cluster around 3–4 genres.
    """
    rng = np.random.default_rng(seed)
    artists = _ARTISTS[:n_artists]
    users   = [f"user_{i:03d}" for i in range(n_users)]

    # Hidden genre clusters (3 groups of artists, 3 groups of users)
    n_clusters = 3
    artist_cluster = np.array([i % n_clusters for i in range(n_artists)])
    user_cluster   = rng.integers(0, n_clusters, n_users)

    user_ids, artist_ids, playcounts = [], [], []
    for ui, u in enumerate(users):
        uc = user_cluster[ui]
        for ai, a in enumerate(artists):
            ac = artist_cluster[ai]
            # Base probability higher for same cluster
            base_prob = 0.6 if ac == uc else 0.15
            if rng.random() < base_prob * (1 / sparsity) * 0.35:
                # Log-normal play counts
                plays = int(np.clip(rng.lognormal(4.5, 1.5), 5, 5000))
                user_ids.append(u)
                artist_ids.append(a)
                playcounts.append(float(plays))

    return user_ids, artist_ids, playcounts


# ── CSV Loaders ────────────────────────────────────────────────────────────────
def load_lastfm_csv(path: str, max_users: int = 200, max_interactions: int = 5000) -> tuple:
    """
    Auto-detect Last.fm CSV format and return (user_ids, item_ids, ratings).
    Ratings are log-normalised playcounts in [1, 10].
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader  = csv.DictReader(f, dialect=dialect)
        rows    = list(reader)

    if not rows:
        raise ValueError("Empty CSV file.")

    cols = [c.lower().strip() for c in rows[0].keys()]

    # Detect column names
    def _find(candidates):
        for c in candidates:
            for col in rows[0].keys():
                if col.lower().strip() == c:
                    return col
        return None

    user_col   = _find(["userid", "user_id", "user"])
    item_col   = _find(["artistid", "artist_id", "artist", "track", "song", "item"])
    weight_col = _find(["weight", "playcount", "plays", "count", "rating", "scrobbles"])

    if user_col is None or item_col is None:
        raise ValueError(f"Cannot detect user/item columns. Found: {list(rows[0].keys())}")

    # Aggregate: if no weight column, count occurrences
    from collections import defaultdict
    agg: dict = defaultdict(float)
    user_set: set = set()

    for row in rows:
        u  = str(row[user_col]).strip()
        it = str(row[item_col]).strip()
        if not u or not it or u == "nan" or it == "nan":
            continue
        w = 1.0
        if weight_col:
            try:
                w = float(row[weight_col])
            except (ValueError, KeyError):
                w = 1.0
        agg[(u, it)] += w
        user_set.add(u)
        if len(agg) >= max_interactions * 3:
            break

    # Limit users
    top_users = sorted(user_set)[:max_users]
    top_set   = set(top_users)

    user_ids, item_ids, ratings = [], [], []
    for (u, it), w in agg.items():
        if u not in top_set:
            continue
        user_ids.append(u)
        item_ids.append(it)
        ratings.append(w)
        if len(user_ids) >= max_interactions:
            break

    if not user_ids:
        raise ValueError("No valid interactions found after filtering.")

    # Log-normalise ratings to [1, 10]
    r_arr = np.array(ratings, dtype=np.float64)
    log_r = np.log1p(r_arr)
    lo, hi = log_r.min(), log_r.max()
    if hi > lo:
        norm = 1.0 + 9.0 * (log_r - lo) / (hi - lo)
    else:
        norm = np.ones_like(log_r) * 5.0
    ratings = norm.tolist()

    return user_ids, item_ids, ratings
