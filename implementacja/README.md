# ConvexML — Convex Optimization Explorer v2

> Desktopowa aplikacja edukacyjno-analityczna: optymalizacja wypukła + systemy rekomendacyjne oparte na Matrix Factorization.

---

## Nowe funkcjonalności (v2)

### Nowe algorytmy
| Algorytm | Typ | Kluczowa cecha |
|---|---|---|
| **Newton Method** | Drugiego rzędu | θ ← θ − α·H⁻¹·∇L — kwadratowa zbieżność |
| **ALS** | Closed-form | Alternating Least Squares dla Matrix Factorization |

### Music Recommendation (nowa zakładka ♫)
- Matrix Factorization: `R ≈ U · V^T` trenowana przez ALS
- Synthetic Last.fm dataset (80 użytkowników, 25 artystów, klastry gatunków)
- Ładowanie realnych CSV z Kaggle Last.fm
- Top-N rekomendacji z predicted score
- Podobni użytkownicy i podobne utwory (cosine similarity)
- Animacja ewolucji latent vectors przez epoki

### Wizualizacje w module rekomendacji
- 📉 ALS Loss — RMSE per epoch
- 🗃 Rating Matrix — heatmapa interakcji
- 🌐 2D Embedding — PCA/t-SNE projekcja
- 🎵 Recommendations — Top-N bar chart
- 🔗 Similar Artists
- 👥 Similar Users
- 🎞 Latent Animation — slider po snapshotach

---

## Instalacja

```bash
pip install -r requirements.txt
python main.py
```

---

## Struktura projektu

```
convex_ml_app/
├── algorithms/
│   ├── gradient_descent.py
│   ├── sgd.py
│   ├── adam.py
│   ├── newton.py          ← NOWY: Hessian-based 2nd order
│   └── als.py             ← NOWY: Closed-form dla MF
├── models/
│   ├── linear_regression.py
│   └── logistic_regression.py  ← dodano hessian() i loss_gradient_hessian()
├── recommendation/        ← NOWY MODUŁ
│   └── matrix_factorization.py
├── datasets/              ← NOWY MODUŁ
│   └── lastfm_loader.py   ← synthetic + CSV autodetect
├── visualization/
│   ├── __init__.py        ← wykresy optymalizacji
│   └── rec_viz.py         ← NOWY: wykresy MF i rekomendacji
├── ui/
│   ├── main_window.py     ← rozszerzony o zakładkę muzyczną
│   └── rec_panel.py       ← NOWY: cały panel rekomendacji
├── training_engine.py     ← rozszerzony o obsługę Hesjanu (Newton)
└── requirements.txt
```

---

## Wyniki edukacyjne

Newton Method: **10 iteracji** → accuracy 93.7%  
Gradient Descent: **200 iteracji** → accuracy 93.0%

Kwadratowa zbieżność Newtona = 20× mniejsza liczba iteracji przy tej samej jakości.

---

## Format CSV (Last.fm)

```csv
userId,artistId,weight
12345,Radiohead,15234
67890,Burial,22100
```

Obsługiwane nazwy kolumn: `userId/user_id/user`, `artistId/artist/track/song`, `weight/playcount/rating`

---

## Linki

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/)
- [Matrix Factorization for Recommender Systems (Netflix)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [ALS for Implicit Feedback](http://yifanhu.net/PUB/cf.pdf)
