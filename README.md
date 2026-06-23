# ConvexML — Convex Optimization Explorer v2

> Desktopowa aplikacja edukacyjno-analityczna: optymalizacja wypukła + systemy rekomendacyjne oparte na Matrix Factorization.

---

## Nowe funkcjonalności (v2)

### Nowe algorytmy
| Algorytm | Typ | Kluczowa cecha |
|---|---|---|
| **Newton Method** | Drugiego rzędu | θ ← θ − α·H⁻¹·∇L — kwadratowa zbieżność |
| **ALS** | Closed-form | Alternating Least Squares dla Matrix Factorization |

### Adam Check (diagnostyka rzeczywistego treningu)
- uruchamia kontrolowane porównanie `Adam` vs `SGD` vs `Gradient Descent` na tym samym modelu, datasecie, podziale train/test i learning rate
- zachowuje konfigurację z zakładki Optimization i nie zmienia wybranego algorytmu ani hiperparametrów
- Adam i SGD korzystają z tej samej deterministycznej sekwencji mini-batchy; każdy optimizer startuje od świeżych, identycznych parametrów modelu
- pokazuje wspólne wykresy lossu mierzonego na tej samej deterministycznej próbce testowej, normy gradientu i normy kroku, ranking oraz trajektorie parametrów
- trajektoria używa bezpośrednio `θ₀–θ₁` dla dwóch parametrów i wspólnej projekcji PCA dla modeli wielowymiarowych
- `Gradient Descent` używa pełnego zbioru treningowego, a Adam i SGD wybranego mini-batcha
- `Adam Health` wykrywa NaN/∞, eksplozję lossu, stagnację i niestabilność oraz zwraca werdykt `OK`, `WARNING` albo `FAILURE`
- osobny fixed-config benchmark pokazuje zwycięzcę i ranking; słabszy wynik Adama nie zmienia jego werdyktu zdrowia
- nie uruchamia osobnych, syntetycznych scenariuszy „Adam Failures”

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
python implementacja/main.py
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
