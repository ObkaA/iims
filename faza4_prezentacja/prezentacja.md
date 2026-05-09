# Prezentacja — Outline (PPTX-ready)
# Przegląd metod optymalizacji wypukłej w modelowaniu statystycznym

> Format: każda sekcja `## Slajd N` = jeden slajd PowerPoint.
> Treść po `-` = punkt na slajdzie. Treść po `>` = notatki prelegenta.

---

## Slajd 1 — Strona tytułowa

- **Przegląd metod optymalizacji wypukłej**
- **w modelowaniu statystycznym**
- Implementacja, wizualizacja i porównanie algorytmów
- [Imię Nazwisko] · [Uczelnia / Kurs IIMS] · Maj 2026

> Krótkie przywitanie. Cel: pokazać, że optymalizacja wypukła jest fundamentem nowoczesnego modelowania statystycznego i ML.

---

## Slajd 2 — Motywacja

- Niemal każdy model statystyczny = **problem optymalizacji**
  - Regresja liniowa → minimalizacja MSE
  - Regresja logistyczna → minimalizacja log-loss (MLE)
  - SVM, sieci neuronowe → różne funkcje wypukłe / quasi-wypukłe
- Pytanie kluczowe: **jak efektywnie znaleźć minimum?**
- Różne algorytmy → różna szybkość zbieżności, koszty obliczeniowe, wymagania pamięciowe

> Motywacja przez przykład: dopasowanie modelu to de facto "szukanie dna" w przestrzeni parametrów.

---

## Slajd 3 — Podstawy: funkcje wypukłe

- **Definicja:** f jest wypukła ⟺ ∀x,y, λ∈[0,1]: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- Geometrycznie: odcinek łączący dwa punkty wykresu leży **powyżej** wykresu
- **Własności kluczowe:**
  - Każde minimum lokalne = minimum globalne
  - Warunek pierwszego rzędu jest wystarczający (jeśli ∇f(x*) = 0, to x* jest minimum)
  - Hessian f jest dodatnio półokreślony: H(x) ⪰ 0
- **Przykłady:** MSE, log-loss, norma L2 — wszystkie wypukłe!

> Podkreślić: gwarancja globalnej zbieżności = ogromna zaleta vs. ogólna optymalizacja nieciągła.

---

## Slajd 4 — Warunki optymalności (KKT)

- **Warunki Kuhna-Tuckera-Karush (KKT)** dla problemu:
  - min f(x) przy ograniczeniach g_i(x) ≤ 0, h_j(x) = 0
- Warunki konieczne i wystarczające (przy wypukłości):
  1. Stacjonarność: ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0
  2. Dopełnienie luzu: λᵢ · gᵢ(x*) = 0
  3. Primal/Dual feasibility: gᵢ(x*) ≤ 0, λᵢ ≥ 0
- **W praktyce (regresja bez ograniczeń):** wystarczy ∇f(θ) = 0

> KKT są podstawą teoretyczną algorytmów iteracyjnych — każdy z nich "zmierza w kierunku gradientu", aż ∇f ≈ 0.

---

## Slajd 5 — Mapa algorytmów

| Metoda | Rząd | Koszt / iterację | Szybkość zbieżności |
|--------|------|-----------------|---------------------|
| Gradient Descent (GD) | 1. | O(np) | Liniowa |
| Metoda Newtona | 2. | O(np² + p³) | Kwadratowa |
| SGD / Adam | 1. (stochast.) | O(batch · p) | Sub-liniowa |

- **Rząd metody** = ile pochodnych używa
- **Złoty kompromis:** Newton szybko, ale drogi; Adam wolniej, ale skalowalny

> Tu można narysować/pokazać "mapę" jako diagram.

---

## Slajd 6 — Gradient Descent — teoria

- **Idea:** poruszaj się w kierunku przeciwnym do gradientu
- **Aktualizacja:** θₜ₊₁ = θₜ − α · ∇f(θₜ)
- **Parametry:**
  - α (learning rate) — kluczowy hiperparametr
  - Zbyt duże α → oscylacje / rozbieżność
  - Zbyt małe α → bardzo wolna zbieżność
- **Zbieżność:** dla funkcji L-gładkich: O(1/t) po t iteracjach
- **Line search (backtracking Armijo):** automatyczny dobór α w każdym kroku

> Podkreślić intuicję: "schodzenie ze wzgórza". Pokazać wykres funkcji kwadratowej 2D z krokami GD.

---

## Slajd 7 — Gradient Descent — pseudokod i implementacja

```
Dane: X, y, loss_fn, grad_fn, α, max_iter, tol
Inicjalizacja: θ ← 0

Dla t = 1, 2, ..., max_iter:
    g ← ∇f(θ) = grad_fn(X, y, θ)
    Jeśli ||g||₂ < tol: STOP
    α ← line_search(θ, g)   # opcjonalnie
    θ ← θ − α · g

Zwróć θ
```

- Implementacja: klasa `GradientDescent(lr, max_iter, tol, line_search)`
- Parametry eksperymentu: lr=0.05, max_iter=500, backtracking Armijo
- Regresja liniowa: **RMSE = 0.8392** (zbiór testowy, n=300, p=5)

> Pokazać fragment kodu z pliku gradient_descent.py. Podkreślić, że nie używamy żadnej gotowej biblioteki do optymalizacji.

---

## Slajd 8 — Metoda Newtona — teoria

- **Idea:** aproksymacja f przez rozwinięcie Taylora 2. rzędu wokół θₜ
- f(θ) ≈ f(θₜ) + ∇f(θₜ)ᵀ(θ-θₜ) + ½(θ-θₜ)ᵀ H(θₜ) (θ-θₜ)
- **Kierunek Newtona:** rozwiązanie H(θₜ) · Δ = −∇f(θₜ)
- **Aktualizacja:** θₜ₊₁ = θₜ + Δ
- **Zbieżność kwadratowa** w okolicach minimum: ||θₜ₊₁ − θ*|| = O(||θₜ − θ*||²)
- **Koszt:** O(p²) na macierz Hessiana + O(p³) na rozwiązanie układu liniowego

> Metoda Newtona dla regresji logistycznej = algorytm IRLS (Iteratively Reweighted Least Squares).

---

## Slajd 9 — Metoda Newtona — wyniki i porównanie

```
Dane: X, y, loss_fn, grad_fn, hess_fn, max_iter, tol, reg

Inicjalizacja: θ ← 0

Dla t = 1, 2, ..., max_iter:
    g ← grad_fn(X, y, θ)
    Jeśli ||g||₂ < tol: STOP
    H ← hess_fn(X, y, θ) + reg · I    # regularyzacja (damped Newton)
    Δ ← solve(H, −g)
    θ ← θ + Δ

Zwróć θ
```

- **Wyniki (regresja logistyczna, n=500):**
  - Newton: **8 iteracji**, czas = 0.003 s, accuracy = 0.86
  - GD:  **1000 iteracji**, czas = 0.28 s, accuracy = 0.86
- **Wniosek:** Newton jest ~100× szybszy iteracyjnie, ale każda iteracja droższa

> Hessian dla log-reg: H = (1/n) XᵀWX, gdzie W = diag(p(1-p)) — macierz Fishera.

---

## Slajd 10 — SGD i Adam — motywacja i pseudokod

- **Problem GD przy dużych danych:** obliczenie ∇f wymaga przejścia przez **cały** zbiór
- **SGD:** gradient na losowym podzbiorze (mini-batch) ≈ gradient pełny w oczekiwaniu
- **Adam (Kingma & Ba, 2015):** adaptacyjny krok per parametr

```
Adam — pseudokod:
m ← 0,  v ← 0,  t ← 0

Dla każdej epoki:
    Losuj mini-batch (X_b, y_b)
    g ← grad_fn(X_b, y_b, θ)
    t ← t + 1
    m ← β₁·m + (1−β₁)·g               # moment 1. rzędu
    v ← β₂·v + (1−β₂)·g²              # moment 2. rzędu
    m̂ ← m / (1−β₁ᵗ)                  # korekcja biasu
    v̂ ← v / (1−β₂ᵗ)
    θ ← θ − α · m̂ / (√v̂ + ε)
```

- Domyślne parametry: α=0.001, β₁=0.9, β₂=0.999, ε=1e-8

> Adam "pamięta" historię gradientów — wolniej rośnie krok dla "szybkich" kierunków.

---

## Slajd 11 — Dane symulowane

- **Regresja liniowa:** y = Xβ + ε, X ~ N(0,I), ε ~ N(0, 0.8²)
  - n = 300 obserwacji, p = 5 cech
  - Podział: 80% trening / 20% test, seed = 42
- **Klasyfikacja binarna:** dwie chmury gaussowskie (p=2 cechy)
  - Klasa 0: X ~ N(0, I), Klasa 1: X ~ N(1.8, I)
  - n = 500, podział 80/20
- **Preprocessing:** standaryzacja (μ=0, σ=1) z parametrami ze zbioru treningowego
- **Dodanie wyrazu wolnego:** kolumna jedynek do macierzy X

> Seed zapewnia pełną reprodukowalność — każdy może odtworzyć wyniki uruchamiając main.py.

---

## Slajd 12 — Wyniki: krzywe zbieżności

**[Wstawić wykres: `wyniki/convergence_logistic.png`]**

- Oś X: numer iteracji / epoki | Oś Y: wartość funkcji straty (log-log)
- **Newton** — zbieżność kwadratowa: 8 iteracji do minimum
- **GD** — zbieżność liniowa: 1000 iteracji, stabilny spadek
- **Adam** — zbieżność stochastyczna: oscylacje na początku, potem stabilizacja

**[Wstawić wykres: `wyniki/convergence_linear.png`]** — regresja liniowa (GD vs. Adam)

> Wykres log-skali wyraźnie pokazuje różnicę w tempie zbieżności. Newton "spada pionowo".

---

## Slajd 13 — Tabela wyników

| Metoda | Model | Metryka | Iteracje | Czas [s] |
|--------|-------|---------|----------|----------|
| GD (line search) | Regresja liniowa | RMSE = 0.8392 | 412 | 0.066 |
| Adam | Regresja liniowa | RMSE = 0.8404 | 300 epok | 0.142 |
| GD (line search) | Reg. logistyczna | Acc = 0.860 | 1000 | 0.283 |
| **Newton** | **Reg. logistyczna** | **Acc = 0.860** | **8** | **0.003** |
| Adam | Reg. logistyczna | Acc = 0.860 | 200 epok | 0.304 |

- Wszystkie metody osiągają **porównywalną dokładność predykcji**
- Newton: drastycznie mniej iteracji — kwadratowa zbieżność w praktyce
- Adam: najwolniejszy, ale skalowalny na duże zbiory

---

## Slajd 14 — Wizualizacje: granica decyzyjna i macierz pomyłek

**[Wstawić wykres: `wyniki/decision_boundary.png`]**
- Granica liniowa — zgodna z liniową separowalnością danych
- Wszystkie 3 metody wyznaczyły praktycznie identyczną granicę

**[Wstawić wykres: `wyniki/confusion_matrix.png`]**
- TN ≈ 86, TP ≈ 80, FP ≈ 14, FN ≈ 20 (podobne dla wszystkich metod)
- Accuracy = 0.86 ↔ 86 dobrych predykcji na 100

**[Wstawić wykres: `wyniki/time_vs_accuracy.png`]**
- Newton: najszybszy + dokładny | Adam: wolniejszy + porównywalny

---

## Slajd 15 — Wnioski i ograniczenia

**Wnioski:**
- Optymalizacja wypukła **gwarantuje** zbieżność do globalnego minimum
- Metoda Newtona zbiega kwadratowo — idealna dla małych/średnich p
- GD z line search — solidna, prosta alternatywa dla umiarkowanych danych
- Adam — kluczowy dla dużych zbiorów i modeli głębokiego uczenia

**Ograniczenia i kierunki dalszych badań:**
- Newton: O(p³) per iterację — nieefektywny dla p >> 1000
- GD: wrażliwość na dobór α, wolna zbieżność dla źle uwarunkowanych problemów
- Adam: brak gwarancji zbieżności w ogólnym przypadku (patrz: AMSGrad)
- Dalej: metody quasi-Newton (L-BFGS), metody proksymalne, regularyzacja (LASSO, Ridge)

---

## Slajd 16 — Literatura i pytania

**Kluczowe źródła:**
- Boyd & Vandenberghe, *Convex Optimization* (2004) — fundament teorii
- Nocedal & Wright, *Numerical Optimization* (2006) — algorytmy numeryczne
- Bottou et al., *Optimization Methods for Large-Scale ML*, SIAM (2018)
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR (2015)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009)

**Kod projektu:**
- `faza2_implementacja/` — implementacje od podstaw (numpy only)
- `faza3_symulacje/main.py` — uruchomienie pełnego pipeline
- `faza3_symulacje/wyniki/` — wszystkie wykresy PNG

---
*Dziękuję za uwagę. Pytania?*
