# Przegląd literatury — Optymalizacja Wypukła w Modelowaniu Statystycznym

## 1. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

Fundamentalna monografia z zakresu optymalizacji wypukłej, obejmująca teorię zbiorów i funkcji wypukłych, warunki optymalności (KKT), dualność Lagrange'a oraz klasyczne algorytmy numeryczne. Stanowi podstawowe źródło definicji i twierdzeń stosowanych w projekcie. Dostępna bezpłatnie pod adresem: https://web.stanford.edu/~boyd/cvxbook/

```bibtex
@book{boyd2004convex,
  title     = {Convex Optimization},
  author    = {Boyd, Stephen and Vandenberghe, Lieven},
  year      = {2004},
  publisher = {Cambridge University Press},
  address   = {Cambridge, UK}
}
```

---

## 2. Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.

Szczegółowe omówienie metod numerycznych optymalizacji, w tym metod gradientowych pierwszego i drugiego rzędu (metoda Newtona, quasi-Newton, L-BFGS), technik liniowego przeszukiwania (line search) oraz kryteriów zbieżności. Rozdział 3 (gradient descent) i 6 (metoda Newtona) są szczególnie istotne dla projektu.

```bibtex
@book{nocedal2006numerical,
  title     = {Numerical Optimization},
  author    = {Nocedal, Jorge and Wright, Stephen J.},
  year      = {2006},
  edition   = {2},
  publisher = {Springer},
  address   = {New York}
}
```

---

## 3. Bottou, L., Curtis, F. E. & Nocedal, J. (2018). Optimization Methods for Large-Scale Machine Learning. *SIAM Review*, 60(2), 223–311.

Kompleksowy przegląd metod stochastycznych (SGD, mini-batch SGD, metody adaptacyjne) ze szczególnym uwzględnieniem zastosowań w uczeniu maszynowym i problemach dużej skali. Autorzy omawiają warunki zbieżności, dobór kroku uczenia oraz porównanie metod deterministycznych i stochastycznych pod kątem złożoności obliczeniowej.

```bibtex
@article{bottou2018optimization,
  title   = {Optimization Methods for Large-Scale Machine Learning},
  author  = {Bottou, L\'eon and Curtis, Frank E. and Nocedal, Jorge},
  journal = {SIAM Review},
  volume  = {60},
  number  = {2},
  pages   = {223--311},
  year    = {2018}
}
```

---

## 4. Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

Artykuł wprowadzający algorytm Adam (Adaptive Moment Estimation) — adaptacyjną metodę gradientową łączącą idee RMSProp i momentum. Adam estymuje momenty pierwszego i drugiego rzędu gradientu, co pozwala na adaptacyjny dobór kroku uczenia per parametr. Jest to jeden z najczęściej stosowanych optymalizatorów w uczeniu maszynowym.

```bibtex
@inproceedings{kingma2015adam,
  title     = {Adam: A Method for Stochastic Optimization},
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  booktitle = {3rd International Conference on Learning Representations (ICLR)},
  year      = {2015}
}
```

---

## 5. Nesterov, Y. (2018). *Lectures on Convex Optimization* (2nd ed.). Springer.

Zaawansowane ujęcie teoretyczne metod optymalizacji wypukłej, w tym metody gradientów sprzężonych, metod momentum (Nesterov Accelerated Gradient) oraz dolnych ograniczeń złożoności algorytmów dla klas funkcji wypukłych. Rozdział 2 zawiera kluczowe wyniki dotyczące szybkości zbieżności metod pierwszego rzędu.

```bibtex
@book{nesterov2018lectures,
  title     = {Lectures on Convex Optimization},
  author    = {Nesterov, Yurii},
  year      = {2018},
  edition   = {2},
  publisher = {Springer},
  address   = {Cham}
}
```

---

## 6. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

Klasyczny podręcznik uczenia statystycznego — szczegółowo omawia regresję liniową i logistyczną jako problemy optymalizacji (rozdziały 3 i 4), a także metryki jakości modeli (RMSE, dokładność, macierz pomyłek). Stanowi most między teorią optymalizacji a praktyką modelowania statystycznego.

```bibtex
@book{hastie2009elements,
  title     = {The Elements of Statistical Learning},
  author    = {Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome},
  year      = {2009},
  edition   = {2},
  publisher = {Springer},
  address   = {New York}
}
```

---

## 7. Ruder, S. (2016). An Overview of Gradient Descent Optimization Algorithms. *arXiv:1609.04747*.

Przystępny przegląd wariantów metody gradientu prostego (SGD, Momentum, Adagrad, RMSProp, Adam i inne), zawierający porównanie ich właściwości, pseudokody oraz intuicje geometryczne. Artykuł jest szczególnie użyteczny przy wyborze hiperparametrów i porównywaniu algorytmów adaptacyjnych.

```bibtex
@article{ruder2016overview,
  title   = {An Overview of Gradient Descent Optimization Algorithms},
  author  = {Ruder, Sebastian},
  journal = {arXiv preprint arXiv:1609.04747},
  year    = {2016}
}
```

---

## 8. McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall.

Fundamentalne źródło dla regresji logistycznej jako uogólnionego modelu liniowego (GLM). Opisuje estymację MLE (Maximum Likelihood Estimation) jako problem optymalizacji wypukłej oraz algorytm IRLS (Iteratively Reweighted Least Squares) będący szczególnym przypadkiem metody Newtona–Raphsona.

```bibtex
@book{mccullagh1989glm,
  title     = {Generalized Linear Models},
  author    = {McCullagh, Peter and Nelder, John A.},
  year      = {1989},
  edition   = {2},
  publisher = {Chapman and Hall},
  address   = {London}
}
```

---

## Zestawienie zbiorcze (BibTeX)

```bibtex
@book{boyd2004convex,
  title={Convex Optimization}, author={Boyd, Stephen and Vandenberghe, Lieven},
  year={2004}, publisher={Cambridge University Press}}

@book{nocedal2006numerical,
  title={Numerical Optimization}, author={Nocedal, Jorge and Wright, Stephen J.},
  year={2006}, edition={2}, publisher={Springer}}

@article{bottou2018optimization,
  title={Optimization Methods for Large-Scale Machine Learning},
  author={Bottou, L\'eon and Curtis, Frank E. and Nocedal, Jorge},
  journal={SIAM Review}, volume={60}, number={2}, pages={223--311}, year={2018}}

@inproceedings{kingma2015adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P. and Ba, Jimmy},
  booktitle={ICLR}, year={2015}}

@book{nesterov2018lectures,
  title={Lectures on Convex Optimization}, author={Nesterov, Yurii},
  year={2018}, edition={2}, publisher={Springer}}

@book{hastie2009elements,
  title={The Elements of Statistical Learning},
  author={Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome},
  year={2009}, edition={2}, publisher={Springer}}

@article{ruder2016overview,
  title={An Overview of Gradient Descent Optimization Algorithms},
  author={Ruder, Sebastian}, journal={arXiv preprint arXiv:1609.04747}, year={2016}}

@book{mccullagh1989glm,
  title={Generalized Linear Models}, author={McCullagh, Peter and Nelder, John A.},
  year={1989}, edition={2}, publisher={Chapman and Hall}}
```
