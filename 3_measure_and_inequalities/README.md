# 3 — Mesure et inégalités

Pont entre **calcul intégral**, **espérance** et **bornes probabilistes** utiles pour borner les risques de queue sans connaître la loi exacte du PnL.

## Fichiers

| Script | Description |
|--------|-------------|
| `lebesgue.py` | Espérance d’un payoff `max(X - K, 0)` : intégrale de Riemann sur une grille vs Monte Carlo (vue Lebesgue). |
| `markov_jensen_chebyshev.py` | Inégalités de Markov, Jensen et Tchebychev sur PnL log-normal et rendements. |
| `stochastic_dominance.py` | Dominance stochastique du premier ordre entre deux lois normales (fonctions de répartition empiriques). |

## Concepts clés

- **Lebesgue vs Riemann** : l’espérance en probabilité correspond à l’intégrale par rapport à la mesure ; le Monte Carlo en est l’estimateur naturel.
- **Markov** : borne \( \mathbb{P}(X \geq a) \leq \mathbb{E}[X]/a \) pour \( X \geq 0 \).
- **Jensen** : pour \( f \) convexe, \( f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)] \) (croissance composée).
- **Tchebychev** : borne les écarts à la moyenne en fonction de la variance.
- **Dominance stochastique** : comparer des stratégies via les fonctions de répartition (préférence pour « plus »).

## Lancer les scripts

```bash
python 3_measure_and_inequalities/lebesgue.py
python 3_measure_and_inequalities/markov_jensen_chebyshev.py
python 3_measure_and_inequalities/stochastic_dominance.py
```

## Dépendances

`numpy`, `matplotlib`, `scipy` (pour `lebesgue.py`).
