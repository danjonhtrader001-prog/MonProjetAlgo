# 2 — Distributions et limites

Exploration des **lois de probabilité** discrètes et continues utilisées en finance, plus un exemple de **limite asymptotique** en théorie des matrices aléatoires (loi du demi-cercle de Wigner).

## Fichiers

| Script | Description |
|--------|-------------|
| `discrete_distributions.py` | Bernoulli, binomiale, Poisson, géométrique — histogrammes et liens trading (win rate, flux d’ordres). |
| `continuous_distributions.py` | Normale, exponentielle, uniforme, gamma, Student, Weibull, Laplace, Gumbel, Cauchy, etc. |
| `gaussian_tail_simulation.py` | Simulation de la partie positive d’une gaussienne (troncature / queue). |
| `wigner_semicircle_law.py` | Spectre d’une matrice de Wigner vs densité théorique du demi-cercle. |

## Lien avec le trading

| Loi | Usage typique |
|-----|----------------|
| Bernoulli / binomiale | Résultat d’un trade, nombre de gains sur N essais |
| Poisson | Arrivées d’ordres, pics dans le carnet |
| Géométrique | Nombre d’essais avant le premier succès |
| Normale / Laplace | Rendements « classiques » |
| Student / Cauchy | Queues épaisses, événements extrêmes |
| Wigner | Bruit vs facteurs dans les matrices de covariance |

## Lancer les scripts

```bash
python 2_distributions_and_limits/discrete_distributions.py
python 2_distributions_and_limits/continuous_distributions.py
python 2_distributions_and_limits/gaussian_tail_simulation.py
python 2_distributions_and_limits/wigner_semicircle_law.py
```

## Dépendances

`numpy`, `matplotlib`.
