# Analyse numérique pour ingénieurs (EPFL / Coursera)

Implémentations Python inspirées du MOOC [**Analyse numérique pour ingénieurs**](https://www.coursera.org/learn/analyse-numerique) (EPFL, Rappaz & Picasso).

Chaque script est autonome, exécutable et accompagné de visualisations lorsque c’est pertinent.

## Programme (ordre prévu)

| # | Fichier | Thème (cours EPFL) | Statut |
|---|---------|-------------------|--------|
| 01 | `01_lagrange_interpolation.py` | Interpolation de Lagrange | ✅ |
| 02 | `02_interpolation_par_morceaux.py` | Interpolation par morceaux (splines / hermite) | À venir |
| 03 | `03_differentiation_numerique.py` | Différences finies (dérivées 1ʳᵉ et 2ᵉ) | À venir |
| 04 | `04_integration_numerique.py` | Quadrature (trapèzes, Simpson, Gauss) | À venir |
| 05 | `05_systemes_lineaires.py` | Gauss, LU, Cholesky LLᵀ | À venir |
| 06 | `06_equations_non_lineaires.py` | Point fixe, Newton, systèmes non linéaires | À venir |
| 07 | `07_equations_differentielles.py` | EDO : schémas d’Euler, Runge–Kutta | À venir |
| 08 | `08_problemes_aux_limites_1d.py` | Problèmes aux limites unidimensionnels (différences finies) | À venir |

## Exécution

```bash
pip install numpy matplotlib
python numerical_analysis/01_lagrange_interpolation.py
```

## Lien avec la finance quantitative

L’interpolation apparaît en calibration de courbes (taux, volatilité), en lissage de données de marché et en approximation de fonctions de payoff pour le pricing numérique.
