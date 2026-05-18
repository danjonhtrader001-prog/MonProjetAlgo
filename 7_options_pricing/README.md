# 7 — Options pricing

**Module 7** du projet `quant-finance-probability`, après les fondamentaux de probabilité (1), les distributions (2), la mesure (3), la finance quantitative (4), le machine learning (5) et l’analyse numérique (6).

Module dédié au **pricing d’options** : formules fermées, arbres, Monte Carlo et modèles stochastiques de volatilité.

## Fichiers

| Fichier | Contenu |
|---------|---------|
| `1_fondamentaux.py` | Black–Scholes analytique, arbre binomial (CRR), parité put–call |
| `2_monte_carlo_options.py` | Monte Carlo : européenne, asiatique, put américain (LSM) |
| `3_volatilite_implicite_heston.py` | Surface de volatilité implicite, simulation du modèle de Heston |

## Exécution

Depuis la racine du projet :

```bash
pip install numpy matplotlib scipy
python 7_options_pricing/1_fondamentaux.py
python 7_options_pricing/2_monte_carlo_options.py
python 7_options_pricing/3_volatilite_implicite_heston.py
```

## Dépendances

`numpy`, `matplotlib`, `scipy`

## Parcours du projet

| # | Dossier |
|---|---------|
| 1 | `1_probability_fundamentals/` |
| 2 | `2_distributions_and_limits/` |
| 3 | `3_measure_and_inequalities/` |
| 4 | `4_quantitative_finance/` |
| 5 | `5_machine_learning/` |
| 6 | `6_numerical_analysis/` |
| 7 | `7_options_pricing/` *(ce module)* |
