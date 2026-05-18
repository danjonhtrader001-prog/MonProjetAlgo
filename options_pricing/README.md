# Options pricing

Module dédié au **pricing d’options** : formules fermées, arbres, Monte Carlo et modèles stochastiques de volatilité.

## Fichiers

| Fichier | Contenu |
|---------|---------|
| `1_fondamentaux.py` | Black–Scholes analytique, arbre binomial (CRR), parité put–call |
| `2_monte_carlo_options.py` | Monte Carlo : européenne, asiatique, put américain (LSM) |
| `3_volatilite_implicite_heston.py` | Surface de volatilité implicite, simulation du modèle de Heston |

## Exécution

```bash
pip install numpy matplotlib scipy
python options_pricing/1_fondamentaux.py
python options_pricing/2_monte_carlo_options.py
python options_pricing/3_volatilite_implicite_heston.py
```

## Dépendances

`numpy`, `matplotlib`, `scipy`
