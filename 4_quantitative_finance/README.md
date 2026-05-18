# 4 — Finance quantitative

Applications directes au **marché** : données réelles, pricing d’options, volatilité implicite, orthogonalisation de facteurs, développements de Taylor et modèles de régression.

## Fichiers

| Script | Description |
|--------|-------------|
| `application_var_exp.py` | Rendements journaliers d’`AAPL` via `yfinance` — espérance et variance empiriques. |
| `bose_einstein.py` | Analogie Bose–Einstein pour l’allocation de capital (concentration sur les actifs « attractifs »). |
| `convergence_in_quantitative_finance.py` | Convergence en moyenne, en probabilité et presque sûre (estimateurs Monte Carlo). |
| `gram_shimt_process.py` | Gram–Schmidt sur des séries de rendements corrélées (facteurs orthogonaux). |
| `newton_raphson.py` | Méthode de Newton–Raphson pour la volatilité implicite (inversion Black–Scholes). |
| `taylor_series.py` | Approximation du prix d’un call par série de Taylor (Delta, Gamma). |
| `optimization_regression_model.py` | Pipeline de régression linéaire (`sklearn`) sur données synthétiques puis features finance. |

## Thèmes

- **Statistiques empiriques** sur données de marché
- **Pricing et sensibilités** (Greeks, Taylor, vol implicite)
- **Algèbre linéaire** pour décorréler des facteurs de risque
- **Asymptotique** des estimateurs de prix et de risque

## Lancer les scripts

```bash
python 4_quantitative_finance/application_var_exp.py
python 4_quantitative_finance/bose_einstein.py
python 4_quantitative_finance/convergence_in_quantitative_finance.py
python 4_quantitative_finance/gram_shimt_process.py
python 4_quantitative_finance/newton_raphson.py
python 4_quantitative_finance/taylor_series.py
python 4_quantitative_finance/optimization_regression_model.py
```

## Dépendances

`numpy`, `matplotlib`, `scipy`, `yfinance`, `scikit-learn`.
