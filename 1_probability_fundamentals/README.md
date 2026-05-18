# 1 — Fondamentaux de probabilité

Scripts introductifs sur l’**intuition probabiliste** appliquée au trading : événements rares mais fréquents en combinaison, jeux de hasard, incertitude informationnelle et simulation par rejet.

## Fichiers

| Script | Description |
|--------|-------------|
| `combinatorics.py` | Dénombrement (permutations, arrangements, combinaisons), Bayes, espérance et variance. |
| `birthday_paradox.py` | Simulation Monte Carlo du paradoxe des anniversaires (probabilité de collision selon la taille du groupe). |
| `black_jack.py` | Simulation de mains de blackjack et estimation empirique des probabilités gain / perte / égalité. |
| `entropy_diversification.py` | Entropie de Shannon sur des scénarios de marché et des allocations de portefeuille. |
| `rejection_method` | Méthode du rejet pour échantillonner une loi tronquée (ex. densité risque-neutre, option digitale). |

## Concepts clés

- **Combinatoire** : compter les configurations possibles avant d’assigner des probabilités (portefeuilles, ordres, scénarios).
- **Paradoxe des anniversaires** : l’intuition sous-estime les coïncidences dans de grands ensembles (ticks, ordres, signaux).
- **Entropie** : mesure l’incertitude d’une loi ; lien avec la diversification (allocation concentrée vs équilibrée).
- **Rejet** : générer des tirages d’une loi cible via une proposition et un critère d’acceptation.

## Lancer les scripts

```bash
python 1_probability_fundamentals/combinatorics.py
python 1_probability_fundamentals/birthday_paradox.py
python 1_probability_fundamentals/black_jack.py
python 1_probability_fundamentals/entropy_diversification.py
python 1_probability_fundamentals/rejection_method
```

## Dépendances

`numpy`, `matplotlib`, `scipy` (pour `rejection_method`).
