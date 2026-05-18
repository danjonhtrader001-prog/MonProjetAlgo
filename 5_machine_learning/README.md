# 5 — Machine learning

Outils numériques pour l’**apprentissage** appliqué à la finance : réseaux de neurones, régression et calcul multivarié (Jacobien, Hessien).

## Fichiers

| Script | Description |
|--------|-------------|
| `backpropagation_nn.py` | Petit réseau (sigmoïde) entraîné par rétropropagation pour approximer un payoff d’option call. |
| `optimization_and_regression_model.py` | Régression linéaire simple et multiple (alpha/beta marché, features sectorielles). |
| `multivariate_calculus_for_macine_learning.py` | Jacobien et Hessien par différences finies (syntaxe JavaScript / `math.js` — à porter en Python si besoin). |

## Concepts clés

- **Backpropagation** : propagation du gradient à travers les couches pour minimiser l’erreur de prédiction.
- **Régression** : modéliser les rendements d’un actif en fonction de facteurs (marché, secteur).
- **Jacobien / Hessien** : sensibilité d’une sortie vectorielle ou courbure d’une perte scalaire (optimisation de second ordre).

## Lancer les scripts

```bash
python 5_machine_learning/backpropagation_nn.py
python 5_machine_learning/optimization_and_regression_model.py
# multivariate_calculus_for_macine_learning.py : exécuter dans un environnement Node/math.js
```

## Dépendances

- Python : `numpy`, `matplotlib`, `scikit-learn`
- `multivariate_calculus_for_macine_learning.py` : environnement JavaScript avec `math.js` (fichier hybride, non exécutable tel quel avec `python`)
