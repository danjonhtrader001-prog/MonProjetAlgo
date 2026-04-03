import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Illustration : Entropie de Shannon appliquée à la répartition des probabilités d'états de marché
# (ex : probabilités d'évolution d'un prix selon un modèle de diffusion, ou d'allocation sur des classes d'actifs)
# ==========================

def shannon_entropy(probas):
    """Calcule l'entropie de Shannon (bits) pour une loi de probabilité discrète."""
    probas = np.array(probas)
    # Enlever les 0 pour éviter les erreurs de log(0)
    probas = probas[probas > 0]
    return -np.sum(probas * np.log2(probas))

# Ex 1 : Entropie d'un marché fortement directionnel (quasi-certain de monter)
proba_monte = [0.98, 0.01, 0.01]  # [monte, stable, baisse]
H_directionnel = shannon_entropy(proba_monte)

# Ex 2 : Marché incertain (incertitude maximale, equiprobabilité)
proba_equilibre = [1/3, 1/3, 1/3]
H_max = shannon_entropy(proba_equilibre)

# Ex 3 : Marché ayant une probabilité intermédiaire
proba_asym = [0.6, 0.2, 0.2]
H_intermediaire = shannon_entropy(proba_asym)

print("--- ENTROPIE ET INCERTITUDE EN FINANCE ---")
print(f"Entropie marché directionnel (proba {proba_monte}): {H_directionnel:.3f} bits")
print(f"Entropie marché très incertain (proba {proba_equilibre}): {H_max:.3f} bits")
print(f"Entropie marché intermédiaire (proba {proba_asym}): {H_intermediaire:.3f} bits")
print()

# Visualisation Entropie selon divers scénarios (ex : allocation sur 3 classes d'actifs)
labels = ['Directionnel', 'Incertain', 'Asymétrique']
entropies = [H_directionnel, H_max, H_intermediaire]

plt.figure(figsize=(7,5))
plt.bar(labels, entropies, color=['crimson', 'navy', 'orange'])
plt.title("Entropie de la distribution des probabilités de scénarios de marché")
plt.ylabel("Entropie de Shannon (bits)")
plt.ylim(0, np.log2(3)+0.3)
for i, ent in enumerate(entropies):
    plt.text(i, ent + 0.05, f"{ent:.2f}", ha='center', va='bottom', fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()

# Application pratique : Diversification et entropie d'un portefeuille
# Plus l'allocation est "équilibrée", plus l'entropie est grande. (Diversification max = entropie max)
portfolios = [
    [1.0, 0.0, 0.0, 0.0],   # Tout sur un actif
    [0.5, 0.5, 0.0, 0.0],   # Deux actifs à égalité
    [0.25, 0.25, 0.25, 0.25], # Parfaitement diversifié (max entropie)
    [0.8, 0.1, 0.05, 0.05], # Concentré mais pas extrême
]

H_portfolios = [shannon_entropy(w) for w in portfolios]
labels_portfolios = [
    "Alloc. extrême",
    "2 actifs égaux",
    "Diversifiée",
    "Concentrée"
]

plt.figure(figsize=(7,5))
plt.bar(labels_portfolios, H_portfolios, color='seagreen')
plt.title("Entropie (diversification) de portefeuilles simulés")
plt.ylabel("Entropie (bits)")
plt.ylim(0, np.log2(4)+0.2)
for i, ent in enumerate(H_portfolios):
    plt.text(i, ent + 0.03, f"{ent:.2f}", ha='center', va='bottom', fontsize=11)
plt.grid(axis='y', alpha=0.15)
plt.tight_layout()
plt.show()

print("--- INTERPRETATION ---")
print("Plus l'entropie est faible, plus l'incertitude est faible : signal de prix fort ou portefeuille concentré.")
print("Plus l'entropie est élevée (~max log2(n)), plus l'incertitude (ou diversification) est grande.")
print("L'entropie aide à quantifier l'information, la diversification, ou l'incertitude sur une distribution.")