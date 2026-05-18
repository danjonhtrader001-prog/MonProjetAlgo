"""
Combinatoire et probabilités de base
------------------------------------
Dénombrement (permutations, arrangements, combinaisons), conditionnement,
formule de Bayes, espérance et variance — avec exemples orientés finance / trading.
"""

import math
from itertools import combinations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. DÉNOMBREMENT
# =============================================================================

def factorial(n: int) -> int:
    """n! — nombre de permutations de n objets distincts."""
    return math.factorial(n)


def permutation(n: int, k: Optional[int] = None) -> int:
    """
    Permutations de k éléments parmi n (ordre compte, sans répétition).
    P(n,k) = n! / (n-k)!  ; si k is None, k = n (toutes les permutations).
    """
    if k is None:
        k = n
    if not (0 <= k <= n):
        raise ValueError("Il faut 0 <= k <= n")
    return math.perm(n, k)


def arrangement(n: int, k: int) -> int:
    """
    Arrangement (synonyme ici de k-permutation) : A(n,k) = n! / (n-k)!.
    Ex. finance : ordre de passage de k événements parmi n scénarios distincts.
    """
    return permutation(n, k)


def combination(n: int, k: int) -> int:
    """
    Combinaison : C(n,k) = n! / (k! (n-k)!).
    Ex. finance : choisir k actifs dans un univers de n (sans ordre).
    """
    if not (0 <= k <= n):
        raise ValueError("Il faut 0 <= k <= n")
    return math.comb(n, k)


def permutation_with_repetition(n: int, k: int) -> int:
    """k tirages parmi n types, ordre compte, avec remise : n^k."""
    return n**k


def combination_with_repetition(n: int, k: int) -> int:
    """
    Combinaisons avec répétition (étoiles et barres) : C(n+k-1, k).
    Ex. répartir k unités de capital sur n classes d'actifs.
    """
    return math.comb(n + k - 1, k)


# =============================================================================
# 2. PROBABILITÉS, CONDITIONNEMENT, BAYES
# =============================================================================

def probability(event_count: int, total: int) -> float:
    """Probabilité laplacienne (équiprobabilité) : |A| / |Ω|."""
    if total <= 0:
        raise ValueError("L'univers doit avoir une taille strictement positive.")
    return event_count / total


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """P(A|B) = P(A ∩ B) / P(B)."""
    if p_b == 0:
        raise ValueError("P(B) ne peut pas être nul.")
    return p_a_and_b / p_b


def bayes(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """
    P(A|B) = P(B|A) P(A) / P(B).
    Pourquoi : mettre à jour une croyance (prior) après observation d'un signal (B).
    """
    if p_b == 0:
        raise ValueError("P(B) ne peut pas être nul.")
    return p_b_given_a * p_a / p_b


def total_probability(probabilities: list[float], conditionals: list[float]) -> float:
    """
    Formule des probabilités totales : P(B) = Σ_i P(B|A_i) P(A_i).
    """
    if len(probabilities) != len(conditionals):
        raise ValueError("Les listes doivent avoir la même longueur.")
    return sum(p * q for p, q in zip(probabilities, conditionals))


# =============================================================================
# 3. ESPÉRANCE, VARIANCE, COVARIANCE
# =============================================================================

def expectation(values: np.ndarray, probabilities: np.ndarray) -> float:
    """E[X] = Σ x_i p_i pour une loi discrète."""
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Les probabilités doivent sommer à 1.")
    return float(np.dot(values, probabilities))


def variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    """Var(X) = E[X²] - E[X]²."""
    mu = expectation(values, probabilities)
    return float(np.dot(values**2, probabilities) - mu**2)


def standard_deviation(values: np.ndarray, probabilities: np.ndarray) -> float:
    return math.sqrt(variance(values, probabilities))


def covariance(
    x: np.ndarray, y: np.ndarray, probabilities: np.ndarray
) -> float:
    """Cov(X,Y) = E[XY] - E[X]E[Y]."""
    mu_x = expectation(x, probabilities)
    mu_y = expectation(y, probabilities)
    return float(np.dot(x * y, probabilities) - mu_x * mu_y)


# =============================================================================
# DÉMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("COMBINATOIRE & PROBABILITÉS DE BASE")
    print("=" * 72)

    # --- Dénombrement ---
    n, k = 10, 3
    print("\n--- 1. DÉNOMBREMENT ---")
    print(f"Univers de taille n = {n}, choix k = {k}")
    print(f"  Factorielle n!           : {factorial(n):,}")
    print(f"  Permutation P(n,k)       : {permutation(n, k):,}")
    print(f"  Arrangement A(n,k)       : {arrangement(n, k):,}")
    print(f"  Combinaison C(n,k)       : {combination(n, k):,}")
    print(f"  Avec répétition (ordre)  : {permutation_with_repetition(n, k):,}")
    print(f"  Avec répétition (sans ordre): {combination_with_repetition(n, k):,}")

    # Vérification itertools
    sample_comb = list(combinations(["A", "B", "C", "D"], 2))
    print(f"\n  itertools combinations('ABCD', 2) → {len(sample_comb)} = C(4,2)")

    # --- Bayes : signal de trading ---
    print("\n--- 2. CONDITIONNEMENT & BAYES ---")
    print("Exemple : un indicateur détecte une hausse (signal +).")
    p_hausse = 0.40  # P(H) — probabilité de hausse du marché
    p_signal_given_hausse = 0.70  # P(S+|H) — sensibilité
    p_signal_given_baisse = 0.25  # P(S+|¬H) — faux positifs

    p_signal = total_probability(
        [p_hausse, 1 - p_hausse],
        [p_signal_given_hausse, p_signal_given_baisse],
    )
    p_hausse_given_signal = bayes(p_signal_given_hausse, p_hausse, p_signal)

    print(f"  P(hausse)                    = {p_hausse:.2f}")
    print(f"  P(signal + | hausse)         = {p_signal_given_hausse:.2f}")
    print(f"  P(signal + | baisse)         = {p_signal_given_baisse:.2f}")
    print(f"  P(signal +) [prob. totale]   = {p_signal:.4f}")
    print(f"  P(hausse | signal +) [Bayes] = {p_hausse_given_signal:.4f}")
    print("  → Le signal révise la probabilité de hausse (posterior).")

    # --- Espérance / variance : PnL discret ---
    print("\n--- 3. ESPÉRANCE & VARIANCE ---")
    pnl = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])  # gains/pertes (%)
    probs = np.array([0.10, 0.20, 0.30, 0.25, 0.15])

    mu = expectation(pnl, probs)
    var = variance(pnl, probs)
    sigma = standard_deviation(pnl, probs)

    print(f"  PnL possibles (%) : {pnl}")
    print(f"  Probabilités      : {probs}")
    print(f"  E[PnL]            = {mu:.4f} %")
    print(f"  Var(PnL)          = {var:.4f}")
    print(f"  σ(PnL)            = {sigma:.4f} %")

    # Portefeuille à 2 actifs corrélés (covariance)
    r1 = np.array([0.02, -0.01])
    r2 = np.array([0.01, 0.03])
    p_joint = np.array([0.6, 0.4])
    cov_12 = covariance(r1, r2, p_joint)
    print(f"\n  Cov(R1, R2) sur scénario joint = {cov_12:.6f}")

    # --- Graphiques ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # C(n,k) en fonction de k
    n_fix = 12
    ks = np.arange(0, n_fix + 1)
    comb_vals = [combination(n_fix, int(k)) for k in ks]
    axes[0, 0].bar(ks, comb_vals, color="steelblue", edgecolor="k")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel(f"C({n_fix}, k)")
    axes[0, 0].set_title(f"Combinaisons : choisir k actifs parmi {n_fix}")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Bayes : posterior vs prior selon qualité du signal
    sensibilities = np.linspace(0.5, 0.95, 50)
    posteriors = [
        bayes(s, p_hausse, total_probability([p_hausse, 1 - p_hausse], [s, p_signal_given_baisse]))
        for s in sensibilities
    ]
    axes[0, 1].plot(sensibilities, posteriors, "darkorange", lw=2)
    axes[0, 1].axhline(p_hausse, color="gray", ls="--", label=f"Prior P(H)={p_hausse}")
    axes[0, 1].set_xlabel("P(signal + | hausse)")
    axes[0, 1].set_ylabel("P(hausse | signal +)")
    axes[0, 1].set_title("Bayes : impact de la qualité du signal")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Loi discrète du PnL
    axes[1, 0].bar(pnl.astype(str), probs, color="seagreen", edgecolor="k")
    axes[1, 0].axvline(mu, color="crimson", ls="--", lw=2, label=f"E[PnL]={mu:.2f}%")
    axes[1, 0].set_xlabel("PnL (%)")
    axes[1, 0].set_ylabel("Probabilité")
    axes[1, 0].set_title("Distribution discrète du PnL")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Variance vs probabilité du scénario extrême (renormalisation des autres cas)
    base_probs = np.array([0.10, 0.20, 0.30, 0.25, 0.15])
    p_extreme = np.linspace(0.01, 0.35, 40)
    variances = []
    for p in p_extreme:
        pr = base_probs.copy()
        scale = (1.0 - p) / (1.0 - base_probs[0])
        pr[0] = p
        pr[1:] = base_probs[1:] * scale
        variances.append(variance(pnl, pr))
    axes[1, 1].plot(p_extreme, variances, color="purple", lw=2)
    axes[1, 1].set_xlabel("P(scénario extrême négatif)")
    axes[1, 1].set_ylabel("Var(PnL)")
    axes[1, 1].set_title("Sensibilité du risque à la queue de distribution")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Combinatoire & probabilités — fondamentaux", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 72)
