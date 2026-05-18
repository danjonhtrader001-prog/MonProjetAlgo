"""
1 — Fondamentaux du pricing d'options
-------------------------------------
- Black–Scholes (formule fermée call / put européen)
- Arbre binomial Cox–Ross–Rubinstein (CRR)
- Parité put–call : C - P = S - K·e^{-rT}
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black–Scholes analytique
# ---------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Prix européen sous Black–Scholes (sans dividendes).
    Pourquoi : référence analytique pour valider arbres et Monte Carlo.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Arbre binomial (CRR)
# ---------------------------------------------------------------------------

def binomial_tree_price(S, K, T, r, sigma, n_steps=100, option_type="call", american=False):
    """
    Pricing par arbre binomial CRR.
    Pourquoi : convergence vers BS quand n_steps augmente ; extension naturelle aux américaines.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    ST = S * u**j * d ** (n_steps - j)
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for i in range(n_steps - 1, -1, -1):
        continuation = discount * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            ST_i = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            if option_type == "call":
                exercise = np.maximum(ST_i - K, 0.0)
            else:
                exercise = np.maximum(K - ST_i, 0.0)
            values = np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])


# ---------------------------------------------------------------------------
# Parité put–call
# ---------------------------------------------------------------------------

def put_call_parity_check(S, K, T, r, sigma, tol=1e-10):
    """
    Vérifie C - P = S - K·e^{-rT}.
    Pourquoi : contrôle d'absence d'arbitrage entre implémentations call / put.
    """
    C = black_scholes(S, K, T, r, sigma, "call")
    P = black_scholes(S, K, T, r, sigma, "put")
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return lhs, rhs, abs(lhs - rhs) < tol


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    bs_call = black_scholes(S, K, T, r, sigma, "call")
    bs_put = black_scholes(S, K, T, r, sigma, "put")
    tree_call = binomial_tree_price(S, K, T, r, sigma, n_steps=200, option_type="call")
    tree_put = binomial_tree_price(S, K, T, r, sigma, n_steps=200, option_type="put")

    lhs, rhs, ok = put_call_parity_check(S, K, T, r, sigma)

    print("=== 1. FONDAMENTAUX — OPTIONS PRICING ===\n")
    print(f"Paramètres : S={S}, K={K}, T={T}, r={r}, σ={sigma}\n")
    print(f"Black–Scholes call : {bs_call:.6f}")
    print(f"Black–Scholes put  : {bs_put:.6f}")
    print(f"Arbre binomial call: {tree_call:.6f}")
    print(f"Arbre binomial put : {tree_put:.6f}")
    print(f"\nParité put–call : C-P = {lhs:.6f}, S-K·e^(-rT) = {rhs:.6f} → {'OK' if ok else 'ERREUR'}")

    # Convergence binomial → BS
    steps_grid = [5, 10, 20, 50, 100, 200, 500]
    tree_prices = [
        binomial_tree_price(S, K, T, r, sigma, n_steps=n, option_type="call")
        for n in steps_grid
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(steps_grid, tree_prices, "o-", label="Binomial call")
    axes[0].axhline(bs_call, color="crimson", ls="--", label="Black–Scholes")
    axes[0].set_xlabel("Nombre de pas (n)")
    axes[0].set_ylabel("Prix call")
    axes[0].set_title("Convergence de l'arbre binomial vers Black–Scholes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Courbe payoffs vs spot (maturité fixe)
    spots = np.linspace(60, 140, 200)
    calls_bs = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
    puts_bs = [black_scholes(s, K, T, r, sigma, "put") for s in spots]
    axes[1].plot(spots, calls_bs, label="Call BS", lw=2)
    axes[1].plot(spots, puts_bs, label="Put BS", lw=2)
    axes[1].axvline(K, color="gray", ls=":", label=f"Strike K={K}")
    axes[1].set_xlabel("Spot S")
    axes[1].set_ylabel("Prix")
    axes[1].set_title("Profils call / put (Black–Scholes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
