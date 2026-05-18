"""
3 — Volatilité implicite et modèle de Heston
---------------------------------------------
- Surface de volatilité implicite (inversion Black–Scholes sur une grille strike × maturité)
- Simulation Monte Carlo du modèle stochastique de volatilité de Heston
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Black–Scholes (pour inversion de vol implicite)
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(market_price, S, K, T, r, vol_low=1e-4, vol_high=3.0):
    """
    Inversion numérique de la vol implicite (Brent).
    Pourquoi : les prix de marché sont souvent cotés en σ_imp, pas en prix absolu.
    """
    if market_price <= max(S - K * np.exp(-r * T), 0.0) + 1e-12:
        return vol_low
    objective = lambda sig: bs_call(S, K, T, r, sig) - market_price
    try:
        return brentq(objective, vol_low, vol_high)
    except ValueError:
        return np.nan


def build_vol_surface(S, r, strikes, maturities, true_vol_fn):
    """
    Construit une surface de vol implicite à partir de prix « marché » synthétiques.
    true_vol_fn(K, T) : smile de vol réel utilisé pour générer les prix.
    """
    n_k, n_t = len(strikes), len(maturities)
    market_prices = np.zeros((n_k, n_t))
    implied_vols = np.zeros((n_k, n_t))

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            sigma_true = true_vol_fn(K, T)
            market_prices[i, j] = bs_call(S, K, T, r, sigma_true)
            implied_vols[i, j] = implied_volatility(market_prices[i, j], S, K, T, r)

    return market_prices, implied_vols


# ---------------------------------------------------------------------------
# Modèle de Heston (Monte Carlo)
# ---------------------------------------------------------------------------

def simulate_heston(
    S0, v0, kappa, theta, xi, rho, r, T, n_steps, n_paths, seed=42
):
    """
    Schéma d'Euler pour Heston :
        dS = rS dt + sqrt(v) S dW1
        dv = kappa(theta - v) dt + xi sqrt(v) dW2
        corr(dW1, dW2) = rho

    Pourquoi : la vol constante de BS est insuffisante pour reproduire le smile ;
    Heston couple spot et variance stochastique.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = max(v0, 0.0)

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v_pos = np.maximum(v[:, t], 0.0)
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * W1
        )
        v[:, t + 1] = v_pos + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * W2
        v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)  # troncature Feller simplifiée

    return S, v


def heston_call_mc(S_paths, K, r, T):
    ST = S_paths[:, -1]
    payoffs = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * np.mean(payoffs)


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, r = 100.0, 0.05
    strikes = np.linspace(80, 120, 9)
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])

    # Smile synthétique : vol plus élevée pour les strikes éloignés du spot
    def smile_vol(K, T):
        moneyness = np.log(K / S)
        return 0.18 + 0.08 * moneyness**2 + 0.03 * np.sqrt(T)

    _, iv_surface = build_vol_surface(S, r, strikes, maturities, smile_vol)

    # Heston
    heston_params = dict(
        S0=S, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, r=r, T=1.0
    )
    K_heston = 100.0
    S_h, v_h = simulate_heston(
        n_steps=100, n_paths=30_000, **heston_params
    )
    heston_price = heston_call_mc(S_h, K_heston, r, heston_params["T"])
    bs_price_const = bs_call(S, K_heston, heston_params["T"], r, np.sqrt(heston_params["theta"]))

    print("=== 3. VOL IMPLICITE & HESTON ===\n")
    print("Surface de vol implicite (extrait, strikes 80–120) :")
    print(f"  σ_imp min = {np.nanmin(iv_surface):.3f} | max = {np.nanmax(iv_surface):.3f}\n")
    print(f"Heston call (MC, K={K_heston}) : {heston_price:.4f}")
    print(f"BS avec σ=√θ                : {bs_price_const:.4f}")

    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(
        K_grid, T_grid, iv_surface, cmap="viridis", edgecolor="none", alpha=0.9
    )
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturité T")
    ax1.set_zlabel("σ implicite")
    ax1.set_title("Surface de volatilité implicite")

    ax2 = fig.add_subplot(132)
    for j, T in enumerate(maturities):
        ax2.plot(strikes, iv_surface[:, j], "o-", label=f"T={T:.2f}a")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("σ implicite")
    ax2.set_title("Smile de vol par maturité")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133)
    t_axis = np.linspace(0, heston_params["T"], S_h.shape[1])
    for i in range(min(20, S_h.shape[0])):
        ax3.plot(t_axis, S_h[i], alpha=0.4, lw=0.8)
    ax3.set_xlabel("Temps")
    ax3.set_ylabel("Spot")
    ax3.set_title("Trajectoires spot — modèle de Heston")

    plt.tight_layout()
    plt.show()

    # Variance stochastique : quelques chemins
    fig2, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(15, v_h.shape[0])):
        ax.plot(t_axis, v_h[i], alpha=0.6)
    ax.axhline(heston_params["theta"], color="crimson", ls="--", label=f"θ = {heston_params['theta']}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Variance v(t)")
    ax.set_title("Chemins de variance — Heston")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
