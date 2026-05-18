"""
2 — Simulations Monte Carlo pour options
----------------------------------------
- Européenne : payoff terminal max(S_T - K, 0)
- Asiatique  : payoff sur la moyenne arithmétique des prix
- Américaine : algorithme de Longstaff–Schwartz (LSM) sur trajectoires discrètes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """
    Simule des trajectoires GBM risque-neutre.
    Pourquoi : moteur commun aux trois types d'options (même dynamique, payoffs différents).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_S[:, 1:] = log_S[:, 0:1] + np.cumsum(drift + diffusion, axis=1)
    return np.exp(log_S)


def mc_european_call(paths, K, r, T):
    """Monte Carlo standard : espérance actualisée du payoff terminal."""
    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * np.mean(payoffs), np.std(payoffs) / np.sqrt(len(payoffs))


def mc_asian_call(paths, K, r, T, average="arithmetic"):
    """
    Option asiatique : payoff sur la moyenne des prix (souvent plus lisse que le spot terminal).
    """
    if average == "arithmetic":
        avg = np.mean(paths, axis=1)
    else:
        avg = np.exp(np.mean(np.log(paths + 1e-12), axis=1))
    payoffs = np.maximum(avg - K, 0.0)
    return np.exp(-r * T) * np.mean(payoffs), np.std(payoffs) / np.sqrt(len(payoffs))


def mc_american_put_lsm(paths, K, r, T):
    """
    Longstaff–Schwartz (LSM) pour put américain.
    Pourquoi : le put américain a une prime d'exercice anticipé (contrairement au call sans dividendes).
    """
    n_paths, n_nodes = paths.shape
    dt = T / (n_nodes - 1)
    discount = np.exp(-r * dt)

    intrinsic = np.maximum(K - paths, 0.0)
    cashflows = intrinsic[:, -1].copy()

    for t in range(n_nodes - 2, 0, -1):
        itm = intrinsic[:, t] > 0
        if np.sum(itm) < 10:
            cashflows *= discount
            continue

        X = paths[itm, t]
        Y = cashflows[itm] * discount

        A = np.column_stack([np.ones_like(X), X, X**2])
        coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ coeffs

        exercise_now = intrinsic[itm, t] > continuation
        idx_itm = np.where(itm)[0]
        cashflows[idx_itm[exercise_now]] = intrinsic[idx_itm[exercise_now], t]
        cashflows[idx_itm[~exercise_now]] *= discount

    price = np.mean(cashflows) * discount
    stderr = np.std(cashflows) / np.sqrt(n_paths) * discount
    return price, stderr


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    n_steps, n_paths = 50, 50_000

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)

    euro_price, euro_se = mc_european_call(paths, K, r, T)
    asian_price, asian_se = mc_asian_call(paths, K, r, T)
    amer_price, amer_se = mc_american_put_lsm(paths, K, r, T)
    bs_call_ref = black_scholes_call(S0, K, T, r, sigma)
    bs_put_ref = black_scholes_put(S0, K, T, r, sigma)

    print("=== 2. MONTE CARLO — OPTIONS ===\n")
    print(f"Paramètres : S₀={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Trajectoires : {n_paths:,} | Pas : {n_steps}\n")
    print(f"Européenne (MC)  : {euro_price:.4f} ± {1.96*euro_se:.4f}")
    print(f"Asiatique (MC)   : {asian_price:.4f} ± {1.96*asian_se:.4f}")
    print(f"Américaine put (LSM): {amer_price:.4f} ± {1.96*amer_se:.4f}")
    print(f"Référence BS call    : {bs_call_ref:.4f}")
    print(f"Référence BS put     : {bs_put_ref:.4f} (européen, < put américain)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Quelques trajectoires
    t_grid = np.linspace(0, T, n_steps + 1)
    for i in range(min(30, n_paths)):
        axes[0].plot(t_grid, paths[i], alpha=0.4, lw=0.8)
    axes[0].axhline(K, color="crimson", ls="--", label=f"Strike K={K}")
    axes[0].set_title("Trajectoires GBM simulées")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Spot")
    axes[0].legend()

    # Comparaison des prix
    labels = ["Européenne", "Asiatique", "Put amér. (LSM)", "BS put"]
    prices = [euro_price, asian_price, amer_price, bs_put_ref]
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    axes[1].bar(labels, prices, color=colors, alpha=0.85)
    axes[1].set_ylabel("Prix call")
    axes[1].set_title("Comparaison des estimateurs Monte Carlo")
    axes[1].tick_params(axis="x", rotation=15)

    # Distribution des payoffs européens
    payoffs = np.maximum(paths[:, -1] - K, 0.0) * np.exp(-r * T)
    axes[2].hist(payoffs, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="k")
    axes[2].axvline(euro_price, color="crimson", lw=2, label=f"Moyenne = {euro_price:.3f}")
    axes[2].set_title("Distribution des payoffs actualisés (européenne)")
    axes[2].set_xlabel("Payoff")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
