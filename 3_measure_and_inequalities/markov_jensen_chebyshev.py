import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Illustration of Markov, Jensen, and Chebyshev inequalities
# in an algo trading context: modeling a random payoff (e.g., PnL).
# ==========================

np.random.seed(42)

n = 20000                # number of simulations
mu, sigma = 0.5, 1.2     # mean and standard deviation of PnL
payoff = np.random.lognormal(mean=mu, sigma=sigma, size=n)  # strictly positive PnL (log-normal = exponentiated return)

# === 1. Markov inequality ===
# P(X >= a) <= E[X] / a for a > 0
a = np.quantile(payoff, 0.95)  # 95th percentile: a “high PnL threshold”
proba_empirique = np.mean(payoff >= a)
maj_markov = np.mean(payoff) / a

print("--- MARKOV INEQUALITY ---")
print(f"P(PnL >= a={a:.2f}) (empirical): {proba_empirique:.4f}")
print(f"Markov upper bound           : {maj_markov:.4f}")
print(f"Expectation E[PnL]           : {np.mean(payoff):.4f}")
print()

# === 2. Jensen inequality ===
# f(E[X]) <= E[f(X)] for convex f and random variable X
# Example: f(x) = exp(x) (useful for compounded earnings or growth)
rendement = np.random.normal(0.05, 0.2, n)  # normalized portfolio log-return
exp_esp = np.exp(np.mean(rendement))
esp_exp = np.mean(np.exp(rendement))
print("--- JENSEN INEQUALITY ---")
print(f"exp(E[X])              = {exp_esp:.4f}")
print(f"E[exp(X)] (empirical)  = {esp_exp:.4f}")
print("Jensen:", "exp(E[X]) <=", "E[exp(X)] ?", exp_esp <= esp_exp)
print()

# === 3. Chebyshev inequality ===
# P(|X - mu| >= k*sigma) <= 1/k²
k = 2
mu_payoff = np.mean(payoff)
sigma_payoff = np.std(payoff)
proba_dev = np.mean(np.abs(payoff - mu_payoff) >= k * sigma_payoff)
maj_tcheb = 1 / k**2
print("--- CHEBYSHEV INEQUALITY ---")
print(f"P(|PnL - mu| >= {k}*sigma) (empirical): {proba_dev:.4f}")
print(f"Chebyshev upper bound                 : {maj_tcheb:.4f}")
print()

# === Visualization: Markov and Chebyshev ===
plt.figure(figsize=(12, 5))
plt.hist(payoff, bins=120, density=True, color='skyblue', alpha=0.7, label="PnL distribution")
plt.axvline(a, color='red', linestyle='--', label=f"Threshold a (95th pct) = {a:.2f}")
plt.fill_betweenx(
    [0, plt.gca().get_ylim()[1]],
    a,
    plt.gca().get_xlim()[1],
    color='salmon',
    alpha=0.25,
    label="Region PnL > a (rare event)",
)
plt.title("Markov inequality on a strategy’s PnL")
plt.legend()
plt.xlabel("Simulated PnL")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Jensen (simple rendering)
plt.figure(figsize=(7, 4))
plt.hist(rendement, bins=80, density=True, alpha=0.5, color='goldenrod', label="Returns (X)")
plt.axvline(np.mean(rendement), color="darkgreen", linestyle="--", lw=2, label="E[X]")
plt.axhline(exp_esp, color="navy", linestyle=":", lw=2, label="exp(E[X])")
plt.axhline(esp_exp, color="red", linestyle=":", lw=2, label="E[exp(X)]")
plt.title("Jensen visualization for exp(X)")
plt.legend()
plt.xlabel("Return X")
plt.tight_layout()
plt.show()

# Visualization: Chebyshev
plt.figure(figsize=(12, 5))
plt.hist(payoff, bins=120, density=True, color='turquoise', alpha=0.7, label="PnL distribution")
plt.axvline(mu_payoff, color='purple', linestyle='--', label="Mean (μ)")
plt.axvline(mu_payoff + k * sigma_payoff, color='crimson', linestyle='--', label=f"μ+{k}σ")
plt.axvline(mu_payoff - k * sigma_payoff, color='crimson', linestyle='--', label=f"μ-{k}σ")
plt.title("Extreme region for Chebyshev inequality")
plt.xlabel("Simulated PnL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
