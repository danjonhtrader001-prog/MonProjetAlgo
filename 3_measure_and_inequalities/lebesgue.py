import numpy as np
import matplotlib.pyplot as plt

# Illustration of Lebesgue vs Riemann integration on a trading payoff (e.g., option)
# Example: we want to compute the expectation of a payoff f(x) = max(x-K, 0)
# where x ~ normal law (possible return), K = strike.

n = 100000
mu, sigma = 0, 1  # Normalized return (for simplicity), typical in finance
K = 0.5           # Strike price

# Generate random samples
x = np.random.normal(mu, sigma, n)
payoff = np.maximum(x - K, 0)

# 1. RIEMANN INTEGRAL (classical approximation by rectangle sum)
# We “cut” the x-axis into small bins
x_grid = np.linspace(np.min(x), np.max(x), 1000)
payoff_grid = np.maximum(x_grid - K, 0)
# Probability density on the x grid
from scipy.stats import norm
pdf = norm.pdf(x_grid, mu, sigma)
dx = x_grid[1] - x_grid[0]
esperance_riemann = np.sum(payoff_grid * pdf * dx)

# 2. LEBESGUE INTEGRAL (via Monte Carlo): expectation as the mean of sample payoffs
esperance_lebesgue = np.mean(payoff)

print("Theoretical payoff expectation (Riemann):", esperance_riemann)
print("Estimated expectation via Lebesgue integral (Monte Carlo):", esperance_lebesgue)

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(x, bins=200, density=True, color='lightgray', label="Simulated returns", alpha=0.7)
plt.plot(x_grid, pdf, color='k', label='Normal density')
plt.fill_between(
    x_grid,
    0,
    payoff_grid * pdf,
    color='skyblue',
    alpha=0.5,
    label="Area: Riemann integral",
)
plt.xlabel("Return x")
plt.ylabel("Density / Weighted payoff")
plt.title("Expectation of an option payoff: Lebesgue vs Riemann integral")
plt.legend()

# Additional: illustrate “Lebesgue sum” via contributing payoff samples
plt.scatter(
    x[x > K],
    np.zeros_like(x[x > K]),
    color='red',
    s=1,
    label="Contributing samples (Lebesgue view)",
    alpha=0.1,
)

plt.legend()
plt.show()

# Conclusion:
print(
    "\nThe Lebesgue integral (Monte Carlo) estimates the expectation as the average of observed payoffs,\n"
    "whereas the Riemann integral sums f(x)*p(x)*dx over the density. The two coincide in probability."
)