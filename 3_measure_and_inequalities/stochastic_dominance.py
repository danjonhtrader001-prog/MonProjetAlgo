import numpy as np
import matplotlib.pyplot as plt

# We simulate two random variables X and Y, where Y stochastically dominates X.
# Classic example: X ~ N(0, 1), Y ~ N(1, 1) (same variance but higher mean).
n = 10000
X = np.random.normal(0, 1, n)
Y = np.random.normal(1, 1, n)


# Empirical distribution functions F_X and F_Y
def F_empirique(data, x):
    return np.mean(data <= x)


x_vals = np.linspace(-4, 5, 500)
F_X = [F_empirique(X, xi) for xi in x_vals]
F_Y = [F_empirique(Y, xi) for xi in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, F_X, label="F_X(x) (X ~ N(0, 1))")
plt.plot(x_vals, F_Y, label="F_Y(x) (Y ~ N(1, 1))")
plt.title("Illustration of stochastic dominance:\nY dominates X (Y ≥_st X)")
plt.xlabel("x")
plt.ylabel("Distribution function")
plt.legend()
plt.grid(True)

# Optional: visualize densities for intuition
plt.figure(figsize=(10, 4))
plt.hist(X, bins=70, alpha=0.6, density=True, label="X", color="tab:blue")
plt.hist(Y, bins=70, alpha=0.6, density=True, label="Y", color="tab:orange")
plt.title("Empirical densities of X and Y")
plt.legend()
plt.grid(True)

plt.show()

print(
    "For every x, F_X(x) >= F_Y(x): Y dominates X in the sense of first‑order stochastic dominance."
)
