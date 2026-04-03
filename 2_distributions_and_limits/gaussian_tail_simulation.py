import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n = 10000                # number of samples
mu = 0                   # mean of the Gaussian
sigma = 1                # standard deviation

# Simulate a Gaussian-tail random variable on the half-line X > 0
# We take the positive part of a standard normal distribution

x = np.random.normal(mu, sigma, n)
queue_gaussienne = x[x > 0]  # keep only strictly positive values

# Plot histogram
plt.hist(queue_gaussienne, bins=50, density=True, color='coral', edgecolor='k')
plt.title("Simulation of a Gaussian-tail random variable (X > 0)")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
