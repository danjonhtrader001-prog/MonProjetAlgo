import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n = 10000

# Normal (Gaussian) law
# Used to model daily returns under a “normal market” hypothesis.
mu_normale, sigma_normale = 0, 1
data_normale = np.random.normal(mu_normale, sigma_normale, n)

# Exponential law
# Can model waiting times between two successive events (e.g., between two trades).
lambda_exp = 1
data_exponentielle = np.random.exponential(1 / lambda_exp, n)  # numpy uses scale = 1/lambda

# Uniform law
# Simple model for basic randomness and for random draws in backtests.
a_uniforme, b_uniforme = 0, 1
data_uniforme = np.random.uniform(a_uniforme, b_uniforme, n)

# Gamma law
# Models accumulation processes of risk, useful for time until multiple incidents in operational risk.
shape_gamma, scale_gamma = 2, 2
data_gamma = np.random.gamma(shape_gamma, scale_gamma, n)

# Student’s t law
# Captures “fat tails” (extreme events), more realistic than the normal law for extreme returns.
df_student = 5
data_student = np.random.standard_t(df_student, n)

# Weibull law
# Used to model lifetime of phenomena (time to crash, trend persistence).
a_weibull = 1.5  # shape
data_weibull = np.random.weibull(a_weibull, n)

# Laplace (double exponential) law
# Good model for returns with sharp central peak and heavy tails.
mu_laplace, b_laplace = 0, 1
data_laplace = np.random.laplace(mu_laplace, b_laplace, n)

# Gumbel law
# Useful for modeling extreme values (maxima or minima), e.g., max drawdown, extreme VaR.
mu_gumbel, beta_gumbel = 0, 1
data_gumbel = np.random.gumbel(mu_gumbel, beta_gumbel, n)

# Cauchy law
# Illustrates very erratic extreme returns with infinite variance (e.g., flash crash).
loc_cauchy, scale_cauchy = 0, 1
data_cauchy = np.random.standard_cauchy(n) * scale_cauchy + loc_cauchy

# Plotting the distributions
fig, axs = plt.subplots(3, 3, figsize=(18, 15))

# 1 - Normal
axs[0, 0].hist(data_normale, bins=50, density=True, color='skyblue', edgecolor='k')
axs[0, 0].set_title("Normal\n(standard returns)")

# 2 - Exponential
axs[0, 1].hist(data_exponentielle, bins=50, density=True, color='orange', edgecolor='k')
axs[0, 1].set_title("Exponential\n(waiting time between events)")

# 3 - Uniform
axs[0, 2].hist(data_uniforme, bins=50, density=True, color='green', edgecolor='k')
axs[0, 2].set_title("Uniform\n(pure randomness)")

# 4 - Gamma
axs[1, 0].hist(data_gamma, bins=50, density=True, color='purple', edgecolor='k')
axs[1, 0].set_title("Gamma\n(risk accumulation)")

# 5 - Student
axs[1, 1].hist(data_student, bins=50, density=True, color='red', edgecolor='k')
axs[1, 1].set_title("Student\n(fat tails, extreme events)")

# 6 - Weibull
axs[1, 2].hist(data_weibull, bins=50, density=True, color='deepskyblue', edgecolor='k')
axs[1, 2].set_title("Weibull\n(trend/crash lifetime)")

# 7 - Laplace
axs[2, 0].hist(data_laplace, bins=50, density=True, color='gold', edgecolor='k')
axs[2, 0].set_title("Laplace\n(heavy tails, sharp peak)")

# 8 - Gumbel
axs[2, 1].hist(data_gumbel, bins=50, density=True, color='brown', edgecolor='k')
axs[2, 1].set_title("Gumbel\n(extreme values, VaR)")

# 9 - Cauchy
axs[2, 2].hist(data_cauchy, bins=100, density=True, color='pink', edgecolor='k', range=(-10, 10))
axs[2, 2].set_title("Cauchy\n(highly erratic markets)")

for ax in axs.flat:
    ax.set_xlabel("x")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
