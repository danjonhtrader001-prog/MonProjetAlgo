import numpy as np
import matplotlib.pyplot as plt


# Why this script exists:
# In quantitative finance, we rely on asymptotic arguments all the time
# (Monte Carlo pricing, risk estimators, backtesting metrics).
# The three convergences below answer different practical questions:
# - L1 / mean convergence: is the average pricing error going to zero?
# - convergence in probability: is the estimator likely to be close?
# - almost sure convergence: does the simulation path itself stabilize?


def mean_convergence_demo(sample_sizes):
    """
    Why this model:
    We price an option via Monte Carlo and compare the estimator with the
    theoretical expectation. If E[|X_n - X|] -> 0, then average absolute
    pricing error vanishes, which is a strong guarantee for budgeting model risk.
    """
    true_price = np.exp(0.5)  # E[e^Z] with Z ~ N(0,1)
    errors = []

    for n in sample_sizes:
        z = np.random.normal(0.0, 1.0, n)
        estimator = np.mean(np.exp(z))
        errors.append(abs(estimator - true_price))

    return np.array(errors), true_price


def probability_convergence_demo(sample_sizes, epsilon=0.1, experiments=1000):
    """
    Why this model:
    We repeat the same estimation many times to measure:
    P(|X_n - X| > epsilon). In risk terms, this is the chance that an
    estimator misses a tolerance band.
    """
    true_price = np.exp(0.5)
    exceedance_probabilities = []

    for n in sample_sizes:
        z = np.random.normal(0.0, 1.0, (experiments, n))
        estimators = np.mean(np.exp(z), axis=1)
        exceedance = np.abs(estimators - true_price) > epsilon
        exceedance_probabilities.append(np.mean(exceedance))

    return np.array(exceedance_probabilities), true_price


def almost_sure_convergence_demo(n_max=5000):
    """
    Why this model:
    We build one single path of cumulative average of i.i.d. returns.
    By the strong law of large numbers, this path converges almost surely
    to the true mean. This mirrors practitioners checking if running
    simulation averages stabilize over time.
    """
    returns = np.random.normal(loc=0.01, scale=0.2, size=n_max)
    running_average = np.cumsum(returns) / np.arange(1, n_max + 1)
    true_mean = 0.01
    return running_average, true_mean


def main():
    np.random.seed(42)
    sample_sizes = np.array([50, 100, 250, 500, 1000, 2500, 5000])

    mean_errors, mean_true = mean_convergence_demo(sample_sizes)
    exceedance_probs, prob_true = probability_convergence_demo(sample_sizes, epsilon=0.1)
    running_avg, as_true = almost_sure_convergence_demo(n_max=5000)

    print("=== Convergence in Quantitative Finance ===")
    print(f"Theoretical target for pricing examples: {mean_true:.6f}")
    print("Mean convergence proxy |X_n - X|:", np.round(mean_errors, 6))
    print("Probability convergence proxy P(|X_n - X| > 0.1):", np.round(exceedance_probs, 6))
    print(f"Almost sure example target mean return: {as_true:.4f}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(sample_sizes, mean_errors, marker="o", color="tab:blue")
    axs[0].set_title("Convergence en moyenne (L1)")
    axs[0].set_xlabel("Nombre d'echantillons n")
    axs[0].set_ylabel("|X_n - X|")
    axs[0].grid(alpha=0.3)

    axs[1].plot(sample_sizes, exceedance_probs, marker="o", color="tab:orange")
    axs[1].set_title("Convergence en probabilite")
    axs[1].set_xlabel("Nombre d'echantillons n")
    axs[1].set_ylabel("P(|X_n - X| > epsilon)")
    axs[1].grid(alpha=0.3)

    axs[2].plot(np.arange(1, len(running_avg) + 1), running_avg, color="tab:green")
    axs[2].axhline(as_true, linestyle="--", color="black", label="Moyenne vraie")
    axs[2].set_title("Convergence presque sure")
    axs[2].set_xlabel("Temps / iteration n")
    axs[2].set_ylabel("Moyenne empirique cumulative")
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
