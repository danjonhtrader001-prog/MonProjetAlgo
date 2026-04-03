import numpy as np
import matplotlib.pyplot as plt

# Bernoulli law (success/failure)
def bernoulli(p, n=10000):
    # Generate n Bernoulli(p) trials
    return np.random.binomial(1, p, n)
# Use in algo trading: models a binary trade outcome (win/loss), widely used to simulate strategies with success probability p.

# Binomial law (sum of n Bernoulli)
def binomiale(n_essais, p, n=10000):
    return np.random.binomial(n_essais, p, n)
# Use in algo trading: estimates the distribution of the number of winning trades over a series of independent trades.

# Poisson law (number of events in an interval, parameter lambda)
def poisson(lam, n=10000):
    return np.random.poisson(lam, n)
# Use in algo trading: simulates the random arrival of events such as orders or spikes in the order book (arrival process).

# Geometric law (number of failures before the first success)
def geometrique(p, n=10000):
    # numpy returns the number of trials until the first success, so we subtract 1
    return np.random.geometric(p, n) - 1
# Use in algo trading: estimates the number of trials (trades) needed before a winning trade, or the waiting time before a given event.

if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Bernoulli
    p_ber = 0.3
    data_ber = bernoulli(p_ber)
    axs[0, 0].hist(data_ber, bins=np.arange(-0.5, 2, 1), rwidth=0.6, color='c')
    axs[0, 0].set_xticks([0, 1])
    axs[0, 0].set_title(f"Bernoulli law (p={p_ber})")
    axs[0, 0].set_xlabel("Possible values")
    axs[0, 0].set_ylabel("Frequency")
    # Use: at each trade, models success/failure (1/0), useful to stress test the robustness of a strategy with a given win rate.

    # Binomial
    n_bin, p_bin = 10, 0.5
    data_bin = binomiale(n_bin, p_bin)
    axs[0, 1].hist(data_bin, bins=np.arange(-0.5, n_bin + 2, 1), rwidth=0.7, color='orange')
    axs[0, 1].set_xticks(range(n_bin + 1))
    axs[0, 1].set_title(f"Binomial law (n={n_bin}, p={p_bin})")
    axs[0, 1].set_xlabel("Number of successes")
    axs[0, 1].set_ylabel("Frequency")
    # Use: the binomial law gives the expected distribution of the number of successes over a fixed series of trades (backtests, finite‑horizon stress tests).

    # Poisson
    lam_poi = 3
    data_poi = poisson(lam_poi)
    axs[1, 0].hist(data_poi, bins=np.arange(-0.5, np.max(data_poi) + 1.5, 1), rwidth=0.7, color='g')
    axs[1, 0].set_xticks(range(0, min(15, np.max(data_poi) + 1)))
    axs[1, 0].set_title(f"Poisson law (lambda={lam_poi})")
    axs[1, 0].set_xlabel("Number of events")
    axs[1, 0].set_ylabel("Frequency")
    # Use: simulates the appearance of orders on markets (arrival process), modeling rare events (flash crashes, large orders).

    # Geometric
    p_geo = 0.4
    data_geo = geometrique(p_geo)
    axs[1, 1].hist(data_geo, bins=np.arange(-0.5, np.max(data_geo) + 1.5, 1), rwidth=0.7, color='purple')
    axs[1, 1].set_title(f"Geometric law (p={p_geo})")
    axs[1, 1].set_xlabel("Number of failures before success")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_xlim(-0.5, 12.5)
    # Use: waiting time before a success (for example, how many orders before getting a favorable fill, useful for execution algorithms).

    plt.tight_layout()
    plt.show()