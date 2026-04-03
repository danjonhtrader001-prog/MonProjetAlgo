import random
import matplotlib.pyplot as plt


def calcule_probabilites_anniversaire(taille_max=60, essais=1000):
    """Compute Monte Carlo estimates of the birthday paradox probability.

    For each class size between 1 and taille_max, run `essais` simulations of random birthdays
    and estimate the probability that at least two people share the same birthday.
    """
    tailles = []
    probabilites = []
    for effectif in range(1, taille_max + 1):
        coincidences = 0
        for _ in range(essais):
            anniversaires = [random.randint(1, 365) for _ in range(effectif)]
            if len(anniversaires) != len(set(anniversaires)):
                coincidences += 1
        proba = coincidences / essais
        tailles.append(effectif)
        probabilites.append(proba)
    return tailles, probabilites


if __name__ == "__main__":
    tailles, probabilites = calcule_probabilites_anniversaire()
    plt.figure(figsize=(10, 6))
    plt.plot(tailles, [p * 100 for p in probabilites], label="Probability (%)")
    plt.axhline(50, color='red', linestyle='--', label='50% threshold')
    plt.xlabel("Number of people in the group")
    plt.ylabel("Probability that at least 2 people\nshare the same birthday (%)")
    plt.title("Birthday paradox: probability as a function of group size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()