import random
import matplotlib.pyplot as plt

def calcule_probabilites_anniversaire(taille_max=60, essais=1000):
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
    plt.plot(tailles, [p * 100 for p in probabilites], label="Probabilité (%)")
    plt.axhline(50, color='red', linestyle='--', label='Seuil 50%')
    plt.xlabel("Nombre de personnes dans la classe")
    plt.ylabel("Probabilité d'au moins 2 personnes\nayant le même anniversaire (%)")
    plt.title("Paradoxe des anniversaires : Probabilité en fonction de la taille de la classe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    