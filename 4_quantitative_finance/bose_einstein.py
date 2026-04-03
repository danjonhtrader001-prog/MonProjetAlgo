import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_entreprises = 10
capital_total = 10000

# Arbitrary increasing attractiveness (fitness)
attractivites = np.linspace(1, 2, num_entreprises)  # e.g. from 1 to 2

# Energy = -log(attractiveness), more attractive = lower energy
energies = -np.log(attractivites)

# Arbitrary temperature (controls condensation)
T = 0.1

# Chemical potential mu approaches minimal energy (for BE condensation)
mu = np.min(energies) - 1e-5


# Bose–Einstein distribution for each company
def bose_einstein(energies, mu, T):
    return 1.0 / (np.exp((energies - mu) / T) - 1)


populations = bose_einstein(energies, mu, T)

# Normalize so the total equals capital_total
parts_capital = populations / np.sum(populations) * capital_total

# For display, round to integers
parts_capital = np.round(parts_capital).astype(int)

noms_entreprises = [f"Company {i+1}" for i in range(num_entreprises)]

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(noms_entreprises, parts_capital, color="skyblue")
plt.xlabel("Company")
plt.ylabel("Allocated capital")
plt.title("Bose–Einstein condensation: capital allocation across companies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("allocation_bose_einstein.png", bbox_inches="tight")
print("The chart has been saved in the current folder!")
plt.savefig("bose_einstein.png")
