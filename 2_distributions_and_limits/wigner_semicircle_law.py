import numpy as np
import matplotlib.pyplot as plt

# Wigner semicircle law: useful in random matrix theory, in particular for eigenvalues of large symmetric matrices.
# Its density on [-R, R] is: p(x) = (2/(pi R^2)) * sqrt(R^2 - x^2)

R = 2  # Radius of the law’s support
n = 100000

# 1. Generate samples approximating the Wigner semicircle law
# We use a Wigner matrix (symmetric Gaussian matrix) and take its eigenvalues.

N = 400  # Size of the Wigner matrix
matrice = np.random.normal(0, 1, size=(N, N))
matrice = (matrice + matrice.T) / 2  # Symmetrization
valeurs_propres = np.linalg.eigvalsh(matrice)

# Normalization (the true semicircle law appears for large N and entries ~N(0,1/N))
valeurs_propres_normalisees = valeurs_propres / (2 * np.sqrt(N))

# 2. Plot empirical histogram (eigenvalue density)
plt.figure(figsize=(9, 5))
plt.hist(
    valeurs_propres_normalisees,
    bins=80,
    density=True,
    color='orchid',
    edgecolor='k',
    alpha=0.7,
    label='Histogram (eigenvalues)',
)

# 3. Plot theoretical semicircle density
x = np.linspace(-1, 1, 500)
densite = (2 / np.pi) * np.sqrt(1 - x**2)
densite[x < -1] = 0
densite[x > 1] = 0
plt.plot(x, densite, color='navy', lw=3, label='Theoretical density (Wigner)')

plt.title("Wigner semicircle law (spectrum of random matrices)")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
