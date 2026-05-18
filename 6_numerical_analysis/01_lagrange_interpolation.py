"""
01 — Interpolation de Lagrange
------------------------------
MOOC EPFL « Analyse numérique pour ingénieurs » — Module interpolation.

On reconstruit un polynôme P(x) de degré ≤ n-1 qui coïncide avec f nos aux nœuds
(x_i, y_i). La base de Lagrange L_i vérifie L_i(x_j) = δ_ij.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Fonction test : f(x) = (x+1)² — polynôme de degré 2."""
    return x**2 + 2 * x + 1


def creer_points(a, b, n):
    """Échantillonne n points équidistants de f sur [a, b]."""
    if n < 2:
        raise ValueError("Il faut au moins 2 points pour interpoler.")
    x_points = np.linspace(a, b, n)
    y_points = f(x_points)
    return x_points, y_points


def lagrange_basis(x, x_points, i):
    """L_i(x) = ∏_{k≠i} (x - x_k) / (x_i - x_k)"""
    L_i = 1.0
    for k in range(len(x_points)):
        if k != i:
            L_i *= (x - x_points[k]) / (x_points[i] - x_points[k])
    return L_i


def lagrange_interpolation(x, x_points, y_points):
    """P(x) = Σ_i y_i L_i(x)"""
    return sum(y_points[i] * lagrange_basis(x, x_points, i) for i in range(len(x_points)))


def lagrange_interpolation_vec(x_grid, x_points, y_points):
    """Évalue P sur un tableau numpy (pour les graphiques)."""
    return np.array([lagrange_interpolation(x, x_points, y_points) for x in x_grid])


if __name__ == "__main__":
    print("=" * 70)
    print("01 — Interpolation de Lagrange (EPFL / analyse numérique)")
    print("=" * 70)

    a, b = 0.0, 3.0
    n_points = 4
    x_data, y_data = creer_points(a, b, n_points)

    print(f"\nNœuds d'interpolation (n = {n_points}) sur [{a}, {b}] :")
    print(f"{'x':>8} {'f(x)':>12}")
    print("-" * 25)
    for x, y in zip(x_data, y_data):
        print(f"{x:8.2f} {y:12.2f}")

    test_points = [0.5, 1.0, 1.5, 2.0, 2.5]
    print("\nComparaison : f(x) vs P(x)")
    print(f"{'x':>8} {'f(x)':>15} {'P(x)':>15} {'|erreur|':>15}")
    print("-" * 60)

    max_err = 0.0
    for x_test in test_points:
        f_real = f(x_test)
        p_interp = lagrange_interpolation(x_test, x_data, y_data)
        err = abs(f_real - p_interp)
        max_err = max(max_err, err)
        print(f"{x_test:8.2f} {f_real:15.6f} {p_interp:15.6f} {err:15.2e}")

    print(f"\nErreur maximale aux points test : {max_err:.2e}")
    print("(Pour f polynomiale de degré ≤ n-1, l'erreur est nulle en arithmétique exacte.)")

    # Visualisation
    x_fine = np.linspace(a, b, 300)
    y_true = f(x_fine)
    y_interp = lagrange_interpolation_vec(x_fine, x_data, y_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_fine, y_true, "b-", lw=2, label="f(x) = (x+1)²")
    axes[0].plot(x_fine, y_interp, "r--", lw=2, label="P(x) Lagrange")
    axes[0].scatter(x_data, y_data, c="k", s=60, zorder=5, label="Nœuds")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Interpolation de Lagrange")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x_fine, np.abs(y_true - y_interp) + 1e-16, color="purple")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|f(x) - P(x)|")
    axes[1].set_title("Erreur d'interpolation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
