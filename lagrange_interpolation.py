"""
Interpolation de Lagrange sur une fonction continue
"""

# ÉTAPE 1: Définir la fonction continue qu'on veut interpoler
def f(x):
    """
    Fonction continue qu'on veut interpoler
    f(x) = x^2 + 2x + 1 = (x+1)^2
    """
    return x**2 + 2*x + 1


# ÉTAPE 2: Créer des points d'échantillonnage sur la fonction
def creer_points(a, b, n):
    """
    Crée n points espacés uniformément entre a et b sur la fonction f(x)
    
    Args:
        a: borne inférieure
        b: borne supérieure
        n: nombre de points
    
    Returns:
        x_points: liste des x
        y_points: liste des f(x)
    """
    x_points = []
    y_points = []
    
    # Créer les x espacés uniformément
    for i in range(n):
        x = a + (b - a) * i / (n - 1)
        x_points.append(x)
        y_points.append(f(x))
    
    return x_points, y_points


# ÉTAPE 3: Coder la base de Lagrange
def lagrange_basis(x, x_points, i):
    """
    Calcule L_i(x) = ∏(k≠i) [(x - x_k) / (x_i - x_k)]
    """
    L_i = 1.0
    for k in range(len(x_points)):
        if k != i:
            L_i *= (x - x_points[k]) / (x_points[i] - x_points[k])
    return L_i


# ÉTAPE 4: Coder l'interpolation de Lagrange complète
def lagrange_interpolation(x, x_points, y_points):
    """
    Calcule P(x) = Σ y_i * L_i(x)
    """
    P = 0.0
    for i in range(len(x_points)):
        P += y_points[i] * lagrange_basis(x, x_points, i)
    return P


# ÉTAPE 5: Tester sur la fonction continue
if __name__ == "__main__":
    print("=" * 70)
    print("Interpolation de Lagrange sur une fonction continue")
    print("=" * 70)
    
    # Créer 4 points d'échantillonnage entre x=0 et x=3
    a, b = 0.0, 3.0
    n_points = 4
    
    x_data, y_data = creer_points(a, b, n_points)
    
    print(f"\nPoints d'échantillonnage (n = {n_points}):")
    print(f"{'x':>8} {'f(x)':>12}")
    print("-" * 25)
    for x, y in zip(x_data, y_data):
        print(f"{x:8.2f} {y:12.2f}")
    
    # Tester l'interpolation à des points intermédiaires
    print(f"\n\nComparaison: Vraie fonction vs Interpolation de Lagrange")
    print(f"{'x':>8} {'f(x) réelle':>15} {'P(x) interp':>15} {'Erreur':>15}")
    print("-" * 60)
    
    # Tester à plusieurs points
    test_points = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for x_test in test_points:
        f_real = f(x_test)
        p_interp = lagrange_interpolation(x_test, x_data, y_data)
        error = abs(f_real - p_interp)
        
        print(f"{x_test:8.2f} {f_real:15.6f} {p_interp:15.6f} {error:15.2e}")
    
    print("\n" + "=" * 70)
