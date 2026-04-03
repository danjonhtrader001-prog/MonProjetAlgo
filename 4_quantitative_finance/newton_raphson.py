import numpy as np
import matplotlib.pyplot as plt

"""
Newton-Raphson Method (Newton's Method)
----------------------------------------
The Newton-Raphson method is an efficient algorithm for finding roots of a real-valued function f(x).
Given an initial guess x0, it iteratively improves the solution using:

    x_{n+1} = x_n - f(x_n) / f'(x_n)

where f'(x) is the derivative of f(x).

This is widely used across science and engineering to solve nonlinear equations.

-----------------------------
Application in Quantitative Finance:
-----------------------------
One classic use is for finding the "implied volatility" of an option. Given an option market price,
we invert the Black-Scholes formula to find the volatility parameter that makes the Black-Scholes
theoretical price match the observed market price.

This requires solving for `sigma` in the equation:
    BlackScholesPrice(S, K, T, r, sigma) = MarketPrice

which cannot be done algebraically and thus Newton's method is frequently used.

Let's demonstrate:
"""

from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def vega(S, K, T, r, sigma):
    """Derivative of Black-Scholes price with respect to volatility (vega)."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def newton_raphson(f, df, x0, tol=1e-7, max_iter=100):
    """Generic Newton-Raphson root solver."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            print(f"Zero derivative. No solution found at iteration {i}.")
            return None
        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print("Did not converge.")
    return None

# Quantitative finance example: Find implied volatility from market call price
S = 100      # stock price
K = 100      # strike price
T = 0.5      # time to maturity (years)
r = 0.01     # risk-free rate
true_sigma = 0.2

# Suppose the observed market price is for sigma = 0.2:
market_price = black_scholes_call(S, K, T, r, true_sigma)

# Now, suppose you only observe market_price and want to estimate implied volatility:
def objective(sigma):
    # The function whose root is the implied volatility: Black-Scholes price - market price
    return black_scholes_call(S, K, T, r, sigma) - market_price

def derivative(sigma):
    # The "Vega", derivative wrt sigma
    return vega(S, K, T, r, sigma)

guess = 0.3  # initial guess (can be far from true sigma)
implied_vol = newton_raphson(objective, derivative, guess)

print(f"\nTrue volatility: {true_sigma}")
print(f"Implied volatility recovered by Newton-Raphson: {implied_vol:.6f}")

# Visual demonstration: convergence
sigmas = [guess]
x = guess
for i in range(8):
    fx = objective(x)
    dfx = derivative(x)
    x_new = x - fx/dfx
    sigmas.append(x_new)
    if abs(x_new - x) < 1e-7:
        break
    x = x_new

plt.figure(figsize=(7, 4))
sigma_grid = np.linspace(0.05, 0.4, 100)
bs_prices = [black_scholes_call(S, K, T, r, sigma) for sigma in sigma_grid]
plt.plot(sigma_grid, bs_prices, label="Call Price vs Volatility")
plt.axhline(market_price, color='k', linestyle='--', label='Market Price')
plt.scatter(sigmas, [black_scholes_call(S, K, T, r, s) for s in sigmas], color='r', zorder=5, label="Newton Steps")
plt.title("Newton-Raphson Method for Implied Volatility")
plt.xlabel("Volatility (sigma)")
plt.ylabel("Call Option Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("""
Summary:
--------
The Newton-Raphson method quickly finds the implied volatility that matches the observed market price under Black-Scholes.
Implied vol is a key concept in options trading, volatility surface construction, and risk management in finance.
""")