import numpy as np
import matplotlib.pyplot as plt

# Demonstration of Taylor Series in Quantitative Finance:
# We'll use the Taylor series to approximate the price of a call option using the
# Black-Scholes formula around a reference point S0, i.e.,
# Price(S) ≈ Price(S0) + (d/dS)Price(S0)*(S-S0) + 0.5*(d^2/dS^2)Price(S0)*(S-S0)^2

from scipy.stats import norm

# Black-Scholes call price and its derivatives with respect to S (the 'Greeks')
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def call_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def call_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Parameters
S0 = 100  # Reference stock price
K = 100   # Strike price
T = 0.5   # Time to maturity (in years)
r = 0.01  # Risk-free rate
sigma = 0.2  # Volatility

# Taylor expansion around S0 (up to 2nd order, i.e., Gamma)
def taylor_call(S):
    c0 = black_scholes_call(S0, K, T, r, sigma)
    delta = call_delta(S0, K, T, r, sigma)
    gamma = call_gamma(S0, K, T, r, sigma)
    return c0 + delta * (S - S0) + 0.5 * gamma * (S - S0) ** 2

# Range of S for demonstration
S_range = np.linspace(70, 130, 200)
call_exact = black_scholes_call(S_range, K, T, r, sigma)
call_taylor = taylor_call(S_range)

plt.figure(figsize=(8, 5))
plt.plot(S_range, call_exact, label="Exact Black-Scholes Price", lw=2, color="navy")
plt.plot(S_range, call_taylor, '--', label="2nd Order Taylor (Delta+Gamma)", lw=2, color="darkorange")
plt.axvline(S0, color='grey', linestyle=':', linewidth=1)
plt.title("Taylor Series Approximation of Option Price (Black-Scholes)")
plt.xlabel("Stock Price S")
plt.ylabel("Call Option Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("This demonstrates how Taylor series (delta/gamma approximation) can be used to quickly estimate option prices\n"
      "for small moves in S, as done in options risk management or portfolio hedging.")

# --- Maclaurin Series for Black-Scholes Call Option (expand around S=0) ---

def taylor_call_maclaurin(S):
    # Expansion about S=0 (Maclaurin). For Black-Scholes, value at S=0 is 0,
    # d/dS of call price at S=0 is 0, and d^2/dS^2 is 0. 
    # All derivatives vanish, so the Maclaurin series (for S near 0) is nearly 0.
    # For educational purpose we return zeros of same shape.
    return np.zeros_like(S)

call_maclaurin = taylor_call_maclaurin(S_range)

plt.figure(figsize=(8, 5))
plt.plot(S_range, call_exact, label="Exact Black-Scholes Price", lw=2, color="navy")
plt.plot(S_range, call_taylor, '--', label="2nd Order Taylor (Delta+Gamma @ S₀)", lw=2, color="darkorange")
plt.plot(S_range, call_maclaurin, ':', label="Maclaurin Series (about S=0)", lw=2, color="green")
plt.axvline(S0, color='grey', linestyle=':', linewidth=1)
plt.title("Taylor vs. Maclaurin Series Approximation of Option Price")
plt.xlabel("Stock Price S")
plt.ylabel("Call Option Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nThis demonstrates how a Taylor (about S₀) approximation gives a very good local fit for small changes in S, "
      "while the Maclaurin series (about S=0) is not useful for practical option pricing since payoffs and sensitivities are zero there. "
      "Such visual contrast is instructive in understanding the importance of expansion point selection in series approximations.")