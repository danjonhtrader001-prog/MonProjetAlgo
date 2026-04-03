import yfinance as yf
import numpy as np

# 1. Choose the stock and download one year of history
nom_action = "AAPL"  # Apple ticker symbol
donnees = yf.download(nom_action, period="1y")

# 2. Extract the daily closing prices
prix_fermeture = donnees["Close"]

# 3. Compute daily returns (percentage change from one day to the next)
# pct_change() computes: (Price_today - Price_yesterday) / Price_yesterday
# dropna() removes the first row because there is no previous day to compare with
rendements = prix_fermeture.pct_change().dropna()

# Check for missing values
if rendements.isnull().any():
    print("Warning: missing values detected in returns!")
else:
    print("All returns are correctly computed (no missing values).")

# 4. Core math: expectation and variance
# np.mean() computes the usual average
esperance = np.mean(rendements)

# np.var() computes the variance
variance = np.var(rendements)

# 5. Display results as percentages for readability
print(f"--- Analysis of stock {nom_action} over 1 year ---")
print(f"Expectation (average daily return): {esperance * 100:.4f} %")
print(f"Variance (daily risk)             : {variance * 10000:.4f}")
# We multiply the variance by 10,000 because it is a squared percentage!
