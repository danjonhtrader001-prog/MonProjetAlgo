/*
Linear Regression in Quantitative Finance: Full Pipeline Example
===============================================================

This script demonstrates building, training, testing, and validating a linear regression model
using synthetic data and then applies the workflow to a practical quantitative finance use case:
Predicting the next day's share price returns using two features (momentum and volatility).

The steps are:

    1. Prepare (simulate) data
    2. Split data into train and test sets
    3. Build and train a linear regression model
    4. Test (predict) on the holdout set
    5. Validate model with metrics and visualization
    6. Apply to a concrete quantitative finance case (share price prediction)

*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Simulate Data (Toy Example)
np.random.seed(42)
n_samples = 200

# Simulate features X1 and X2
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)

# True relationship: y = 2*X1 - 3*X2 + 5 + noise
y = 2 * X1 - 3 * X2 + 5 + np.random.normal(0, 1, n_samples)

# Stack features
X = np.vstack([X1, X2]).T

# 2. Split Data: Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# 3. Build and Train Linear Regression Model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("=== Linear Regression Training ===")
print("Intercept:", linreg.intercept_)
print("Coefficients:", linreg.coef_, "\n")

# 4. Test: Predict on Test Set
y_pred = linreg.predict(X_test)

# 5. Validate: Metrics and Visualization
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("=== Model Validation ===")
print("Test MSE:", mse)
print("Test R^2:", r2, "\n")

# Visualize predictions vs true
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.title("Linear Regression: True vs Predicted (Test Set)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()

# 6. Example in Quantitative Finance: Stock Price Prediction
np.random.seed(123)
n_days = 500
# Simulate "stock" prices (geometric Brownian motion)
S = np.zeros(n_days)
S[0] = 100
for t in range(1, n_days):
    S[t] = S[t-1] * np.exp(0.0005 + 0.01*np.random.randn())

# Create features:
#   - F1: 5-day return (momentum)
#   - F2: 5-day rolling volatility
returns = np.diff(S) / S[:-1]
returns = np.concatenate([[0], returns])
window = 5
momentum = np.array([np.sum(returns[max(0, i-window+1):i+1]) for i in range(n_days)])
volatility = np.array([np.std(returns[max(0, i-window+1):i+1]) for i in range(n_days)])

# The 'target' is the next-day return (shifted -1)
target = np.roll(returns, -1)
# Remove last row (no future value)
momentum = momentum[:-1]
volatility = volatility[:-1]
target = target[:-1]

# Build features matrix
X_finance = np.vstack([momentum, volatility]).T

# Train/test split on finance data
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_finance, target, test_size=0.2, shuffle=False)

# Train linear regression on financial data
linreg_finance = LinearRegression()
linreg_finance.fit(Xf_train, yf_train)
yf_pred = linreg_finance.predict(Xf_test)

print("=== Quantitative Finance Example ===")
print("Regression coefficients: (intercept, momentum, volatility)")
print(linreg_finance.intercept_, linreg_finance.coef_)
print("Out-of-sample R^2:", r2_score(yf_test, yf_pred))

plt.figure(figsize=(7,4))
plt.plot(yf_test, label='True returns')
plt.plot(yf_pred, label='Predicted returns")
plt.title("Predicted vs Actual Next-day Returns")
plt.legend()
plt.show()

/*
Summary:
----------
- We built, trained, and validated a linear regression both on synthetic and financial data.
- The concrete quantitative finance example uses technical factors (momentum/volatility)
  to predict next-day return.
- This workflow generalizes to any time series regression problem in quantitative strategies.
*/
# Use step: Gradient Descent Optimization appears here

# For demonstration, we'll implement a simple batch gradient descent to train a linear regression
# on the financial features (momentum, volatility) to predict next-day returns.

class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X_ = np.c_[np.ones(X.shape[0]), X]  # add intercept term
        self.theta = np.zeros(X_.shape[1])
        m = X_.shape[0]
        for _ in range(self.n_iter):
            grad = 2/m * X_.T @ (X_ @ self.theta - y)
            self.theta -= self.lr * grad
        return self

    def predict(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]
        return X_ @ self.theta

# Train/test split already done: Xf_train, Xf_test, yf_train, yf_test

gd_reg = LinearRegressionGD(lr=0.05, n_iter=2000)
gd_reg.fit(Xf_train, yf_train)
yf_pred_gd = gd_reg.predict(Xf_test)

print("=== Gradient Descent Linear Regression (Quantitative Finance) ===")
print("GD Regression coefficients (intercept, momentum, volatility):")
print(gd_reg.theta)
print("Out-of-sample GD R^2:", r2_score(yf_test, yf_pred_gd))
plt.figure(figsize=(7,4))
plt.plot(yf_test, label='True returns')
plt.plot(yf_pred_gd, label='GD Predicted returns")
plt.title("GD Predicted vs Actual Next-day Returns")
plt.legend()
plt.show()