import numpy as np

# Backpropagation illustration:
# Simple neural network applied to a quantitative finance example: predicting option payoff

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Example: European call option payoff prediction [max(S-K, 0)] ---
# Input features: [normalized stock price S, normalized strike K]
# Network: 2 inputs -> 2 hidden neurons (sigmoid) -> 1 output (sigmoid, scaled)

# Simulate data (random option contracts, normalized)
np.random.seed(42)
n_samples = 1000
S = np.random.uniform(50, 150, n_samples)         # Simulate stock prices
K = np.random.uniform(50, 150, n_samples)         # Simulate strike prices

# Normalize data
S_norm = (S - 100) / 50
K_norm = (K - 100) / 50
X = np.stack([S_norm, K_norm], axis=1)            # Shape (n_samples, 2)

# True target: scaled call payoff (range roughly 0..1 for sigmoid output)
payoff = np.maximum(S - K, 0) / 100              # Scale payoff to be reasonable for sigmoid

y = payoff.reshape(-1, 1)

# --- Network parameters ---
W1 = np.random.randn(2, 2)
b1 = np.random.randn(2)
W2 = np.random.randn(2, 1)
b2 = np.random.randn(1)

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    loss_epoch = 0
    for i in range(n_samples):
        # --- Forward pass ---
        x = X[i]                             # (2,)
        target = y[i]                        # scalar

        z1 = np.dot(x, W1) + b1              # (2,)
        a1 = sigmoid(z1)                     # (2,)
        z2 = np.dot(a1, W2) + b2             # (1,)
        a2 = sigmoid(z2)                     # (1,)

        loss = 0.5 * (a2 - target) ** 2
        loss_epoch += loss

        # --- Backpropagation ---
        dloss_da2 = a2 - target             # shape (1,)
        da2_dz2 = sigmoid_deriv(z2)         # shape (1,)
        
        # Output layer
        dL_dz2 = dloss_da2 * da2_dz2        # shape (1,)
        dL_dW2 = np.outer(a1, dL_dz2)       # shape (2,1)
        dL_db2 = dL_dz2                     # shape (1,)
        
        # Backprop to hidden
        dL_da1 = W2 @ dL_dz2                # shape (2,)
        da1_dz1 = sigmoid_deriv(z1)         # shape (2,)
        dL_dz1 = dL_da1 * da1_dz1           # shape (2,)
        dL_dW1 = np.outer(x, dL_dz1)        # shape (2,2)
        dL_db1 = dL_dz1                     # shape (2,)

        # --- Update parameters ---
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1

    if (epoch + 1) % 200 == 0:
        avg_epoch_loss = loss_epoch.mean()
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

# --- Test: Predict payoff for some contracts ---
show_n = 8
print("\nTest: Neural net output vs. true option payoff (inputs scaled back):")
for i in range(show_n):
    s_act = S[i]
    k_act = K[i]
    x_input = np.array([(s_act - 100) / 50, (k_act - 100) / 50])
    z1 = np.dot(x_input, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    pred = sigmoid(z2)[0] * 100             # Scale output back
    true_payoff = np.maximum(s_act - k_act, 0)
    print(f"Stock={s_act:.1f}, Strike={k_act:.1f} | Predicted Payoff: {pred:.2f}, True Payoff: {true_payoff:.2f}")

print("\nThis illustrates backpropagation on a simple neural network to approximate option payoffs,\nan essential quantitative tool for pricing, hedging, or risk management in finance.")
