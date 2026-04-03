import numpy as np
import matplotlib.pyplot as plt

# Gram-Schmidt Orthonormalization in Quantitative Finance
# -------------------------------------------------------
# Example: Orthonormalizing a set of return vectors to create uncorrelated portfolios.
# This is foundational in constructing orthogonal risk factors (e.g. PCA, factor models).

# Suppose we have 3 assets with different return time series (correlated)
np.random.seed(42)
n_days = 250
# Simulate 3 asset returns, correlated
mean_returns = [0.0005, 0.0003, 0.0004]
cov_matrix = [[0.0001, 0.00008, 0.00004],
              [0.00008, 0.00009, 0.00003],
              [0.00004, 0.00003, 0.00012]]
rets = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
A = rets.T  # shape (3, n_days), columns = vectors to orthonormalize

# Gram-Schmidt process:
def gram_schmidt(X):
    """Orthonormalize the rows of X using Gram-Schmidt."""
    Q = np.zeros_like(X)
    for i in range(X.shape[0]):
        qi = X[i]
        for j in range(i):
            # Remove projection onto previous q's
            proj = np.dot(qi, Q[j]) * Q[j]
            qi = qi - proj
        # Normalize
        Q[i] = qi / np.linalg.norm(qi)
    return Q

Q = gram_schmidt(A)

# Check orthogonality and norms
print("Dot products between orthonormalized vectors (should be close to identity matrix):")
dotp = np.dot(Q, Q.T)
print(np.round(dotp, 2))

# Compare before/after: correlations
corr_before = np.corrcoef(A)
corr_after = np.corrcoef(Q)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
im0 = ax[0].imshow(corr_before, vmin=-1, vmax=1, cmap='coolwarm')
ax[0].set_title("Correlations Before (assets)")
ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
fig.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(corr_after, vmin=-1, vmax=1, cmap='coolwarm')
ax[1].set_title("Correlations After (orthogonal)")
ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3))
fig.colorbar(im1, ax=ax[1])
plt.tight_layout()
plt.show()

# Demonstrate that Q's rows are now uncorrelated (orthonormal basis from original returns)
print("\nAfter Gram-Schmidt, the basis vectors can be interpreted as uncorrelated portfolio returns.\n"
      "This method is the backbone of principal component analysis and factor model construction in finance.")
