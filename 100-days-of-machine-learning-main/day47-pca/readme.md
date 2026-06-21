# Day 47 — Principal Component Analysis (PCA)

## What Is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms a dataset with many correlated features into a smaller set of **uncorrelated components** that capture most of the variance in the data.

Each principal component is a linear combination of the original features, ordered so that the first component captures the most variance, the second captures the next most, and so on.

---

## Why Use PCA?

- **Curse of dimensionality**: models perform poorly when features >> samples
- **Multicollinearity**: many features are correlated — PCA combines them
- **Visualization**: reduce to 2–3 components for plotting
- **Speed**: fewer features → faster training
- **Noise reduction**: low-variance components often capture noise

---

## Mathematics of PCA

```
1. Center the data: X_centered = X - mean(X)

2. Compute covariance matrix: C = (1/n) * X_centered.T @ X_centered

3. Eigendecompose: C = V @ Λ @ V.T
   - V: eigenvectors (principal components directions)
   - Λ: eigenvalues (variance captured by each component)

4. Sort by eigenvalue (descending)

5. Project: X_pca = X_centered @ V[:, :k]  (keep top k components)
```

---

## Step-by-Step Implementation from Scratch

```python
import numpy as np

# 1. Standardize
X_centered = X - X.mean(axis=0)

# 2. Covariance matrix
cov_matrix = np.cov(X_centered.T)

# 3. Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Sort by eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Explained variance ratio
explained = eigenvalues / eigenvalues.sum()

# 6. Project to k dimensions
k = 2
X_pca = X_centered @ eigenvectors[:, :k]
```

---

## sklearn PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Always scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
# [0.729, 0.228]  ← first 2 components explain 95.7% of variance
```

---

## Choosing the Number of Components

### Explained Variance Ratio

```python
pca = PCA()
pca.fit(X_scaled)

# Cumulative explained variance
import numpy as np
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(cumvar)
# Find n_components where cumvar >= 0.95 (95% of variance retained)
```

### Scree Plot

```python
import matplotlib.pyplot as plt

plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
```

Look for the "elbow" — the point where adding more components yields diminishing returns.

### Automatic Selection

```python
pca = PCA(n_components=0.95)  # keep enough components to explain 95% variance
X_pca = pca.fit_transform(X_scaled)
print(pca.n_components_)
```

---

## Interpreting Components

```python
# Loading matrix: each component's weights on original features
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)
```

High absolute loading = that original feature contributes strongly to that component.

---

## PCA for Visualization

```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA — 2D Projection')
```

---

## PCA Assumptions and Limitations

| Assumption | Description |
|------------|-------------|
| Linearity | PCA finds linear combinations — non-linear structures need kernel PCA or autoencoders |
| Scale sensitivity | Always standardize before PCA |
| Variance = Information | PCA assumes high-variance = high-information; this may not hold for all problems |
| Interpretability loss | Components are mixtures of original features — hard to interpret |

---

## Practical Tips

- **Always standardize** before PCA — without it, high-magnitude features dominate.
- Keep components explaining at least **90–95%** of total variance as a rule of thumb.
- PCA is unsupervised — it ignores class labels. For supervised tasks, try LDA (Linear Discriminant Analysis) instead.
- PCA in sklearn Pipeline: `Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('model', ...)])`.
- Use `pca.inverse_transform()` to reconstruct original features from reduced representation — useful for visualization and anomaly detection.
