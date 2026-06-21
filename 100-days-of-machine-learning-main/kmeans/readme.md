# K-Means Clustering

## What Is K-Means?

**K-Means** is an **unsupervised** clustering algorithm that partitions n data points into k clusters by iteratively assigning each point to its nearest centroid and updating centroids to the mean of assigned points.

K-Means is one of the most widely used clustering algorithms due to its simplicity and scalability.

---

## The Algorithm

**Input**: dataset X, number of clusters k

1. **Initialize**: randomly place k centroids (or use k-means++ initialization)
2. **Assignment step**: assign each point to the nearest centroid:
   ```
   label(xᵢ) = argmin_j ||xᵢ − μⱼ||²
   ```
3. **Update step**: move each centroid to the mean of its assigned points:
   ```
   μⱼ = (1/|Cⱼ|) × Σ xᵢ  for xᵢ ∈ Cⱼ
   ```
4. **Repeat** steps 2–3 until assignments stop changing (convergence) or `max_iter` is reached

---

## Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Scale features first — K-Means uses Euclidean distance, so scale matters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
km = KMeans(
    n_clusters=3,
    init='k-means++',   # Smart initialization
    n_init=10,          # Run 10 times with different initializations
    max_iter=300,
    random_state=42
)
km.fit(X_scaled)

# Results
labels = km.labels_        # Cluster assignment per sample
centers = km.cluster_centers_  # Centroid coordinates
inertia = km.inertia_      # Sum of squared distances to nearest centroid
```

---

## Choosing k: The Elbow Method

The **inertia** (within-cluster sum of squares) decreases as k increases — always. Find the "elbow" where adding more clusters gives diminishing returns:

```python
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
```

The elbow is where the curve bends — the rate of improvement slows significantly. If inertia drops steeply then flattens, the elbow is the optimal k.

---

## Choosing k: Silhouette Score

The silhouette score is a more rigorous metric that measures how similar a point is to its own cluster vs. other clusters:

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by k')
```

Silhouette score ranges from -1 to 1:
- Close to 1: well-clustered
- Close to 0: overlapping clusters
- Negative: misassigned points

Choose k with the **highest** silhouette score.

---

## k-means++ Initialization

Random initialization can converge to poor local minima. **k-means++** spreads initial centroids apart:

1. Choose first centroid randomly
2. For each subsequent centroid: choose a point with probability proportional to its squared distance from the nearest existing centroid
3. This ensures initial centroids are diverse

```python
# Always use init='k-means++' instead of init='random'
km = KMeans(n_clusters=3, init='k-means++', n_init=10)
```

`n_init=10` runs the algorithm 10 times with different initializations and keeps the best result.

---

## Visualizing Clusters (2D)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i in range(km.n_clusters):
    mask = labels == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=colors[i], label=f'Cluster {i}', alpha=0.6)

plt.scatter(centers[:, 0], centers[:, 1],
            c='black', marker='*', s=200, label='Centroids')
plt.legend()
plt.title('K-Means Clustering Results')
```

For data with more than 2 features, use PCA to reduce to 2D before plotting:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
```

---

## Cluster Profiling

After fitting, profile each cluster to understand what it represents:

```python
df['cluster'] = labels

# Compare feature means across clusters
cluster_profile = df.groupby('cluster').mean()
print(cluster_profile)

# How many samples per cluster
print(df['cluster'].value_counts())
```

This turns cluster labels into actionable insights: "Cluster 0 has high purchase frequency and low basket size → frequent small buyers."

---

## Limitations

| Limitation | Description | Alternative |
|-----------|-------------|-------------|
| Must specify k | No automatic k selection | DBSCAN, Hierarchical |
| Assumes spherical clusters | Fails on elongated/irregular shapes | DBSCAN, Gaussian Mixture |
| Sensitive to outliers | Outliers distort centroids | DBSCAN (outlier-robust) |
| Sensitive to scale | Large-scale features dominate | Always scale first |
| Local minima | Different initializations give different results | k-means++ + n_init=10 |
| Only works with numerical features | Can't handle categorical directly | Encode first, or use k-modes |

---

## Mini-Batch K-Means (For Large Datasets)

```python
from sklearn.cluster import MiniBatchKMeans

# Much faster than KMeans for large datasets
mbkm = MiniBatchKMeans(n_clusters=3, batch_size=1024, random_state=42)
mbkm.fit(X_scaled)
```

Processes small random batches instead of the full dataset per iteration — 10-100× faster with only slightly worse cluster quality.

---

## Practical Tips

- **Always scale features** with StandardScaler or MinMaxScaler before K-Means.
- Use the elbow method AND silhouette score together to choose k — they may disagree.
- Run with `n_init=10` to avoid local minima from bad initialization.
- K-Means is NOT suitable for: non-spherical clusters, clusters of very different sizes/densities, data with many outliers.
- Use `km.fit_predict(X)` instead of `km.fit(X); km.labels_` — they are equivalent but the former is more concise.
- For validation with ground-truth labels: use Adjusted Rand Index (ARI) — `sklearn.metrics.adjusted_rand_score(y_true, labels)`.
