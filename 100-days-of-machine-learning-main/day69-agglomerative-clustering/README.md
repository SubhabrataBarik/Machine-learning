# Agglomerative (Hierarchical) Clustering

Bottom-up hierarchical clustering applied to Mall Customer Segmentation data.

## Files
| File | Description |
|------|-------------|
| `agglomerative-clustering.ipynb` | Main notebook — dendrogram, model fitting, cluster visualisation |
| `hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv` | Mall customer dataset (200 rows, 5 columns) |

## Dataset Columns
| Column | Description |
|--------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall (1=low, 100=high spending) |

## How to Run
1. Open `agglomerative-clustering.ipynb` in VS Code / Jupyter
2. Select the project venv kernel
3. Run all cells (Shift+Enter)

## Key Concepts Covered
- Dendrogram reading — how to pick K visually
- Ward linkage — minimising within-cluster variance
- AgglomerativeClustering (sklearn)
- Customer segmentation use case

## sklearn Note
`affinity` parameter renamed to `metric` in sklearn 1.2 — this notebook uses `metric='euclidean'`.
