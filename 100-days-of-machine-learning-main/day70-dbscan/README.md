# DBSCAN — Density-Based Spatial Clustering

Density-based clustering that finds arbitrarily-shaped clusters and automatically detects outliers.

## Files
| File | Description |
|------|-------------|
| `dbscan_demo.ipynb` | Main notebook — tiny demo + concentric circles demo |

## Key Concepts Covered
- Core points, border points, noise points (label = -1)
- `eps` and `min_samples` hyperparameters
- Why K-Means fails on non-spherical data
- DBSCAN correctly clustering concentric rings
- Effect of `eps` on clustering results

## How to Run
1. Open `dbscan_demo.ipynb` in VS Code / Jupyter
2. Select the project venv kernel
3. Run all cells (Shift+Enter)

## DBSCAN vs K-Means at a Glance
| | K-Means | DBSCAN |
|--|---------|--------|
| Specify K upfront | Yes | No |
| Handles outliers | No | Yes (-1 label) |
| Cluster shape | Spherical | Arbitrary |
| Deterministic | No (random init) | Yes |
