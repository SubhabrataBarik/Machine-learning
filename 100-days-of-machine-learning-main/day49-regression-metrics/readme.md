# Day 49 — Regression Metrics

## Overview

Regression metrics quantify how well a regression model's predictions match the actual values. Choosing the right metric depends on the problem — specifically, how you want to penalize large errors and whether scale matters.

---

## Dataset: Placement Data (placement.csv)

```python
df = pd.read_csv('placement.csv')
```

---

## 1. Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ |y_actual - y_predicted|
```

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

- Average of absolute differences between predictions and actuals
- **In the same units as y** (easy to interpret)
- **Robust to outliers** — large errors are not disproportionately penalized
- When to use: when outlier errors are acceptable and you want an interpretable average error

---

## 2. Mean Squared Error (MSE)

```
MSE = (1/n) × Σ (y_actual - y_predicted)²
```

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

- Squares the errors — large errors are penalized much more than small ones
- **Not in the same units as y** (squared units) — harder to interpret directly
- **Sensitive to outliers** — one large error dominates the metric
- The loss function minimized during linear regression training
- When to use: when large errors are unacceptable (e.g., financial predictions)

---

## 3. Root Mean Squared Error (RMSE)

```
RMSE = √MSE
```

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# or
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

- Square root of MSE → **back in the same units as y**
- Most commonly reported metric for regression
- Still sensitive to outliers (inherited from MSE)
- When to use: when you want a metric in the original units but still care about penalizing large errors

---

## 4. R² (Coefficient of Determination)

```
R² = 1 - (SS_residual / SS_total)
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

- Measures what fraction of variance in y is explained by the model
- **Scale-independent** — 0 to 1 (can be negative for very bad models)
- R² = 0.85 means the model explains 85% of the variance
- Baseline: R² = 0 → predicting the mean; R² = 1 → perfect fit
- When to use: for comparing models across different datasets (scale-free)

---

## 5. Adjusted R²

```
Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)
```

Where n = number of samples, p = number of features.

- Penalizes adding features that don't improve the model
- R² always increases when you add more features (even random ones)
- Adjusted R² decreases if a new feature adds less than chance would
- When to use: for comparing models with different numbers of features

---

## 6. Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) × Σ |y - ŷ| / |y|
```

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
```

- Percentage error — unit-free
- MAPE = 5% means predictions are off by 5% on average
- **Cannot handle y = 0** (division by zero)
- When to use: business reporting where percentage errors are meaningful

---

## Comparison Summary

| Metric | Units | Outlier-sensitive | Scale-free | Interpretable |
|--------|-------|------------------|-----------|---------------|
| MAE | Same as y | No | No | High |
| MSE | y² | Yes | No | Low |
| RMSE | Same as y | Yes | No | Medium |
| R² | None | Moderate | Yes | Medium |
| MAPE | % | No | Yes | High |

---

## Choosing the Right Metric

| Scenario | Recommended Metric |
|----------|--------------------|
| Outliers are not a concern | MAE |
| Outliers must be heavily penalized | MSE or RMSE |
| Comparing models on different scales | R² |
| Business reporting | MAPE |
| Feature selection across models | Adjusted R² |

---

## Practical Tips

- Always report multiple metrics — a model can look good on R² but terrible on MAE.
- RMSE and MAE together tell you if outlier errors are distorting the average.
- If RMSE >> MAE, there are large individual errors (outliers).
- Never evaluate on training data — always use held-out test data or cross-validation.
- For cross-validation: `cross_val_score(model, X, y, scoring='neg_mean_squared_error')`.
