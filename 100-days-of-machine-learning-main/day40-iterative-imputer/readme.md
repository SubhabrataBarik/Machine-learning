# Day 40 — Iterative Imputer (MICE)

## What Is Iterative Imputation?

**Iterative Imputation** (also called MICE — Multiple Imputation by Chained Equations) fills missing values by modeling each feature with missing data as the **dependent variable**, using all other features as predictors. It iterates this process multiple times until the imputations converge.

---

## Dataset: 50 Startups (50_Startups.csv)

```python
df = pd.read_csv('50_Startups.csv')
```

Business dataset with R&D spend, Administration, Marketing spend, State, and Profit.

---

## How It Works (Step by Step)

```
Step 1: Initial fill — replace all NaNs with column means (initial guess)

Step 2: For each column with missing values:
    a. Treat it as the TARGET variable
    b. Treat all other columns (with NaNs filled) as FEATURES
    c. Train a regression/classification model
    d. Predict the missing values in this column
    e. Replace the NaN with predictions

Step 3: Repeat Step 2 for all missing columns

Step 4: Repeat Steps 2-3 for max_iter rounds (default: 10)

Step 5: Return the final imputed dataset
```

Each iteration improves on the previous — earlier columns benefit from better-imputed later columns.

---

## Implementation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=0
)

imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed  = imputer.transform(X_test)
```

**Note**: `IterativeImputer` is experimental in sklearn — you must import `enable_iterative_imputer` first.

---

## Estimator Options

The underlying regression model used for imputation can be changed:

```python
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

IterativeImputer(estimator=BayesianRidge())         # default, fast
IterativeImputer(estimator=LinearRegression())       # simple linear
IterativeImputer(estimator=RandomForestRegressor())  # captures non-linearity
IterativeImputer(estimator=ExtraTreesRegressor())    # faster than RF
```

`BayesianRidge` (default) is a good general-purpose choice. `RandomForestRegressor` can capture non-linear relationships but is slower.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `estimator` | `BayesianRidge()` | Model used for imputation |
| `max_iter` | 10 | Number of full imputation rounds |
| `tol` | 1e-3 | Convergence tolerance |
| `initial_strategy` | `'mean'` | Initial fill before iteration starts |
| `imputation_order` | `'ascending'` | Order of feature imputation |
| `random_state` | None | For reproducibility |

---

## Step-by-Step Notebook Demo

The notebook `step-by-step.ipynb` manually implements one round of MICE to show what happens:

```python
# Round 1: impute column A using B, C, D
# Round 1: impute column B using A (updated), C, D
# Round 1: impute column C using A (updated), B (updated), D
# Round 2: repeat with updated values
# ...
```

This reveals why iterative imputation is more accurate — each column benefits from the latest best estimates of all other columns.

---

## Iterative vs KNN vs Simple Imputation

| Method | Multivariate | Captures non-linearity | Speed | Best for |
|--------|-------------|----------------------|-------|----------|
| SimpleImputer (mean) | No | No | Fast | Low missing rate, MCAR |
| KNNImputer | Partially | No | Medium | Moderate missing, local patterns |
| IterativeImputer | Yes | With RF estimator | Slow | Complex multivariate patterns |

---

## Practical Tips

- IterativeImputer is more accurate than KNN for datasets where features are highly correlated with each other.
- Use `max_iter=10` (default) — convergence is usually reached early. Check with verbose output.
- Always set `random_state` for reproducible results.
- For large datasets (> 100k rows), IterativeImputer with a linear estimator is much faster than with RandomForest.
- Like all imputers: fit only on training data, transform both train and test.
