# Day 65 — Random Forest

## What Is Random Forest?

**Random Forest** is an **ensemble** of decision trees built using two sources of randomness:

1. **Bootstrap sampling** (Bagging): each tree is trained on a random subset of the training data (sampled with replacement)
2. **Random feature selection**: at each split, only a random subset of features is considered

The final prediction is made by **majority vote** (classification) or **averaging** (regression) across all trees.

---

## Why Randomness Makes It Better

A single decision tree is high-variance — it memorizes training data (overfit). By averaging many trees that are:
- Trained on different data samples (bootstrap)
- Making splits on different features (random feature subsets)

...the ensemble has much lower variance while maintaining low bias.

```
Bias-Variance: Random Forest reduces variance, bias stays low
Single tree: low bias, HIGH variance (overfit)
Random Forest: low bias, LOW variance (generalizes)
```

---

## Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Default: 100 trees
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',    # sqrt(n_features) features per split
    max_depth=None,         # Trees grow until pure
    min_samples_split=2,
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

For regression:
```python
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 100 | Number of trees — more is almost always better (up to diminishing returns) |
| `max_features` | 'sqrt' | Features per split: 'sqrt' for classification, 'log2' or float for regression |
| `max_depth` | None | Maximum depth of each tree — controls overfitting |
| `min_samples_split` | 2 | Minimum samples to split a node |
| `min_samples_leaf` | 1 | Minimum samples in a leaf node |
| `bootstrap` | True | Whether to use bootstrap sampling |
| `oob_score` | False | Use out-of-bag samples as a free validation set |

---

## Out-of-Bag (OOB) Score

Each tree is trained on ~63.2% of samples (bootstrap). The remaining ~36.8% are out-of-bag samples — a free validation set:

```python
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print("OOB Score:", rf_oob.oob_score_)
# OOB score ≈ cross-validation score without the CV cost
```

OOB score is computed without any held-out test set, making it useful for large datasets.

---

## Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Random Forest Feature Importances')
plt.xlabel('Mean Decrease in Impurity')
```

Importance = mean decrease in Gini impurity (or MSE for regression) contributed by each feature across all trees and all splits.

**Caution**: This measure is biased toward high-cardinality features. Use `permutation_importance` for unbiased estimates:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=X.columns)
perm_imp.sort_values().plot(kind='barh')
```

---

## Hyperparameter Tuning with RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['sqrt', 'log2', 0.3],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50, cv=5, scoring='accuracy',
    random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
```

---

## Random Forest vs. Single Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.3f}")
print(f"Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
# Random Forest is almost always significantly better
```

---

## Random Forest for Classification vs. Regression

| Task | Aggregation | Default max_features |
|------|-------------|---------------------|
| Classification | Majority vote | 'sqrt' |
| Regression | Mean | 1.0 (all features) |

For regression, `max_features=1.0` (use all features) is the default — set to `'sqrt'` or a fraction to increase randomness.

---

## Practical Tips

- Start with `n_estimators=100` — increase if test performance is still improving.
- Use `n_jobs=-1` to parallelize across all CPU cores.
- Enable `oob_score=True` for a free validation estimate.
- Feature importance from `feature_importances_` is fast but biased — use `permutation_importance` for critical decisions.
- Random Forest handles missing values poorly — impute before training.
- Random Forest is naturally robust to outliers and doesn't require feature scaling.
