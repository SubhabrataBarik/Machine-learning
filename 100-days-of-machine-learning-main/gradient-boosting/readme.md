# Gradient Boosting

## What Is Gradient Boosting?

**Gradient Boosting** is an ensemble technique that builds trees **sequentially**, where each new tree corrects the **residual errors** of all previous trees. It applies gradient descent in **function space** — instead of updating model parameters, it fits new trees to the negative gradient of the loss function.

---

## Core Idea: Fit Residuals Iteratively

```
Initial prediction: F₀(x) = mean(y)  [constant]

Round 1: Compute residuals = y - F₀(x)
         Fit tree h₁ to residuals
         F₁(x) = F₀(x) + lr × h₁(x)

Round 2: Compute residuals = y - F₁(x)
         Fit tree h₂ to residuals
         F₂(x) = F₁(x) + lr × h₂(x)

...repeat for n_estimators rounds
```

Each tree makes a small improvement. The learning rate `lr` controls each tree's contribution — small lr + many trees → better generalization.

---

## Implementation with sklearn

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score

# Classification
gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    random_state=42
)
gbc.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, gbc.predict(X_test)))

# Regression
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbr.fit(X_train, y_train)
print("R²:", r2_score(y_test, gbr.predict(X_test)))
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 100 | Number of boosting rounds — more = potentially better, slower |
| `learning_rate` | 0.1 | Shrinks each tree's contribution — coupled with n_estimators |
| `max_depth` | 3 | Maximum depth of each tree — controls model complexity |
| `subsample` | 1.0 | Fraction of samples per tree — `< 1.0` adds randomness (Stochastic GBM) |
| `min_samples_split` | 2 | Minimum samples to split a node |
| `max_features` | None | Features per split — adding randomness like Random Forest |
| `loss` | 'log_loss' (clf) | Loss function: 'log_loss', 'exponential', 'squared_error' |

---

## Learning Rate vs. n_estimators

These two are inversely linked:
```python
# Equivalent models (approximately):
GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)
```

Lower learning rate + more estimators → better generalization but slower training. The right approach is to use a small learning rate and find the optimal n_estimators with early stopping.

---

## Early Stopping

```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    validation_fraction=0.1,   # hold-out 10% for early stopping
    n_iter_no_change=10,       # stop if no improvement for 10 rounds
    random_state=42
)
gbc.fit(X_train, y_train)
print("Best n_estimators:", gbc.n_estimators_)
```

---

## Staged Predictions (Learning Curve)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

staged_losses = [
    log_loss(y_test, y_prob)
    for y_prob in gbc.staged_predict_proba(X_test)
]

plt.plot(staged_losses)
plt.xlabel('Number of trees')
plt.ylabel('Log Loss')
plt.title('Gradient Boosting: Test Loss by n_estimators')
```

Find the elbow — where test loss stops improving — and set `n_estimators` there.

---

## Stochastic Gradient Boosting

Setting `subsample < 1.0` randomly samples a fraction of training data per tree:

```python
gbc_stochastic = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,   # use 80% of samples per tree
    random_state=42
)
```

Benefits:
- Adds randomness → better generalization
- Faster training (fewer samples per tree)
- More robust to outliers

---

## XGBoost, LightGBM, and CatBoost

sklearn's GradientBoosting is the reference implementation. For production, use:

| Library | Key advantage |
|---------|--------------|
| XGBoost | Regularized, fast, GPU support |
| LightGBM | Fastest, leaf-wise growth (vs. level-wise) |
| CatBoost | Native categorical feature handling |

```python
# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# LightGBM
import lightgbm as lgb
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
```

---

## Feature Importance

```python
import pandas as pd
importances = pd.Series(gbc.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Gradient Boosting Feature Importances')
```

GBM's feature importance = total reduction in loss from splits on each feature, summed across all trees.

---

## Gradient Boosting vs. Other Ensembles

| Property | Random Forest | AdaBoost | Gradient Boosting |
|----------|--------------|---------|-----------------|
| Strategy | Bagging | Boosting (weights) | Boosting (residuals) |
| Trees | Deep, parallel | Stumps, sequential | Shallow, sequential |
| Variance | Low | Moderate | Low |
| Bias | Moderate | Low | Low |
| Overfitting risk | Low | Moderate | High without tuning |
| Performance | Strong | Strong | Often state-of-art |

---

## Practical Tips

- Start with `learning_rate=0.1, max_depth=3, n_estimators=100` — a reliable baseline.
- Use early stopping to find the right `n_estimators` automatically.
- Lower `subsample` (0.5–0.8) adds regularization for noisy datasets.
- Gradient Boosting is sensitive to outliers (squared loss is used internally) — consider `loss='huber'` for robust regression.
- Feature scaling is NOT required — tree-based models are invariant to monotonic transformations.
- For datasets > 100K rows, switch to LightGBM for 10–100× faster training.
